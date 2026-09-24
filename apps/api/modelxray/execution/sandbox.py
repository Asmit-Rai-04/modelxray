from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from threading import Event
from typing import Any

WORKER_PROTOCOL_VERSION = 2


class SandboxExecutionError(RuntimeError):
    """Worker-level execution failure.

    Attributes:
        is_timeout: True when the failure was a wall-clock timeout.
        is_environment: True when the worker could not start for environment
            reasons (missing packages, broken interpreter) as opposed to a
            user/model contract problem.
    """

    def __init__(self, message: str, *, is_timeout: bool = False, is_environment: bool = False, is_cancelled: bool = False) -> None:
        super().__init__(message)
        self.is_timeout = is_timeout
        self.is_environment = is_environment
        self.is_cancelled = is_cancelled


def _limit_resources() -> None:
    """Apply best-effort POSIX limits in the child process.

    Runs via preexec_fn. To stay fork-safe when the parent is threaded (uvicorn's
    threadpool runs sync handlers), only allocation-free C-level calls are used
    here: `resource` is imported in the parent before forking and getenv is the
    C-level call. No Python-level logging, imports, or I/O from the child.
    """
    import resource  # imported in the parent before fork

    cpu_seconds = int(os.getenv("MODELXRAY_WORKER_CPU_SECONDS", "120"))
    memory_bytes = int(os.getenv("MODELXRAY_WORKER_MEMORY_MB", "1536")) * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 5))
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024 * 1024, 256 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
    except (OSError, ValueError):
        # Restricted containers may refuse some limits; keep the rest.
        pass


def _environment_summary() -> str:
    import importlib.metadata

    packages = ("scikit-learn", "numpy", "pandas", "scipy", "joblib", "statsmodels", "modelxray")
    lines = []
    for name in packages:
        try:
            lines.append(f"{name}=={importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            lines.append(f"{name}: NOT INSTALLED")
    return "; ".join(lines)


def _classify_worker_failure(stderr_text: str) -> str | None:
    """Map worker-import failures to an actionable environment message."""
    lowered = stderr_text.lower()
    if "no module named" in lowered or "modulenotfounderror" in lowered:
        return (
            "The isolated worker could not import its dependencies. The worker uses the "
            "same interpreter and environment as the API process; install the project "
            f"dependencies in that environment. Environment: {_environment_summary()}"
        )
    if "dll load failed" in lowered:
        return (
            "The isolated worker failed to load native libraries. The worker uses the "
            "same interpreter as the API process; the scientific stack appears broken "
            f"for this interpreter. Environment: {_environment_summary()}"
        )
    return None


def run_isolated_investigation(
    *,
    model_path: str | Path,
    dataset_path: str | Path,
    target_column: str,
    budget: int,
    random_state: int,
    timeout_seconds: int = 180,
    metric: str = "accuracy",
    feature_manifest: list[str] | None = None,
    extra_pythonpath: list[str] | str | None = None,
    cancel_event: Event | None = None,
) -> dict[str, Any]:
    """Run model deserialization + inference in a separate worker process.

    Security note: this is defense-in-depth, not a sandbox. Pickle/joblib remains
    executable code; the worker runs with this process's privileges (plus POSIX
    rlimits where available). A public multi-tenant deployment must place the
    worker in a restricted container/VM with network isolation and a dedicated
    unprivileged user.

    Result transport: the worker writes its JSON result to a unique tempfile and
    prints only the result path on stdout. Worker stdout is therefore safe for
    arbitrary model logging (models that print during predict() cannot corrupt
    the protocol).
    """
    package_root = Path(__file__).resolve().parents[2]
    # The worker runs with cwd=package_root; resolve user-supplied asset paths
    # against THIS process's cwd so relative upload paths stay valid.
    model_path = Path(model_path)
    dataset_path = Path(dataset_path)
    if not model_path.is_absolute():
        model_path = (Path.cwd() / model_path).resolve()
    if not dataset_path.is_absolute():
        dataset_path = (Path.cwd() / dataset_path).resolve()
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [str(package_root)]
    if extra_pythonpath:
        # Model-artifact imports (e.g. custom classes defined next to the model
        # file) must be importable by the worker at unpickle time.
        extras = [extra_pythonpath] if isinstance(extra_pythonpath, str) else list(extra_pythonpath)
        pythonpath_parts.extend(extras)
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    # NOTE: deliberately NOT setting PYTHONNOUSERSITE. The worker must resolve
    # packages from the same environment/context as this process, including
    # user site-packages. Removing this flag fixes investigations on standard
    # `pip install --user` setups (previously: ModuleNotFoundError in worker).
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MODELXRAY_SANDBOX_WORKER": "1",
        }
    )

    result_handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="mx_result_",
        delete=False,
        encoding="utf-8",
    )
    result_handle.close()
    result_path = Path(result_handle.name)

    cmd = [
        sys.executable,
        "-m",
        "modelxray.execution.worker",
        "--model",
        str(model_path),
        "--dataset",
        str(dataset_path),
        "--target",
        target_column,
        "--budget",
        str(budget),
        "--random-state",
        str(random_state),
        "--metric",
        metric,
        "--feature-manifest",
        json.dumps(feature_manifest) if feature_manifest else "",
        "--result-path",
        str(result_path),
    ]

    preexec = _limit_resources if os.name == "posix" else None

    process = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=str(package_root),
            preexec_fn=preexec,
        )
        start = time.monotonic()
        while True:
            if cancel_event is not None and cancel_event.is_set():
                process.terminate()
                try:
                    stdout, stderr = process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate()
                result_path.unlink(missing_ok=True)
                raise SandboxExecutionError("Model execution was cancelled.", is_cancelled=True)
            if time.monotonic() - start > timeout_seconds:
                process.kill()
                stdout, stderr = process.communicate()
                result_path.unlink(missing_ok=True)
                raise SandboxExecutionError(
                    f"Model execution exceeded the {timeout_seconds}s worker timeout.",
                    is_timeout=True,
                )
            return_code = process.poll()
            if return_code is not None:
                stdout, stderr = process.communicate()
                break
            time.sleep(0.10)
    except SandboxExecutionError:
        raise
    except OSError as exc:
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate()
        result_path.unlink(missing_ok=True)
        raise SandboxExecutionError(f"Could not start the isolated worker: {exc}", is_environment=True) from exc
    completed = subprocess.CompletedProcess(cmd, process.returncode if process is not None else -1, stdout or "", stderr or "")

    def _fail(message: str, **kwargs: bool) -> SandboxExecutionError:
        result_path.unlink(missing_ok=True)
        return SandboxExecutionError(message, **kwargs)

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        env_hint = _classify_worker_failure(stderr)
        if env_hint:
            raise _fail(env_hint, is_environment=True)
        # Worker protocol: a nonzero exit with a JSON {"error": ...} on the last
        # stdout line is a reported error (contract/contract-adjacent); anything
        # else is an internal worker failure.
        stdout_lines = [line for line in completed.stdout.strip().splitlines() if line.strip()]
        for line in reversed(stdout_lines[-5:]):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and "error" in payload:
                raise _fail(str(payload["error"]))
        detail = stderr or "worker failed without details"
        raise _fail(f"Internal worker failure: {detail[-2000:]}")

    if not result_path.exists():
        raise _fail(
            "Worker did not produce a result file. "
            f"Worker stdout tail: {completed.stdout.strip()[-500:]!r}",
            is_environment=True,
        )

    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise _fail(
            "Worker result file contained invalid JSON. "
            f"Worker stdout tail: {completed.stdout.strip()[-500:]!r}",
        ) from exc
    finally:
        result_path.unlink(missing_ok=True)

    protocol = payload.get("protocol_version")
    if protocol != WORKER_PROTOCOL_VERSION:
        raise _fail(
            f"Worker protocol mismatch: expected v{WORKER_PROTOCOL_VERSION}, got {protocol!r}. "
            "The API and worker installations appear to be out of sync."
        )
    if "error" in payload:
        raise _fail(str(payload["error"]))
    return payload["result"]
