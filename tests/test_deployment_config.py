from pathlib import Path


def test_production_dockerfile_exists_and_is_non_root():
    path = Path("deploy/Dockerfile.api")
    text = path.read_text(encoding="utf-8")
    assert "USER modelxray" in text
    assert "HEALTHCHECK" in text


def test_production_compose_has_hardening():
    path = Path("deploy/docker-compose.prod.yml")
    text = path.read_text(encoding="utf-8")
    for token in ["read_only: true", "no-new-privileges:true", "cap_drop:", "mem_limit:", "cpus:"]:
        assert token in text
