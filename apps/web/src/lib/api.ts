export const API_BASE = (
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
).replace(/\/$/, "");

export function errorMessage(error: unknown, fallback = "The request could not be completed.") {
  if (!(error instanceof Error)) return fallback;
  if (/failed to fetch|fetch failed|networkerror|network request failed|load failed/i.test(error.message))
    return "Cannot reach the ModelXray API. Check that the service is running, then retry.";
  return error.message || fallback;
}

export async function readApiError(response: Response) {
  try {
    const payload = await response.json();
    const detail = payload?.detail;
    if (typeof detail === "string") return detail;
    if (detail?.message) return String(detail.message);
    if (Array.isArray(detail))
      return detail
        .map((item) => `${item.loc?.join(".") ?? "Request"}: ${item.msg}`)
        .join("; ");
  } catch {
    /* Non-JSON server response. */
  }
  return `Request failed (HTTP ${response.status})`;
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    cache: "no-store",
  });
  if (!response.ok) throw new Error(await readApiError(response));
  return response.json() as Promise<T>;
}

// Browser persistence can be disabled by the user. The workspace still works.
export const storage = {
  get(key: string) {
    try {
      return localStorage.getItem(`${API_BASE}:${key}`);
    } catch {
      return null;
    }
  },
  set(key: string, value: string) {
    try {
      localStorage.setItem(`${API_BASE}:${key}`, value);
    } catch {
      /* optional persistence */
    }
  },
  remove(key: string) {
    try {
      localStorage.removeItem(`${API_BASE}:${key}`);
    } catch {
      /* optional persistence */
    }
  },
};
