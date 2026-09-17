export const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");
const SESSION_KEY = "dfg.session";
const TOKEN_KEY = "dfg.token";

/** API token (only needed when the backend sets DFG_API_TOKENS). */
export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token: string | null) {
  try {
    if (token) window.localStorage.setItem(TOKEN_KEY, token);
    else window.localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* storage unavailable */
  }
}

export class ApiError extends Error {
  status: number;
  details: unknown;
  constructor(status: number, message: string, details?: unknown) {
    super(message);
    this.status = status;
    this.details = details;
  }
}

export function getSessionId(): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(SESSION_KEY);
  } catch {
    return null;
  }
}

export function setSessionId(id: string | null) {
  try {
    if (id) window.localStorage.setItem(SESSION_KEY, id);
    else window.localStorage.removeItem(SESSION_KEY);
  } catch {
    /* storage unavailable */
  }
}

async function handle<T>(res: Response): Promise<T> {
  const sid = res.headers.get("X-Session-ID");
  if (sid) setSessionId(sid);
  const type = res.headers.get("content-type") || "";
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    let details: unknown;
    if (type.includes("application/json")) {
      const body = await res.json();
      message = body.message || message;
      details = body.details;
    }
    throw new ApiError(res.status, message, details);
  }
  if (type.includes("application/json")) return (await res.json()) as T;
  return (await res.text()) as unknown as T;
}

function headers(extra?: Record<string, string>): Record<string, string> {
  const h: Record<string, string> = { ...(extra || {}) };
  const sid = getSessionId();
  if (sid) h["X-Session-ID"] = sid;
  const token = getToken();
  if (token) h["Authorization"] = `Bearer ${token}`;
  return h;
}

export const api = {
  get: async <T,>(path: string): Promise<T> => handle<T>(await fetch(`${API_URL}${path}`, { headers: headers(), cache: "no-store" })),
  post: async <T,>(path: string, body?: unknown): Promise<T> =>
    handle<T>(
      await fetch(`${API_URL}${path}`, {
        method: "POST",
        headers: headers({ "Content-Type": "application/json" }),
        body: body === undefined ? undefined : JSON.stringify(body),
      }),
    ),
  del: async <T,>(path: string): Promise<T> => handle<T>(await fetch(`${API_URL}${path}`, { method: "DELETE", headers: headers() })),
  upload: async <T,>(files: File[]): Promise<T> => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    return handle<T>(await fetch(`${API_URL}/datasets/upload`, { method: "POST", headers: headers(), body: form }));
  },
  downloadUrl: (file: string) => {
    const sid = getSessionId();
    const token = getToken();
    return `${API_URL}/integration/export?file=${encodeURIComponent(file)}${sid ? `&session_id=${sid}` : ""}${token ? `&token=${encodeURIComponent(token)}` : ""}`;
  },
};
