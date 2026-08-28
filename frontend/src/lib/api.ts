export type ServiceStatus = {
  id: "postgres" | "huggingface" | "modal" | "arms";
  label: string;
  status: "connected" | "configured" | "missing" | "error";
  detail: string;
  required: boolean;
};

export type SettingsStatus = {
  mode: "mock" | "hardware";
  setup_complete: boolean;
  services: ServiceStatus[];
};

export type PublicSettings = {
  hf_namespace: string | null;
  recording_fps: number;
  default_runtime: "lerobot" | "openpi";
  default_compute: "Modal: A10G" | "Modal: A100" | "Modal: H100";
  modal_timeout_minutes: number;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(payload?.detail ?? `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export function fetchSettingsStatus(): Promise<SettingsStatus> {
  return request<SettingsStatus>("/api/settings/status");
}

export function fetchPublicSettings(): Promise<PublicSettings> {
  return request<PublicSettings>("/api/settings");
}

export function savePublicSettings(
  settings: Omit<PublicSettings, "hf_namespace">,
): Promise<PublicSettings> {
  return request<PublicSettings>("/api/settings", {
    method: "PATCH",
    body: JSON.stringify(settings),
  });
}

