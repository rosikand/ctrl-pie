import {
  Bot,
  Check,
  CheckCircle2,
  Cloud,
  Database,
  KeyRound,
  RefreshCw,
  Save,
  Server,
  TriangleAlert,
} from "lucide-react";
import type { FormEvent } from "react";
import { useEffect, useState } from "react";

import {
  fetchPublicSettings,
  savePublicSettings,
  type PublicSettings,
  type ServiceStatus,
  type SettingsStatus,
} from "../lib/api";

const serviceIcons = {
  postgres: Database,
  huggingface: Cloud,
  modal: Server,
  arms: Bot,
};

const readyStates = new Set(["connected", "configured"]);

function ConnectionCard({ service }: { service: ServiceStatus }) {
  const Icon = serviceIcons[service.id];
  const ready = readyStates.has(service.status);

  return (
    <article className="rounded-xl border border-slate-200 bg-white p-5 shadow-panel">
      <div className="flex items-start justify-between gap-4">
        <div className="grid h-9 w-9 place-items-center rounded-lg bg-slate-100 text-slate-600">
          <Icon className="h-[18px] w-[18px]" strokeWidth={1.8} />
        </div>
        <span
          className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-semibold capitalize ${
            ready ? "bg-emerald-50 text-emerald-700" : "bg-amber-50 text-amber-700"
          }`}
        >
          <span className={`h-1.5 w-1.5 rounded-full ${ready ? "bg-emerald-500" : "bg-amber-500"}`} />
          {service.status}
        </span>
      </div>
      <h3 className="mt-4 text-sm font-semibold text-slate-900">{service.label}</h3>
      <p className="mt-1 min-h-10 text-xs leading-5 text-slate-500">{service.detail}</p>
    </article>
  );
}

function Checklist({ status }: { status: SettingsStatus }) {
  return (
    <section className="rounded-xl border border-slate-200 bg-white shadow-panel">
      <div className="border-b border-slate-100 px-5 py-4 sm:px-6">
        <h2 className="text-sm font-semibold text-slate-900">First-run checklist</h2>
        <p className="mt-1 text-xs leading-5 text-slate-500">
          Secrets stay in the backend environment and are never sent to this browser.
        </p>
      </div>
      <ul className="divide-y divide-slate-100">
        {status.services.map((service) => {
          const ready = readyStates.has(service.status);
          return (
            <li key={service.id} className="flex items-start gap-3 px-5 py-4 sm:px-6">
              {ready ? (
                <CheckCircle2 className="mt-0.5 h-[18px] w-[18px] shrink-0 text-emerald-500" />
              ) : (
                <TriangleAlert className="mt-0.5 h-[18px] w-[18px] shrink-0 text-amber-500" />
              )}
              <div>
                <p className="text-sm font-medium text-slate-800">
                  {service.label}
                  {!service.required && <span className="ml-2 text-xs font-normal text-slate-400">Optional</span>}
                </p>
                <p className="mt-0.5 text-xs leading-5 text-slate-500">{service.detail}</p>
              </div>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

function InferenceCredentials({ status }: { status: SettingsStatus }) {
  const readiness = status.inference;
  if (!readiness) return null;
  const items = [
    {
      label: "Hugging Face model access",
      variables: "HF_TOKEN + HF_NAMESPACE",
      ready: readiness.hf_configured,
      detail: "Resolves configured-namespace models to an exact revision before deployment.",
    },
    {
      label: "Modal API credentials",
      variables: "MODAL_TOKEN_ID + MODAL_TOKEN_SECRET",
      ready: readiness.modal_configured,
      detail: "Creates, inspects, and tears down the owned Modal application.",
    },
    {
      label: "Modal proxy tokens",
      variables: "MODAL_PROXY_TOKEN_ID + MODAL_PROXY_TOKEN_SECRET",
      ready: readiness.modal_proxy_configured,
      detail: "Authenticates backend-only health and inference traffic to the protected endpoint.",
    },
  ];
  return (
    <section className="mt-6 rounded-xl border border-slate-200 bg-white shadow-panel">
      <div className="border-b border-slate-100 px-5 py-4 sm:px-6">
        <div className="flex items-center gap-2">
          <KeyRound className="h-4 w-4 text-slate-400" />
          <h2 className="text-sm font-semibold text-slate-900">Inference credential readiness</h2>
        </div>
        <p className="mt-1 text-xs leading-5 text-slate-500">
          These checks expose only pair readiness. Credential values remain in the backend environment.
        </p>
      </div>
      <div className="grid gap-px bg-slate-100 lg:grid-cols-3">
        {items.map((item) => (
          <article key={item.label} className="bg-white px-5 py-4 sm:px-6">
            <div className="flex items-start justify-between gap-3">
              <h3 className="text-xs font-semibold text-slate-800">{item.label}</h3>
              <span className={`shrink-0 rounded-full px-2 py-1 text-[10px] font-semibold ${item.ready ? "bg-emerald-50 text-emerald-700" : readiness.mock_mode ? "bg-blue-50 text-blue-700" : "bg-amber-50 text-amber-700"}`}>
                {item.ready ? "Ready" : readiness.mock_mode ? "Optional in mock" : "Missing"}
              </span>
            </div>
            <p className="mt-2 text-[11px] leading-5 text-slate-500">{item.detail}</p>
            {!item.ready && (
              <p className="mt-2 font-mono text-[10px] leading-4 text-slate-400">
                Set {item.variables} in <code>.env</code>.
              </p>
            )}
          </article>
        ))}
      </div>
    </section>
  );
}

function PreferencesForm() {
  const [settings, setSettings] = useState<PublicSettings | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    void fetchPublicSettings()
      .then(setSettings)
      .catch((reason: unknown) => {
        setError(reason instanceof Error ? reason.message : "Could not load settings.");
      });
  }, []);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!settings) return;
    setSaving(true);
    setSaved(false);
    setError(null);
    try {
      const updated = await savePublicSettings({
        recording_fps: settings.recording_fps,
        default_runtime: settings.default_runtime,
        default_compute: settings.default_compute,
        modal_timeout_minutes: settings.modal_timeout_minutes,
      });
      setSettings(updated);
      setSaved(true);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not save settings.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="rounded-xl border border-slate-200 bg-white shadow-panel">
      <div className="border-b border-slate-100 px-5 py-4 sm:px-6">
        <h2 className="text-sm font-semibold text-slate-900">Non-secret defaults</h2>
        <p className="mt-1 text-xs leading-5 text-slate-500">
          Operational preferences are stored in PostgreSQL. Credentials remain in <code>.env</code>.
        </p>
      </div>
      {!settings ? (
        <div className="px-5 py-6 text-sm text-slate-500 sm:px-6">
          {error ?? "Loading preferences…"}
        </div>
      ) : (
        <form onSubmit={submit} className="space-y-5 px-5 py-5 sm:px-6">
          <div className="grid gap-5 sm:grid-cols-2">
            <label className="block text-xs font-medium text-slate-700">
              Hugging Face namespace
              <input
                value={settings.hf_namespace ?? "Not configured"}
                disabled
                className="mt-2 w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-500"
              />
              <span className="mt-1.5 block font-normal text-slate-400">Read from HF_NAMESPACE.</span>
            </label>
            <label className="block text-xs font-medium text-slate-700">
              Recording FPS
              <input
                type="number"
                min={1}
                max={60}
                value={settings.recording_fps}
                onChange={(event) => setSettings({ ...settings, recording_fps: Number(event.target.value) })}
                className="mt-2 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none ring-brand-100 focus:border-brand-500 focus:ring-4"
              />
            </label>
            <label className="block text-xs font-medium text-slate-700">
              Default runtime
              <select
                value={settings.default_runtime}
                onChange={(event) =>
                  setSettings({ ...settings, default_runtime: event.target.value as PublicSettings["default_runtime"] })
                }
                className="mt-2 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm outline-none ring-brand-100 focus:border-brand-500 focus:ring-4"
              >
                <option value="lerobot">LeRobot</option>
                <option value="openpi">OpenPI</option>
              </select>
            </label>
            <label className="block text-xs font-medium text-slate-700">
              Default compute
              <select
                value={settings.default_compute}
                onChange={(event) =>
                  setSettings({ ...settings, default_compute: event.target.value as PublicSettings["default_compute"] })
                }
                className="mt-2 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm outline-none ring-brand-100 focus:border-brand-500 focus:ring-4"
              >
                <option>Modal: A10G</option>
                <option>Modal: A100</option>
                <option>Modal: H100</option>
              </select>
            </label>
            <label className="block text-xs font-medium text-slate-700 sm:col-span-2">
              Deployment timeout (minutes)
              <input
                type="number"
                min={1}
                max={30}
                value={settings.modal_timeout_minutes}
                onChange={(event) =>
                  setSettings({ ...settings, modal_timeout_minutes: Number(event.target.value) })
                }
                className="mt-2 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none ring-brand-100 focus:border-brand-500 focus:ring-4 sm:max-w-[calc(50%-0.625rem)]"
              />
            </label>
          </div>
          {error && <p className="text-xs text-rose-600">{error}</p>}
          <div className="flex items-center justify-end gap-3 border-t border-slate-100 pt-4">
            {saved && (
              <span className="flex items-center gap-1.5 text-xs font-medium text-emerald-600">
                <Check className="h-3.5 w-3.5" /> Saved
              </span>
            )}
            <button
              type="submit"
              disabled={saving}
              className="inline-flex items-center gap-2 rounded-lg bg-ink px-3.5 py-2 text-xs font-semibold text-white transition hover:bg-slate-700 disabled:opacity-50"
            >
              <Save className="h-3.5 w-3.5" />
              {saving ? "Saving…" : "Save defaults"}
            </button>
          </div>
        </form>
      )}
    </section>
  );
}

export function SettingsPage({
  status,
  loading,
  error,
  refresh,
}: {
  status: SettingsStatus | null;
  loading: boolean;
  error: string | null;
  refresh: () => void;
}) {
  return (
    <div className="mx-auto max-w-7xl px-5 py-8 sm:px-8 lg:px-10 lg:py-10">
      <header className="flex items-start justify-between gap-6">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-brand-600">Configuration</p>
          <h1 className="mt-2 text-2xl font-semibold tracking-tight text-slate-950 sm:text-3xl">Settings</h1>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">
            Check local service connections and configure non-secret defaults.
          </p>
        </div>
        <button
          type="button"
          onClick={refresh}
          disabled={loading}
          className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-600 shadow-sm hover:bg-slate-50 disabled:opacity-50"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
          Recheck
        </button>
      </header>

      {error ? (
        <div className="mt-8 rounded-xl border border-rose-200 bg-rose-50 p-5 text-sm text-rose-800">
          {error} Start the backend on port 8000, then recheck.
        </div>
      ) : status ? (
        <>
          <div className="mt-8 flex items-center gap-2 text-sm font-medium text-slate-700">
            {status.setup_complete ? (
              <CheckCircle2 className="h-4 w-4 text-emerald-500" />
            ) : (
              <TriangleAlert className="h-4 w-4 text-amber-500" />
            )}
            {status.setup_complete ? "Setup complete" : "Setup needs attention"}
            <span className="ml-1 rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-slate-500">
              {status.mode} mode
            </span>
          </div>
          <div className="mt-4 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            {status.services.map((service) => (
              <ConnectionCard key={service.id} service={service} />
            ))}
          </div>
          <InferenceCredentials status={status} />
          <div className="mt-6 grid gap-6 xl:grid-cols-2">
            <Checklist status={status} />
            <PreferencesForm />
          </div>
        </>
      ) : (
        <div className="mt-8 rounded-xl border border-slate-200 bg-white p-8 text-sm text-slate-500 shadow-panel">
          Checking service connections…
        </div>
      )}
    </div>
  );
}
