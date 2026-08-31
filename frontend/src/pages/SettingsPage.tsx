import { Check, CheckCircle2, RefreshCw, Save, TriangleAlert } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import { useLocation } from "react-router-dom";

import { Page, PageHeader, PageSection } from "../components/layout/Page";
import { YamSetupPanel } from "../components/YamSetupPanel";
import { Alert } from "../components/ui/Alert";
import { Badge } from "../components/ui/Badge";
import { Button } from "../components/ui/Button";
import { CodeBlock, InlineCode } from "../components/ui/Code";
import { Field, Select, TextInput } from "../components/ui/Form";
import { Panel, PanelHeader, SectionHeading } from "../components/ui/Panel";
import { Skeleton } from "../components/ui/Skeleton";
import { TabPanel, Tabs } from "../components/ui/Tabs";
import { savePublicSettings, type PublicSettings, type SettingsStatus } from "../lib/api";
import { usePublicSettings } from "../hooks/usePublicSettings";
import { useSystemStatus } from "../state/systemStatus";

type SettingsTab = "connections" | "robots" | "credentials" | "defaults";

const READY_STATES = new Set(["connected", "configured"]);

function ServiceRows({ status }: { status: SettingsStatus }) {
  return (
    <ul className="divide-y divide-line-subtle overflow-hidden rounded-xl border border-line bg-surface">
      {status.services.map((service) => {
        const ready = READY_STATES.has(service.status);
        return (
          <li key={service.id} className="flex items-start justify-between gap-4 px-5 py-4">
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                {ready ? (
                  <CheckCircle2 className="h-4 w-4 shrink-0 text-positive-600" aria-hidden="true" />
                ) : (
                  <TriangleAlert
                    className={`h-4 w-4 shrink-0 ${service.required ? "text-caution-600" : "text-ink-faint"}`}
                    aria-hidden="true"
                  />
                )}
                <p className="text-[13px] font-medium text-ink">{service.label}</p>
                {!service.required && (
                  <span className="text-2xs text-ink-faint">Optional</span>
                )}
              </div>
              <p className="mt-1 text-xs leading-5 text-ink-muted">{service.detail}</p>
            </div>
            <Badge tone={ready ? "success" : service.required ? "warning" : "neutral"}>
              {service.status}
            </Badge>
          </li>
        );
      })}
    </ul>
  );
}

function CredentialGuide({ status }: { status: SettingsStatus }) {
  const readiness = status.inference;
  const postgres = status.services.find((service) => service.id === "postgres");
  const items = [
    {
      label: "PostgreSQL",
      variables: "DATABASE_URL",
      ready: postgres ? READY_STATES.has(postgres.status) : false,
      detail:
        "The control-plane database. Any PostgreSQL 14+ endpoint works, including a hosted Supabase, Neon, or RDS connection string.",
      example: "DATABASE_URL=postgresql://ctrl_pi:password@db.example.internal:5432/ctrl_pi",
    },
    {
      label: "Hugging Face model and dataset access",
      variables: "HF_TOKEN + HF_NAMESPACE",
      ready: readiness.hf_configured,
      detail:
        "The bounded whoami check must identify the exact configured user or organization namespace.",
      example: "HF_TOKEN=hf_xxx\nHF_NAMESPACE=your-org",
    },
    {
      label: "Modal API credentials",
      variables: "MODAL_TOKEN_ID + MODAL_TOKEN_SECRET",
      ready: readiness.modal_configured,
      detail:
        "Requires one complete environment pair or the selected complete Modal profile; this is not a live API call.",
      example: "MODAL_TOKEN_ID=ak-xxx\nMODAL_TOKEN_SECRET=as-xxx",
    },
    {
      label: "Modal proxy tokens",
      variables: "MODAL_PROXY_TOKEN_ID + MODAL_PROXY_TOKEN_SECRET",
      ready: readiness.modal_proxy_configured,
      detail:
        "Requires a complete, valid wk-/ws- pair for backend-only endpoint traffic. API credentials cannot be used as proxy credentials.",
      example: "MODAL_PROXY_TOKEN_ID=wk-xxx\nMODAL_PROXY_TOKEN_SECRET=ws-xxx",
    },
  ];

  return (
    <div className="space-y-6">
      <Alert tone="info" title="Keys are set in the backend environment, never in this browser">
        ctrl-π runs on the machine attached to the robot and deliberately has no endpoint that
        accepts a secret from the browser. Add each value to the gitignored{" "}
        <InlineCode>.env</InlineCode> file next to <InlineCode>docker-compose.yml</InlineCode>,
        restart the service, then use <strong>Recheck</strong> here. This page only ever shows
        whether a pair is present.
      </Alert>

      <div className="space-y-5">
        {items.map((item) => (
          <Panel key={item.label}>
            <PanelHeader
              title={item.label}
              description={item.detail}
              actions={
                <Badge
                  tone={item.ready ? "success" : readiness.mock_mode ? "info" : "warning"}
                >
                  {item.ready ? "Ready" : readiness.mock_mode ? "Optional in mock" : "Missing"}
                </Badge>
              }
            />
            <div className="px-5 py-4">
              <p className="mb-3 font-mono text-2xs text-ink-muted">{item.variables}</p>
              <CodeBlock caption=".env" code={item.example} />
            </div>
          </Panel>
        ))}
      </div>
    </div>
  );
}

function PreferencesForm() {
  const { settings, error: loadError, setSettings } = usePublicSettings();
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

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

  if (!settings) {
    return loadError ? (
      <Alert tone="danger" title="Preferences unavailable">
        {loadError}
      </Alert>
    ) : (
      <Skeleton className="h-64 w-full" />
    );
  }

  return (
    <Panel>
      <PanelHeader
        title="Non-secret defaults"
        description="Operational preferences stored in PostgreSQL. Credentials stay in .env."
      />
      <form onSubmit={submit} className="space-y-6 px-5 py-5">
        <div className="grid gap-5 sm:grid-cols-2">
          <Field label="Hugging Face namespace" hint="Read from HF_NAMESPACE.">
            <TextInput value={settings.hf_namespace ?? "Not configured"} disabled />
          </Field>
          <Field label="Recording FPS">
            <TextInput
              type="number"
              min={1}
              max={60}
              value={settings.recording_fps}
              onChange={(event) =>
                setSettings({ ...settings, recording_fps: Number(event.target.value) })
              }
            />
          </Field>
          <Field label="Default runtime">
            <Select
              value={settings.default_runtime}
              onChange={(event) =>
                setSettings({
                  ...settings,
                  default_runtime: event.target.value as PublicSettings["default_runtime"],
                })
              }
            >
              <option value="lerobot">LeRobot</option>
              <option value="openpi">OpenPI</option>
            </Select>
          </Field>
          <Field label="Default compute">
            <Select
              value={settings.default_compute}
              onChange={(event) =>
                setSettings({
                  ...settings,
                  default_compute: event.target.value as PublicSettings["default_compute"],
                })
              }
            >
              <option>Modal: A10G</option>
              <option>Modal: A100</option>
              <option>Modal: H100</option>
            </Select>
          </Field>
          <Field label="Deployment timeout (minutes)">
            <TextInput
              type="number"
              min={1}
              max={30}
              value={settings.modal_timeout_minutes}
              onChange={(event) =>
                setSettings({ ...settings, modal_timeout_minutes: Number(event.target.value) })
              }
            />
          </Field>
        </div>

        {error && <Alert tone="danger">{error}</Alert>}

        <div className="flex items-center justify-end gap-3 border-t border-line pt-5">
          {saved && (
            <span className="flex items-center gap-1.5 text-xs font-medium text-positive-700">
              <Check className="h-3.5 w-3.5" aria-hidden="true" /> Saved
            </span>
          )}
          <Button type="submit" variant="primary" icon={Save} loading={saving}>
            Save defaults
          </Button>
        </div>
      </form>
    </Panel>
  );
}

export function SettingsPage() {
  const { status, loading, error, refresh } = useSystemStatus();
  const location = useLocation();
  const [tab, setTab] = useState<SettingsTab>(
    location.hash === "#yam-setup" ? "robots" : "connections",
  );
  const focusedYamHash = useRef(false);

  useEffect(() => {
    if (location.hash !== "#yam-setup") {
      focusedYamHash.current = false;
      return;
    }
    setTab("robots");
    if (!status || focusedYamHash.current) return;
    focusedYamHash.current = true;
    const frame = window.requestAnimationFrame(() => {
      const target = document.getElementById("yam-setup");
      // Only scroll when the panel is actually off-screen; it is already the
      // first thing under the Robots tab on a normal viewport.
      const bounds = target?.getBoundingClientRect();
      if (bounds && (bounds.top < 0 || bounds.top > window.innerHeight)) {
        target?.scrollIntoView({ block: "start" });
      }
      target?.focus({ preventScroll: true });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [location.hash, status]);

  return (
    <Page>
      <PageHeader
        title="Settings"
        description="Service connections, robot cell configuration, credential readiness, and non-secret defaults."
        meta={
          status ? (
            <>
              <Badge tone={status.setup_complete ? "success" : "warning"} dot>
                {status.setup_complete ? "Setup complete" : "Setup needs attention"}
              </Badge>
              <Badge tone={status.mode === "mock" ? "info" : "neutral"}>{status.mode} mode</Badge>
            </>
          ) : undefined
        }
        actions={
          <Button icon={RefreshCw} loading={loading} disabled={loading} onClick={refresh}>
            Recheck
          </Button>
        }
      />

      {error ? (
        <PageSection className="mt-8">
          <Alert tone="danger" title="The backend is unavailable">
            {error} Start it on port 8000, then recheck.
          </Alert>
        </PageSection>
      ) : !status ? (
        <PageSection className="mt-8">
          <Skeleton className="h-64 w-full" />
        </PageSection>
      ) : (
        <PageSection className="mt-8">
          <Tabs
            label="Settings sections"
            value={tab}
            onChange={setTab}
            items={[
              { id: "connections", label: "Connections" },
              { id: "robots", label: "Robots" },
              { id: "credentials", label: "Credentials" },
              { id: "defaults", label: "Defaults" },
            ]}
          />

          <div className="mt-6">
            <TabPanel id="connections" active={tab === "connections"}>
              <div className="space-y-6">
                <SectionHeading
                  title="First-run checklist"
                  description={
                    status.mode === "mock"
                      ? "PostgreSQL and mock arms are required; Hugging Face and Modal remain optional until you use their cloud workflows."
                      : "Hardware mode requires PostgreSQL, verified Hugging Face namespace access, Modal API and proxy credentials, and connected arms."
                  }
                />
                <ServiceRows status={status} />
                <p className="text-xs leading-5 text-ink-muted">
                  Secrets stay in the backend environment and are never sent to this browser.
                </p>
              </div>
            </TabPanel>

            <TabPanel id="robots" active={tab === "robots"} keepMounted>
              <YamSetupPanel onSettingsRefresh={refresh} />
            </TabPanel>

            <TabPanel id="credentials" active={tab === "credentials"}>
              <CredentialGuide status={status} />
            </TabPanel>

            <TabPanel id="defaults" active={tab === "defaults"}>
              <PreferencesForm />
            </TabPanel>
          </div>
        </PageSection>
      )}
    </Page>
  );
}
