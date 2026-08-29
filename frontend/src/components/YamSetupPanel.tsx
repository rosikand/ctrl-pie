import {
  Bot,
  Cable,
  Check,
  CheckCircle2,
  CircleDashed,
  FileCheck2,
  LoaderCircle,
  PlugZap,
  Radar,
  RefreshCw,
  RotateCcw,
  Save,
  ShieldAlert,
  Trash2,
  TriangleAlert,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { useYamSetup } from "../hooks/useYamSetup";
import type {
  YamSetupCandidate,
  YamSetupConfig,
  YamSetupStatus,
} from "../types/yamSetup";

const EMPTY_CONFIG: YamSetupConfig = {
  can_interface: "",
  leader_port: "",
  mujoco_xml_path: "",
  leader_calibration_id: "yam-leader",
  leader_calibration_dir: "",
};

function formatTimestamp(value: string | null): string | null {
  if (!value) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date.toLocaleString();
}

function setupLabel(setup: YamSetupStatus): string {
  if (setup.mode === "mock") return "Mock rig ready";
  if (setup.connected) return setup.restored_on_boot ? "Restored and connected" : "Connected";
  if (setup.state === "ready_to_connect") return "Ready to connect";
  if (setup.state === "awaiting_hardware") {
    return setup.saved ? "Waiting for saved devices" : "Configured hardware missing";
  }
  if (setup.state === "error") return "Needs attention";
  return "Setup required";
}

function statusTone(setup: YamSetupStatus): string {
  if (setup.mode === "mock") return "bg-blue-50 text-blue-700";
  if (setup.connected) return "bg-emerald-50 text-emerald-700";
  if (setup.state === "error") return "bg-rose-50 text-rose-700";
  return "bg-amber-50 text-amber-700";
}

function SetupMilestones({ setup }: { setup: YamSetupStatus }) {
  const milestones = [
    {
      title: "Configuration",
      detail: setup.configured ? "A complete single-rig configuration is selected." : "Discover or enter one YAM pair.",
      ready: setup.configured,
      Icon: Cable,
    },
    {
      title: setup.mode === "mock" ? "Physical persistence" : "Saved setup",
      detail: setup.mode === "mock"
        ? "Not used in mock mode; any saved physical setup remains untouched."
        : setup.saved
          ? (setup.auto_restore ? "Saved with automatic boot restoration." : "Saved; automatic restoration is off.")
          : "Not yet saved to PostgreSQL.",
      ready: setup.mode === "mock" || setup.saved,
      Icon: Save,
    },
    {
      title: "Calibration file",
      detail: setup.calibration_ready ? "A bounded, structurally valid calibration artifact was detected." : "A readable, structurally valid leader calibration artifact is required.",
      ready: setup.calibration_ready,
      Icon: FileCheck2,
    },
    {
      title: "Connection",
      detail: setup.connected ? "The configured devices connected and returned a sample." : "No live device connection has been verified.",
      ready: setup.connected,
      Icon: PlugZap,
    },
  ];

  return (
    <div className="grid gap-px border-y border-slate-100 bg-slate-100 sm:grid-cols-2 xl:grid-cols-4">
      {milestones.map(({ title, detail, ready, Icon }) => (
        <article key={title} className="bg-white px-5 py-4 sm:px-6">
          <div className="flex items-center justify-between gap-3">
            <Icon className={`h-4 w-4 ${ready ? "text-emerald-500" : "text-slate-400"}`} aria-hidden="true" />
            <span className={`rounded-full px-2 py-1 text-[10px] font-semibold ${ready ? "bg-emerald-50 text-emerald-700" : "bg-slate-100 text-slate-500"}`}>
              {ready ? "Ready" : "Pending"}
            </span>
          </div>
          <h3 className="mt-3 text-xs font-semibold text-slate-800">{title}</h3>
          <p className="mt-1 text-[11px] leading-5 text-slate-500">{detail}</p>
        </article>
      ))}
    </div>
  );
}

function CandidateSummary({
  label,
  candidates,
}: {
  label: string;
  candidates: YamSetupCandidate[];
}) {
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2.5">
      <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-400">{label}</p>
      {candidates.length === 0 ? (
        <p className="mt-1 text-xs text-amber-700">None found; check host permissions and device passthrough, then rescan.</p>
      ) : (
        <p className="mt-1 text-xs text-slate-600">
          {candidates.length} candidate{candidates.length === 1 ? "" : "s"} found
          {candidates.length > 1 ? " — choose intentionally below." : "."}
        </p>
      )}
    </div>
  );
}

function ConfigInput({
  label,
  help,
  field,
  value,
  maxLength,
  candidates,
  onChange,
}: {
  label: string;
  help: string;
  field: keyof YamSetupConfig;
  value: string;
  maxLength: number;
  candidates?: YamSetupCandidate[];
  onChange: (field: keyof YamSetupConfig, value: string) => void;
}) {
  const listId = candidates?.length ? `yam-${field}-candidates` : undefined;
  return (
    <label className="block text-xs font-medium text-slate-700">
      {label}
      <input
        value={value}
        onChange={(event) => onChange(field, event.target.value)}
        maxLength={maxLength}
        required
        autoComplete="off"
        spellCheck={false}
        list={listId}
        className="mt-2 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none ring-brand-100 placeholder:text-slate-300 focus:border-brand-500 focus:ring-4"
      />
      {listId && (
        <datalist id={listId}>
          {candidates?.map((candidate) => (
            <option key={candidate.id} value={candidate.id}>{candidate.label}</option>
          ))}
        </datalist>
      )}
      <span className="mt-1.5 block font-normal leading-5 text-slate-400">{help}</span>
    </label>
  );
}

export function YamSetupPanel({ onSettingsRefresh }: { onSettingsRefresh: () => void }) {
  const yam = useYamSetup();
  const [config, setConfig] = useState<YamSetupConfig>(EMPTY_CONFIG);
  const [dirty, setDirty] = useState(false);
  const [autoRestore, setAutoRestore] = useState(false);
  const [autoRestoreDirty, setAutoRestoreDirty] = useState(false);
  const [safetyConfirmed, setSafetyConfirmed] = useState(false);
  const [confirmForget, setConfirmForget] = useState(false);
  const lastGlobalStatus = useRef<string | null>(null);

  useEffect(() => {
    if (!yam.setup || dirty || autoRestoreDirty) return;
    setConfig(yam.setup.config ?? EMPTY_CONFIG);
    setAutoRestore(yam.setup.auto_restore);
  }, [autoRestoreDirty, dirty, yam.setup]);

  useEffect(() => {
    if (!yam.setup) return;
    const signature = [
      yam.setup.state,
      yam.setup.connected,
      yam.setup.diagnostic.status,
    ].join(":");
    if (lastGlobalStatus.current !== null && lastGlobalStatus.current !== signature) {
      onSettingsRefresh();
    }
    lastGlobalStatus.current = signature;
  }, [onSettingsRefresh, yam.setup]);

  const busy = yam.operation !== null;
  const normalizedConfig = useMemo<YamSetupConfig>(() => ({
    can_interface: config.can_interface.trim(),
    leader_port: config.leader_port.trim(),
    mujoco_xml_path: config.mujoco_xml_path.trim(),
    leader_calibration_id: config.leader_calibration_id.trim(),
    leader_calibration_dir: config.leader_calibration_dir.trim(),
  }), [config]);
  const configComplete = Object.values(normalizedConfig).every(Boolean)
    && /^[A-Za-z0-9][A-Za-z0-9_.-]{0,14}$/.test(normalizedConfig.can_interface)
    && /^\/dev\//.test(normalizedConfig.leader_port)
    && normalizedConfig.mujoco_xml_path.startsWith("/")
    && normalizedConfig.leader_calibration_dir.startsWith("/")
    && /^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,62}[A-Za-z0-9])?$/.test(normalizedConfig.leader_calibration_id);
  const canDisableAutomaticConnection = Boolean(
    yam.setup?.mode === "hardware"
    && yam.setup.saved
    && yam.setup.auto_restore
    && autoRestoreDirty
    && !autoRestore
    && !dirty,
  );
  const canSaveCheckedSetup = Boolean(yam.preflight?.ready && yam.preflight.calibration_ready);

  function updateConfig(field: keyof YamSetupConfig, value: string) {
    setConfig((current) => ({ ...current, [field]: value }));
    setDirty(true);
    yam.clearPreflight();
  }

  async function discover() {
    const result = await yam.discover();
    if (!result) return;
    setConfig((current) => ({
      ...current,
      mujoco_xml_path: current.mujoco_xml_path || result.suggested_config?.mujoco_xml_path || "",
      leader_calibration_id: current.leader_calibration_id || result.suggested_config?.leader_calibration_id || "yam-leader",
      leader_calibration_dir: current.leader_calibration_dir || result.suggested_config?.leader_calibration_dir || "",
      can_interface: result.can_interfaces.length === 1 ? result.can_interfaces[0].id : current.can_interface,
      leader_port: result.leader_ports.length === 1 ? result.leader_ports[0].id : current.leader_port,
    }));
    if (result.suggested_config || result.can_interfaces.length === 1 || result.leader_ports.length === 1) {
      setDirty(true);
    }
  }

  async function save() {
    const result = await yam.save(
      normalizedConfig,
      autoRestore,
      yam.setup?.mode === "hardware" && autoRestore,
    );
    if (!result) return;
    setDirty(false);
    setAutoRestoreDirty(false);
    setConfirmForget(false);
  }

  async function connect() {
    const result = await yam.connect(yam.setup?.mode === "hardware" && safetyConfirmed);
    if (!result) return;
    setSafetyConfirmed(false);
  }

  async function forget() {
    const result = await yam.forget();
    if (!result) return;
    setConfig(EMPTY_CONFIG);
    setDirty(false);
    setAutoRestoreDirty(false);
    setConfirmForget(false);
    setSafetyConfirmed(false);
  }

  return (
    <section id="yam-setup" tabIndex={-1} aria-labelledby="yam-setup-heading" className="mt-6 scroll-mt-24 overflow-hidden rounded-xl border border-slate-200 bg-white shadow-panel focus:outline-none">
      <div className="flex flex-col gap-4 px-5 py-5 sm:flex-row sm:items-start sm:justify-between sm:px-6">
        <div className="flex items-start gap-3">
          <div className="grid h-10 w-10 shrink-0 place-items-center rounded-lg bg-slate-100 text-slate-600">
            <Bot className="h-5 w-5" strokeWidth={1.8} aria-hidden="true" />
          </div>
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-brand-600">YAM onboarding</p>
            <h2 id="yam-setup-heading" className="mt-1 text-base font-semibold text-slate-900">Set up one leader / follower pair</h2>
            <p className="mt-1 max-w-3xl text-xs leading-5 text-slate-500">
              Safely discover host-visible devices, verify configuration and calibration-file readiness, then save the rig for automatic restoration on later boots.
            </p>
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {yam.setup && (
            <span className={`rounded-full px-2.5 py-1 text-[11px] font-semibold ${statusTone(yam.setup)}`}>
              {yam.operation === "connect" ? "Connecting…" : setupLabel(yam.setup)}
            </span>
          )}
          <button
            type="button"
            onClick={() => void yam.refresh()}
            disabled={busy}
            aria-label="Refresh YAM setup status"
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-2.5 py-2 text-xs font-semibold text-slate-600 hover:bg-slate-50 disabled:opacity-50"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${yam.operation === "refresh" ? "animate-spin" : ""}`} aria-hidden="true" />
            Recheck
          </button>
        </div>
      </div>

      {yam.loading && !yam.setup ? (
        <div aria-label="Loading YAM setup" className="space-y-3 border-t border-slate-100 px-5 py-8 sm:px-6">
          <div className="h-4 w-48 animate-pulse rounded bg-slate-100" />
          <div className="h-20 animate-pulse rounded-lg bg-slate-100" />
        </div>
      ) : !yam.setup ? (
        <div className="border-t border-rose-100 bg-rose-50 px-5 py-5 text-sm text-rose-800 sm:px-6" role="alert">
          <p className="font-semibold">YAM setup is unavailable</p>
          <p className="mt-1 text-xs leading-5">{yam.error ?? "The backend did not return setup state."}</p>
          <button type="button" onClick={() => void yam.refresh()} disabled={busy} className="mt-3 rounded-lg border border-rose-200 bg-white px-3 py-2 text-xs font-semibold hover:bg-rose-50 disabled:opacity-50">Try again</button>
        </div>
      ) : (
        <>
          <SetupMilestones setup={yam.setup} />

          <div aria-live="polite">
            {yam.stale && (
              <div className="flex items-start gap-2 border-b border-amber-200 bg-amber-50 px-5 py-3 text-xs leading-5 text-amber-900 sm:px-6">
                <TriangleAlert className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
                Showing the last known YAM setup state because the latest recheck failed. {yam.error}
              </div>
            )}
            {yam.error && !yam.stale && (
              <div className="flex items-start gap-2 border-b border-rose-200 bg-rose-50 px-5 py-3 text-xs leading-5 text-rose-900 sm:px-6" role="alert">
                <TriangleAlert className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
                <span>
                  {yam.error}
                  {yam.setup.saved && " The last known saved setup remains shown; review it before retrying."}
                </span>
              </div>
            )}
          </div>

          {!yam.setup.requires_physical_validation ? (
            <div className="flex items-start gap-3 border-b border-blue-100 bg-blue-50 px-5 py-4 text-xs leading-5 text-blue-900 sm:px-6">
              <CircleDashed className="mt-0.5 h-4 w-4 shrink-0 text-blue-600" aria-hidden="true" />
              <p>
                Mock mode provides a deterministic leader, follower, discovery result, and calibration-ready setup. It does not detect, calibrate, connect, or validate a physical YAM.
              </p>
            </div>
          ) : (
            <div className="flex items-start gap-3 border-b border-amber-100 bg-amber-50 px-5 py-4 text-xs leading-5 text-amber-950 sm:px-6">
              <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" aria-hidden="true" />
              <p>
                Physical directions, offsets, limits, model fidelity, bus behavior, and emergency-stop operation remain unvalidated until checked on the target Ubuntu/YAM box. A bounded, structurally valid calibration file is readiness evidence only—not proof that the arm is calibrated correctly.
              </p>
            </div>
          )}

          {yam.setup.saved && (
            <div className={`flex items-start gap-3 border-b px-5 py-4 text-xs leading-5 sm:px-6 ${yam.setup.restored_on_boot && yam.setup.connected ? "border-emerald-100 bg-emerald-50 text-emerald-900" : "border-slate-100 bg-slate-50 text-slate-700"}`}>
              <RotateCcw className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
              <div>
                <p className="font-semibold">
                  {yam.setup.restored_on_boot && yam.setup.connected
                    ? "Saved configuration restored this boot"
                    : yam.setup.state === "awaiting_hardware"
                      ? "Saved configuration found; devices are not currently available"
                      : yam.setup.auto_restore
                        ? "Automatic restoration is enabled"
                        : "Setup is saved without automatic restoration"}
                </p>
                <p className="mt-0.5">
                  {yam.setup.diagnostic.detail}
                  {formatTimestamp(yam.setup.last_attempt_at) && ` Last attempt: ${formatTimestamp(yam.setup.last_attempt_at)}.`}
                </p>
              </div>
            </div>
          )}

          <div className="grid gap-8 px-5 py-6 sm:px-6 xl:grid-cols-[minmax(0,1fr)_22rem]">
            <div>
              <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                <div>
                  <h3 className="text-sm font-semibold text-slate-900">1. Discover host-visible devices</h3>
                  <p className="mt-1 max-w-2xl text-xs leading-5 text-slate-500">
                    Discovery only lists network interfaces and stable serial candidates. It does not open a bus, start a controller, or move an arm.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => void discover()}
                  disabled={busy}
                  className="inline-flex shrink-0 items-center justify-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-50"
                >
                  {yam.operation === "discover" ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" aria-hidden="true" /> : <Radar className="h-3.5 w-3.5" aria-hidden="true" />}
                  {yam.operation === "discover" ? "Scanning…" : yam.discovery ? "Rescan safely" : "Discover devices"}
                </button>
              </div>

              {yam.discovery && (
                <div className="mt-4" aria-live="polite">
                  <div className="grid gap-3 sm:grid-cols-2">
                    <CandidateSummary label="CAN interfaces" candidates={yam.discovery.can_interfaces} />
                    <CandidateSummary label="Leader serial ports" candidates={yam.discovery.leader_ports} />
                  </div>
                  <p className="mt-2 text-[11px] leading-5 text-slate-500">{yam.discovery.detail}</p>
                  {(yam.discovery.can_interfaces.length > 1 || yam.discovery.leader_ports.length > 1) && (
                    <p className="mt-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-[11px] leading-5 text-amber-900">
                      Multiple candidates are intentionally not auto-selected. V1.1 supports one configured pair; identify the intended devices before continuing.
                    </p>
                  )}
                </div>
              )}

              <form className="mt-7" onSubmit={(event) => { event.preventDefault(); void yam.check(normalizedConfig); }}>
                <fieldset disabled={busy}>
                  <legend className="text-sm font-semibold text-slate-900">2. Configure and check readiness</legend>
                  <p className="mt-1 text-xs leading-5 text-slate-500">Paths are resolved by the backend on the YAM-connected host. These fields never access devices from the browser.</p>
                  <div className="mt-4 grid gap-5 sm:grid-cols-2">
                    <ConfigInput label="Follower CAN interface" help="For example can0. Choose deliberately when multiple interfaces are listed." field="can_interface" value={config.can_interface} maxLength={15} candidates={yam.discovery?.can_interfaces} onChange={updateConfig} />
                    <ConfigInput label="Leader serial port" help="Prefer a stable /dev/serial/by-id path over a changing ttyUSB index." field="leader_port" value={config.leader_port} maxLength={200} candidates={yam.discovery?.leader_ports} onChange={updateConfig} />
                    <ConfigInput label="MuJoCo model XML" help="Absolute path to the standard YAM model on the backend host." field="mujoco_xml_path" value={config.mujoco_xml_path} maxLength={1024} onChange={updateConfig} />
                    <ConfigInput label="Leader calibration directory" help="Directory containing the existing calibration JSON; setup does not create or rewrite it." field="leader_calibration_dir" value={config.leader_calibration_dir} maxLength={1024} onChange={updateConfig} />
                    <ConfigInput label="Leader calibration ID" help="The calibration filename without .json. Letters, numbers, dot, underscore, and hyphen only." field="leader_calibration_id" value={config.leader_calibration_id} maxLength={64} onChange={updateConfig} />
                  </div>
                  {!configComplete && (
                    <p className="mt-3 text-[11px] leading-5 text-amber-700">Complete all five fields with a Linux-safe interface, a /dev leader path, absolute model/calibration paths, and a valid calibration ID before preflight. The two devices must also appear in the latest discovery result.</p>
                  )}
                  <div className="mt-4 flex flex-wrap items-center gap-3 border-t border-slate-100 pt-4">
                    <button type="submit" disabled={!configComplete || busy} className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-3.5 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-50">
                      {yam.operation === "preflight" ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" aria-hidden="true" /> : <FileCheck2 className="h-3.5 w-3.5" aria-hidden="true" />}
                      {yam.operation === "preflight" ? "Checking…" : "Run read-only preflight"}
                    </button>
                    <span className="text-[11px] text-slate-400">No connection or motion occurs.</span>
                  </div>
                </fieldset>
              </form>

              {yam.preflight && (
                <div className={`mt-4 flex items-start gap-3 rounded-lg border px-4 py-3 text-xs leading-5 ${yam.preflight.ready ? "border-emerald-200 bg-emerald-50 text-emerald-900" : "border-amber-200 bg-amber-50 text-amber-950"}`} role="status">
                  {yam.preflight.ready ? <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" /> : <TriangleAlert className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />}
                  <div>
                    <p className="font-semibold">{yam.preflight.ready ? "Read-only preflight passed" : "Preflight needs attention"}</p>
                    <p className="mt-0.5">{yam.preflight.diagnostic.detail}</p>
                    <p className="mt-0.5">
                      {yam.preflight.calibration_ready
                        ? "The calibration file is readable and matches the bounded pinned structure, but its physical correctness has not been validated."
                        : "No readable, structurally valid calibration file was detected. Create and verify it with the pinned YAM tooling before connecting."}
                    </p>
                  </div>
                </div>
              )}

              <div className="mt-7 border-t border-slate-100 pt-6">
                <h3 className="text-sm font-semibold text-slate-900">3. Save and connect explicitly</h3>
                {yam.setup.mode === "hardware" ? (
                  <label className="mt-3 flex items-start gap-3 rounded-lg border border-slate-200 bg-slate-50 px-3 py-3 text-xs leading-5 text-slate-700">
                    <input
                      type="checkbox"
                      checked={autoRestore}
                      onChange={(event) => {
                        const next = event.target.checked;
                        setAutoRestore(next);
                        setAutoRestoreDirty(next !== yam.setup?.auto_restore);
                      }}
                      disabled={busy}
                      className="mt-0.5 h-4 w-4 rounded border-slate-300 text-brand-600 focus:ring-brand-500"
                    />
                    <span><strong className="font-semibold text-slate-800">I understand and enable automatic connection.</strong> On later boots and hot-plug, ctrl-π will connect when this saved rig becomes ready. That can engage the pinned follower gravity-compensation controller without another prompt; missing or failed hardware still fails closed.</span>
                  </label>
                ) : (
                  <p className="mt-3 rounded-lg border border-blue-100 bg-blue-50 px-3 py-3 text-xs leading-5 text-blue-900">
                    Applying this deterministic mock setup does not overwrite or delete a saved hardware configuration.
                  </p>
                )}
                <button
                  type="button"
                  onClick={() => void save()}
                  disabled={busy || (!canSaveCheckedSetup && !canDisableAutomaticConnection)}
                  className="mt-3 inline-flex items-center gap-2 rounded-lg bg-ink px-3.5 py-2 text-xs font-semibold text-white hover:bg-slate-700 disabled:opacity-50"
                >
                  {yam.operation === "save" ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" aria-hidden="true" /> : <Save className="h-3.5 w-3.5" aria-hidden="true" />}
                  {yam.operation === "save"
                    ? "Applying…"
                    : canDisableAutomaticConnection
                      ? "Disable automatic connection"
                      : yam.setup.mode === "mock"
                        ? "Apply mock setup"
                        : "Save YAM setup"}
                </button>
                {(dirty || autoRestoreDirty) && yam.setup.config && (
                  <button
                    type="button"
                    onClick={() => {
                      setConfig(yam.setup?.config ?? EMPTY_CONFIG);
                      setDirty(false);
                      setAutoRestore(yam.setup?.auto_restore ?? false);
                      setAutoRestoreDirty(false);
                      yam.clearPreflight();
                    }}
                    disabled={busy}
                    className="mt-3 ml-2 rounded-lg border border-slate-200 bg-white px-3.5 py-2 text-xs font-semibold text-slate-600 hover:bg-slate-50 disabled:opacity-50"
                  >
                    Discard edits
                  </button>
                )}
                {canDisableAutomaticConnection ? (
                  <p className="mt-2 text-[11px] leading-5 text-slate-500">This revokes future boot/hot-plug connection consent without requiring the rig to be present. It does not disconnect hardware that is already connected.</p>
                ) : !yam.preflight?.ready ? (
                  <p className="mt-2 text-[11px] text-slate-400">Run a passing preflight before changing configuration or enabling automatic connection.</p>
                ) : null}
              </div>
            </div>

            <aside className="space-y-4">
              <section className="rounded-xl border border-slate-200 bg-slate-50 p-4" aria-labelledby="yam-connect-heading">
                <div className="flex items-center gap-2">
                  <PlugZap className="h-4 w-4 text-slate-500" aria-hidden="true" />
                  <h3 id="yam-connect-heading" className="text-sm font-semibold text-slate-900">Connection gate</h3>
                </div>
                <p className="mt-2 text-xs leading-5 text-slate-600">{yam.setup.diagnostic.detail}</p>
                {yam.setup.connected ? (
                  <div className="mt-4 rounded-lg border border-emerald-200 bg-white px-3 py-3 text-xs leading-5 text-emerald-800">
                    <p className="flex items-center gap-2 font-semibold"><Check className="h-4 w-4" aria-hidden="true" /> Connection verified</p>
                    <p className="mt-1">Both configured devices returned a sample{formatTimestamp(yam.setup.last_connected_at) ? ` at ${formatTimestamp(yam.setup.last_connected_at)}` : ""}.</p>
                  </div>
                ) : yam.setup.configured && (yam.setup.mode === "mock" || yam.setup.saved) ? (
                  <>
                    {yam.setup.requires_physical_validation && (
                      <label className="mt-4 flex items-start gap-2.5 rounded-lg border border-amber-200 bg-amber-50 px-3 py-3 text-[11px] leading-5 text-amber-950">
                        <input type="checkbox" checked={safetyConfirmed} onChange={(event) => setSafetyConfirmed(event.target.checked)} disabled={busy} className="mt-0.5 h-4 w-4 shrink-0 rounded border-amber-300 text-amber-600 focus:ring-amber-500" />
                        <span>I secured the workspace, verified power/motion controls, and have an operator at the emergency stop. I understand Connect may engage the pinned follower gravity-compensation controller.</span>
                      </label>
                    )}
                    <button
                      type="button"
                      onClick={() => void connect()}
                      disabled={busy || dirty || autoRestoreDirty || !yam.setup.calibration_ready || (yam.setup.requires_physical_validation && !safetyConfirmed)}
                      className="mt-3 inline-flex w-full items-center justify-center gap-2 rounded-lg bg-brand-600 px-3.5 py-2.5 text-xs font-semibold text-white hover:bg-brand-700 disabled:opacity-50"
                    >
                      {yam.operation === "connect" ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" aria-hidden="true" /> : <PlugZap className="h-3.5 w-3.5" aria-hidden="true" />}
                      {yam.operation === "connect" ? "Connecting safely…" : "Connect and verify one sample"}
                    </button>
                    {(dirty || autoRestoreDirty) && <p className="mt-2 text-[11px] leading-5 text-amber-700">Save or discard pending setup preferences before connecting; Connect always uses the applied backend configuration and persisted automatic-connection choice.</p>}
                  </>
                ) : (
                  <p className="mt-4 rounded-lg border border-slate-200 bg-white px-3 py-3 text-xs leading-5 text-slate-500">Complete and save setup before connecting.</p>
                )}
              </section>

              <section className="rounded-xl border border-slate-200 p-4">
                <h3 className="text-xs font-semibold text-slate-800">What automatic restoration means</h3>
                <ul className="mt-2 space-y-2 text-[11px] leading-5 text-slate-500">
                  <li>• The backend loads the saved non-secret device configuration at boot.</li>
                  <li>• It re-detects prerequisites and reports missing hardware instead of substituting mocks.</li>
                  <li>• While this page is visible, waiting status refreshes automatically so a newly plugged-in saved rig can appear.</li>
                  <li>• Live device access, CAN, and vendor details remain behind the YAMDriver service.</li>
                </ul>
              </section>

              {yam.setup.saved && (
                <section className="rounded-xl border border-rose-100 bg-rose-50/50 p-4">
                  {!confirmForget ? (
                    <button type="button" onClick={() => setConfirmForget(true)} disabled={busy} className="inline-flex items-center gap-2 text-xs font-semibold text-rose-700 hover:text-rose-900 disabled:opacity-50"><Trash2 className="h-3.5 w-3.5" aria-hidden="true" /> Forget saved YAM setup</button>
                  ) : (
                    <div>
                      <p className="text-xs font-semibold text-rose-900">Forget this saved rig?</p>
                      <p className="mt-1 text-[11px] leading-5 text-rose-800">This disables automatic restoration and safely closes any setup-owned connection. You can rediscover it later.</p>
                      <div className="mt-3 flex flex-wrap gap-2">
                        <button type="button" onClick={() => void forget()} disabled={busy} className="rounded-lg bg-rose-700 px-3 py-2 text-[11px] font-semibold text-white hover:bg-rose-800 disabled:opacity-50">{yam.operation === "forget" ? "Forgetting…" : "Confirm forget"}</button>
                        <button type="button" onClick={() => setConfirmForget(false)} disabled={busy} className="rounded-lg border border-rose-200 bg-white px-3 py-2 text-[11px] font-semibold text-rose-700 hover:bg-rose-50 disabled:opacity-50">Cancel</button>
                      </div>
                    </div>
                  )}
                </section>
              )}
            </aside>
          </div>
        </>
      )}
    </section>
  );
}
