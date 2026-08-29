import {
  Activity,
  Bot,
  Cable,
  CheckCircle2,
  CircleDashed,
  FileCheck2,
  Grip,
  LoaderCircle,
  Plus,
  PlugZap,
  PowerOff,
  Radar,
  RefreshCw,
  Save,
  ShieldAlert,
  Trash2,
  TriangleAlert,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { useYamSetup } from "../hooks/useYamSetup";
import type {
  YamCellArmConfig,
  YamCellConfig,
  YamDiscoveryDevice,
  YamEndEffectorKind,
  YamSetupArmStatus,
  YamSetupStatus,
} from "../types/yamSetup";
import { isYamCellConfig } from "../types/yamSetup";

const EMPTY_CELL: YamCellConfig = {
  kind: "cell",
  name: "My YAM cell",
  i2rt_root: "",
  i2rt_commit: "",
  arms: [],
  pair_ports: {},
};

const END_EFFECTORS: YamEndEffectorKind[] = [
  "yam_teaching_handle",
  "linear_4310",
  "crank_4310",
  "gello",
  "none",
];

const CALIBRATED_4310_END_EFFECTORS = new Set<YamEndEffectorKind>([
  "linear_4310",
  "crank_4310",
]);

function requiresJawCalibrationConsent(arm: YamCellArmConfig): boolean {
  return arm.role === "follower" && CALIBRATED_4310_END_EFFECTORS.has(arm.end_effector_kind);
}

function blankArm(device?: YamDiscoveryDevice): YamCellArmConfig {
  const serial = device?.stable_identity ?? "";
  const suffix = serial.replace(/[^A-Za-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(-18);
  return {
    logical_id: suffix ? `yam-arm-${suffix.toLowerCase()}` : "",
    name: "",
    role: "follower",
    pair_id: null,
    group_id: null,
    side: null,
    transport_kind: device?.transport_kind ?? "socketcan",
    stable_identity: serial,
    end_effector_kind: device?.transport_kind === "serial" ? "gello" : "linear_4310",
    frame_map_path: null,
    soft_limits_path: null,
    mujoco_xml_path: null,
    calibration_id: null,
    calibration_dir: null,
  };
}

function cloneCell(config: YamCellConfig): YamCellConfig {
  return {
    ...config,
    arms: config.arms.map((arm) => ({ ...arm })),
    pair_ports: { ...config.pair_ports },
  };
}

function setupLabel(setup: YamSetupStatus): string {
  if (setup.mode === "mock") return setup.all_connected ? "Mock cell connected" : "Mock cell ready";
  if (setup.all_connected) return setup.restored_on_boot ? "Restored and connected" : "All arms connected";
  if (setup.any_connected) return `${setup.connected_arm_count}/${setup.configured_arm_count} arms connected`;
  if (setup.state === "ready_to_connect") return "Ready to connect";
  if (setup.state === "awaiting_hardware") return "Waiting for hardware";
  if (setup.state === "error") return "Needs attention";
  return "Cell setup required";
}

function statusTone(setup: YamSetupStatus): string {
  if (setup.state === "error") return "bg-rose-50 text-rose-700";
  if (setup.all_connected) return "bg-emerald-50 text-emerald-700";
  if (setup.mode === "mock") return "bg-blue-50 text-blue-700";
  return "bg-amber-50 text-amber-700";
}

function cleanOptional(value: string | null): string | null {
  const trimmed = value?.trim() ?? "";
  return trimmed || null;
}

function normalizedCell(config: YamCellConfig): YamCellConfig {
  const pairIds = new Set(
    config.arms.map((arm) => cleanOptional(arm.pair_id)).filter((value): value is string => Boolean(value)),
  );
  return {
    ...config,
    name: config.name.trim(),
    i2rt_root: config.i2rt_root.trim(),
    i2rt_commit: config.i2rt_commit.trim().toLowerCase(),
    arms: config.arms.map((arm) => ({
      ...arm,
      logical_id: arm.logical_id.trim(),
      name: arm.name.trim(),
      stable_identity: arm.stable_identity.trim(),
      pair_id: cleanOptional(arm.pair_id),
      group_id: cleanOptional(arm.group_id),
      side: cleanOptional(arm.side),
      frame_map_path: cleanOptional(arm.frame_map_path),
      soft_limits_path: cleanOptional(arm.soft_limits_path),
      mujoco_xml_path: cleanOptional(arm.mujoco_xml_path),
      calibration_id: cleanOptional(arm.calibration_id),
      calibration_dir: cleanOptional(arm.calibration_dir),
    })),
    pair_ports: Object.fromEntries(
      Object.entries(config.pair_ports).filter(
        ([pairId, port]) => pairIds.has(pairId) && Number.isInteger(port) && port >= 1_024 && port <= 65_535,
      ),
    ),
  };
}

function cellIssue(config: YamCellConfig): string | null {
  if (!config.name || !config.i2rt_root.startsWith("/")) {
    return "Enter a cell name and an absolute read-only i2rt checkout path.";
  }
  if (!/^[0-9a-f]{40}$/.test(config.i2rt_commit)) {
    return "Enter the exact 40-character commit of the mounted i2rt checkout.";
  }
  if (config.arms.some((arm) => !arm.logical_id || !arm.name || !arm.stable_identity)) {
    return "Every arm needs a logical ID, display name, and durable physical identity.";
  }
  if (config.arms.some((arm) => arm.transport_kind === "socketcan" && /^can\d+$/.test(arm.stable_identity))) {
    return "Store each USB-CAN adapter serial as identity; canN is runtime state only.";
  }
  if (new Set(config.arms.map((arm) => arm.logical_id)).size !== config.arms.length) {
    return "Arm logical IDs must be unique.";
  }
  const identities = config.arms.map((arm) => `${arm.transport_kind}:${arm.stable_identity.toLowerCase()}`);
  if (new Set(identities).size !== identities.length) return "Physical identities must be unique.";
  const pairs = new Map<string, YamCellArmConfig[]>();
  for (const arm of config.arms) {
    if (arm.pair_id) pairs.set(arm.pair_id, [...(pairs.get(arm.pair_id) ?? []), arm]);
    const supported = arm.transport_kind === "socketcan"
      ? (arm.role === "leader" ? arm.end_effector_kind === "yam_teaching_handle" : ["linear_4310", "crank_4310"].includes(arm.end_effector_kind))
      : arm.role === "leader" && arm.end_effector_kind === "gello" && arm.stable_identity.startsWith("/dev/serial/by-id/");
    if (!supported) return `Arm ${arm.logical_id || "(unnamed)"} has an unsupported transport, role, and end-effector combination.`;
  }
  for (const [pairId, arms] of pairs) {
    if (arms.length !== 2 || !arms.some((arm) => arm.role === "leader") || !arms.some((arm) => arm.role === "follower")) {
      return `Pair ${pairId} must contain exactly one leader and one follower.`;
    }
  }
  return null;
}

function Milestones({ setup }: { setup: YamSetupStatus }) {
  const items = [
    ["Topology", `${setup.configured_arm_count} configured arm${setup.configured_arm_count === 1 ? "" : "s"}`, setup.configured, Cable],
    ["Saved cell", setup.mode === "mock" ? "Built-in deterministic fixture" : setup.saved ? "Stored in PostgreSQL" : "Not saved", setup.mode === "mock" || setup.saved, Save],
    ["Passive preflight", setup.calibration_ready ? "Configuration is ready" : "Run preflight", setup.calibration_ready, FileCheck2],
    ["Connections", `${setup.connected_arm_count}/${setup.configured_arm_count} connected`, setup.all_connected, PlugZap],
  ] as const;
  return (
    <div className="grid gap-px border-y border-slate-100 bg-slate-100 sm:grid-cols-2 xl:grid-cols-4">
      {items.map(([title, detail, ready, Icon]) => (
        <article key={title} className="bg-white px-5 py-4 sm:px-6">
          <div className="flex items-center justify-between"><Icon className={`h-4 w-4 ${ready ? "text-emerald-500" : "text-slate-400"}`} /><span className={`rounded-full px-2 py-1 text-[10px] font-semibold ${ready ? "bg-emerald-50 text-emerald-700" : "bg-slate-100 text-slate-500"}`}>{ready ? "Ready" : "Pending"}</span></div>
          <h3 className="mt-3 text-xs font-semibold text-slate-800">{title}</h3>
          <p className="mt-1 text-[11px] leading-5 text-slate-500">{detail}</p>
        </article>
      ))}
    </div>
  );
}

function DeviceList({ devices, assigned, onAssign }: { devices: YamDiscoveryDevice[]; assigned: Set<string>; onAssign: (device: YamDiscoveryDevice) => void }) {
  if (devices.length === 0) return <p className="rounded-lg border border-dashed border-slate-200 px-3 py-4 text-xs text-slate-400">No durable transport identities were found.</p>;
  return (
    <div className="overflow-hidden rounded-lg border border-slate-200">
      {devices.map((device) => {
        const key = `${device.transport_kind}:${device.stable_identity}`;
        return (
          <article key={key} className="flex flex-col gap-3 border-b border-slate-100 px-3 py-3 last:border-b-0 sm:flex-row sm:items-center sm:justify-between">
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2"><span className="rounded bg-slate-100 px-2 py-0.5 text-[10px] font-semibold uppercase text-slate-600">{device.transport_kind}</span><span className={`text-[10px] font-semibold ${device.link_state === "up" ? "text-emerald-600" : "text-amber-600"}`}>{device.link_state}</span>{device.duplicate_identity && <span className="text-[10px] font-semibold text-rose-600">Duplicate identity</span>}</div>
              <p className="mt-1 break-all font-mono text-xs font-semibold text-slate-800">{device.stable_identity}</p>
              <p className="mt-1 text-[11px] text-slate-500">{device.product ?? "Unlabeled adapter"} · current interface <span className="font-mono">{device.runtime_interface ?? "unresolved"}</span></p>
            </div>
            <button type="button" onClick={() => onAssign(device)} disabled={assigned.has(key) || device.duplicate_identity} className="shrink-0 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-40">{assigned.has(key) ? "Assigned" : "Assign"}</button>
          </article>
        );
      })}
    </div>
  );
}

function TextField({ label, value, help, onChange, placeholder, mono = false }: { label: string; value: string; help?: string; onChange: (value: string) => void; placeholder?: string; mono?: boolean }) {
  return (
    <label className="block text-[11px] font-medium text-slate-700">{label}<input value={value} onChange={(event) => onChange(event.target.value)} placeholder={placeholder} className={`mt-1.5 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs outline-none ring-brand-100 focus:border-brand-500 focus:ring-4 ${mono ? "font-mono" : ""}`} />{help && <span className="mt-1 block font-normal leading-4 text-slate-400">{help}</span>}</label>
  );
}

function ArmEditor({ arm, runtimeInterface, onChange, onRemove }: { arm: YamCellArmConfig; runtimeInterface: string | null; onChange: (arm: YamCellArmConfig) => void; onRemove: () => void }) {
  const update = <K extends keyof YamCellArmConfig>(field: K, value: YamCellArmConfig[K]) => onChange({ ...arm, [field]: value });
  const endEffectors = arm.transport_kind === "serial" ? ["gello"] as YamEndEffectorKind[] : END_EFFECTORS.filter((kind) => kind !== "gello");
  return (
    <article className="rounded-xl border border-slate-200 bg-slate-50/50 p-4">
      <div className="flex items-start justify-between gap-3">
        <div><p className="text-xs font-semibold text-slate-800">{arm.name || arm.logical_id || "Unassigned arm"}</p><p className="mt-1 text-[10px] text-slate-400">Durable identity and logical routing metadata</p></div>
        <button type="button" onClick={onRemove} aria-label="Remove arm" className="rounded-md p-1.5 text-slate-400 hover:bg-rose-50 hover:text-rose-600"><Trash2 className="h-3.5 w-3.5" /></button>
      </div>
      <div className="mt-4 grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        <TextField label="Logical ID" value={arm.logical_id} onChange={(value) => update("logical_id", value)} placeholder="yam-follower-right" mono />
        <TextField label="Display name" value={arm.name} onChange={(value) => update("name", value)} placeholder="Right follower" />
        <label className="block text-[11px] font-medium text-slate-700">Role<select value={arm.role} onChange={(event) => { const role = event.target.value as YamCellArmConfig["role"]; onChange({ ...arm, role, end_effector_kind: arm.transport_kind === "serial" ? "gello" : role === "leader" ? "yam_teaching_handle" : "linear_4310" }); }} className="mt-1.5 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs"><option value="leader">Leader</option><option value="follower">Follower</option></select></label>
        <TextField label="Pair" value={arm.pair_id ?? ""} onChange={(value) => update("pair_id", value || null)} placeholder="right" />
        <TextField label="Group" value={arm.group_id ?? ""} onChange={(value) => update("group_id", value || null)} placeholder="bimanual (optional)" />
        <TextField label="Side" value={arm.side ?? ""} onChange={(value) => update("side", value || null)} placeholder="right (optional)" />
        <label className="block text-[11px] font-medium text-slate-700">Transport<select value={arm.transport_kind} onChange={(event) => { const transport = event.target.value as YamCellArmConfig["transport_kind"]; onChange({ ...arm, transport_kind: transport, end_effector_kind: transport === "serial" ? "gello" : arm.role === "leader" ? "yam_teaching_handle" : "linear_4310" }); }} className="mt-1.5 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs"><option value="socketcan">SocketCAN</option><option value="serial">Serial GELLO</option></select></label>
        <TextField label={arm.transport_kind === "socketcan" ? "USB-CAN serial (stable)" : "Serial by-id path (stable)"} value={arm.stable_identity} onChange={(value) => update("stable_identity", value)} help={arm.transport_kind === "socketcan" ? `Current interface: ${runtimeInterface ?? "unresolved"}. Never save canN here.` : "Use /dev/serial/by-id/…, not ttyUSB indices."} mono />
        <label className="block text-[11px] font-medium text-slate-700">End effector<select value={arm.end_effector_kind} onChange={(event) => update("end_effector_kind", event.target.value as YamEndEffectorKind)} className="mt-1.5 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs">{endEffectors.map((kind) => <option key={kind} value={kind}>{kind}</option>)}</select></label>
        {arm.role === "follower" && <><TextField label="Frame map path" value={arm.frame_map_path ?? ""} onChange={(value) => update("frame_map_path", value || null)} placeholder="Blank = identity map" mono /><TextField label="Soft-limit path" value={arm.soft_limits_path ?? ""} onChange={(value) => update("soft_limits_path", value || null)} placeholder="Blank = NO SASH GUARD" mono /></>}
        {arm.transport_kind === "serial" && <><TextField label="MuJoCo model XML" value={arm.mujoco_xml_path ?? ""} onChange={(value) => update("mujoco_xml_path", value || null)} mono /><TextField label="Calibration ID" value={arm.calibration_id ?? ""} onChange={(value) => update("calibration_id", value || null)} /><TextField label="Calibration directory" value={arm.calibration_dir ?? ""} onChange={(value) => update("calibration_dir", value || null)} mono /></>}
      </div>
      {arm.role === "follower" && !arm.soft_limits_path && <p className="mt-3 rounded-lg border border-rose-300 bg-rose-50 px-3 py-2 text-xs font-bold tracking-wide text-rose-800">NO SASH GUARD</p>}
    </article>
  );
}

function ArmConnectionRow({ arm, config, checked, onChecked }: { arm: YamSetupArmStatus; config: YamCellArmConfig | undefined; checked: boolean; onChecked: (checked: boolean) => void }) {
  return (
    <label className="flex items-start gap-3 border-b border-slate-100 px-3 py-3 last:border-b-0">
      <input type="checkbox" checked={checked} onChange={(event) => onChecked(event.target.checked)} className="mt-1 h-4 w-4 rounded border-slate-300 text-brand-600 focus:ring-brand-500" />
      <span className="min-w-0 flex-1"><span className="flex flex-wrap items-center gap-2"><strong className="text-xs font-semibold text-slate-800">{config?.name ?? arm.arm_id}</strong><span className="rounded bg-slate-100 px-1.5 py-0.5 text-[9px] font-semibold uppercase text-slate-500">{arm.role}</span>{arm.pair_id && <span className="text-[10px] text-slate-500">pair {arm.pair_id}</span>}</span><span className="mt-1 block text-[10px] text-slate-500">{arm.connected ? `${arm.control_state.replaceAll("_", " ")} · ${arm.energized ? "energized" : "not energized"}${arm.holding ? " · holding" : ""}` : "disconnected"} · runtime <span className="font-mono">{arm.runtime_interface ?? "unresolved"}</span></span>{arm.error && <span className="mt-1 block text-[10px] text-rose-600">{arm.error}</span>}</span>
    </label>
  );
}

export function YamSetupPanel({ onSettingsRefresh }: { onSettingsRefresh: () => void }) {
  const yam = useYamSetup();
  const [config, setConfig] = useState<YamCellConfig>(EMPTY_CELL);
  const [dirty, setDirty] = useState(false);
  const [autoRestore, setAutoRestore] = useState(false);
  const [selectedArms, setSelectedArms] = useState<Set<string>>(new Set());
  const [motionConfirmed, setMotionConfirmed] = useState(false);
  const [jawsConfirmed, setJawsConfirmed] = useState(false);
  const [handleConfirmed, setHandleConfirmed] = useState(false);
  const [handleArmId, setHandleArmId] = useState("");
  const [confirmForget, setConfirmForget] = useState(false);
  const lastStatus = useRef<string | null>(null);

  useEffect(() => {
    if (!yam.setup || dirty) return;
    if (isYamCellConfig(yam.setup.config)) setConfig(cloneCell(yam.setup.config));
    setAutoRestore(yam.setup.auto_restore);
  }, [dirty, yam.setup]);

  useEffect(() => {
    if (!yam.setup) return;
    const setup = yam.setup;
    setSelectedArms((current) => {
      const valid = new Set(setup.arms.map((arm) => arm.arm_id));
      const next = new Set([...current].filter((id) => valid.has(id)));
      if (next.size === 0) setup.arms.filter((arm) => !arm.connected).forEach((arm) => next.add(arm.arm_id));
      return next;
    });
    const leaders = setup.arms.filter((arm) => arm.role === "leader");
    if (!leaders.some((arm) => arm.arm_id === handleArmId)) setHandleArmId(leaders[0]?.arm_id ?? "");
  }, [handleArmId, yam.setup]);

  useEffect(() => {
    if (!yam.setup) return;
    const signature = `${yam.setup.state}:${yam.setup.connected_arm_count}:${yam.setup.diagnostic.status}`;
    if (lastStatus.current !== null && lastStatus.current !== signature) onSettingsRefresh();
    lastStatus.current = signature;
  }, [onSettingsRefresh, yam.setup]);

  const normalized = useMemo(() => normalizedCell(config), [config]);
  const issue = useMemo(() => cellIssue(normalized), [normalized]);
  const configById = useMemo(() => new Map(normalized.arms.map((arm) => [arm.logical_id, arm])), [normalized.arms]);
  const assigned = useMemo(() => new Set(normalized.arms.map((arm) => `${arm.transport_kind}:${arm.stable_identity}`)), [normalized.arms]);
  const pairIds = useMemo(() => [...new Set(normalized.arms.map((arm) => arm.pair_id).filter((value): value is string => Boolean(value)))], [normalized.arms]);
  const selectedConfigs = normalized.arms.filter((arm) => selectedArms.has(arm.logical_id));
  const selectedNeedJawAck = selectedConfigs.some(requiresJawCalibrationConsent);
  const hasArms = normalized.arms.length > 0;
  const revokingAutoRestoreOnly = Boolean(
    yam.setup?.mode === "hardware"
      && yam.setup.saved
      && yam.setup.auto_restore
      && !autoRestore
      && isYamCellConfig(yam.setup.config)
      && JSON.stringify(normalized) === JSON.stringify(normalizedCell(yam.setup.config)),
  );
  const preflightAllowsSave = revokingAutoRestoreOnly
    || yam.preflight?.ready === true
    || (!hasArms && yam.preflight?.i2rt_ready === true);
  const busy = yam.operation !== null;

  function edit(next: YamCellConfig) {
    setConfig(next);
    if (next.arms.length === 0) setAutoRestore(false);
    setDirty(true);
    yam.clearPreflight();
  }

  async function discover() {
    const result = await yam.discover();
    if (!result) return;
    if (normalized.arms.length === 0 && isYamCellConfig(result.suggested_config)) {
      setConfig(cloneCell(result.suggested_config));
      setDirty(true);
    }
  }

  async function save() {
    const enabledAutoRestore = hasArms && autoRestore;
    const result = await yam.save(normalized, enabledAutoRestore, yam.setup?.mode === "hardware" && enabledAutoRestore, yam.setup?.mode === "hardware" && enabledAutoRestore && normalized.arms.some(requiresJawCalibrationConsent) && jawsConfirmed);
    if (result) { setDirty(false); setConfirmForget(false); }
  }

  async function connect() {
    const ids = [...selectedArms];
    const result = await yam.connect(ids.length === yam.setup?.arms.length ? null : ids, yam.setup?.mode === "hardware" && motionConfirmed, yam.setup?.mode === "hardware" && selectedNeedJawAck && jawsConfirmed);
    if (result) { setMotionConfirmed(false); setJawsConfirmed(false); }
  }

  async function disconnect() {
    const ids = [...selectedArms];
    await yam.disconnect(ids.length === yam.setup?.arms.length ? null : ids);
  }

  async function forget() {
    const result = await yam.forget();
    if (result) { setConfig(EMPTY_CELL); setDirty(false); setConfirmForget(false); setSelectedArms(new Set()); }
  }

  return (
    <section id="yam-setup" tabIndex={-1} aria-labelledby="yam-cell-heading" className="mt-6 scroll-mt-24 overflow-hidden rounded-xl border border-slate-200 bg-white shadow-panel focus:outline-none">
      <div className="flex flex-col gap-4 px-5 py-5 sm:flex-row sm:items-start sm:justify-between sm:px-6">
        <div className="flex items-start gap-3"><div className="grid h-10 w-10 shrink-0 place-items-center rounded-lg bg-slate-100 text-slate-600"><Bot className="h-5 w-5" /></div><div><p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-brand-600">YAM Cell</p><h2 id="yam-cell-heading" className="mt-1 text-base font-semibold text-slate-900">Configure a multi-arm YAM cell</h2><p className="mt-1 max-w-3xl text-xs leading-5 text-slate-500">Assign stable physical identities to logical arms, validate the topology passively, then connect only the arms you select.</p></div></div>
        <div className="flex shrink-0 items-center gap-2">{yam.setup && <span className={`rounded-full px-2.5 py-1 text-[11px] font-semibold ${statusTone(yam.setup)}`}>{setupLabel(yam.setup)}</span>}<button type="button" onClick={() => void yam.refresh()} disabled={busy} className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-2.5 py-2 text-xs font-semibold text-slate-600 hover:bg-slate-50 disabled:opacity-50"><RefreshCw className={`h-3.5 w-3.5 ${yam.operation === "refresh" ? "animate-spin" : ""}`} />Recheck</button></div>
      </div>

      {yam.loading && !yam.setup ? <div className="border-t border-slate-100 px-6 py-8 text-sm text-slate-400">Loading YAM cell…</div> : !yam.setup ? <div className="border-t border-rose-100 bg-rose-50 px-6 py-5 text-sm text-rose-800">{yam.error ?? "The YAM cell API is unavailable."}</div> : <>
        <Milestones setup={yam.setup} />
        {yam.error && <div role="alert" className="flex items-start gap-2 border-b border-rose-200 bg-rose-50 px-5 py-3 text-xs text-rose-900"><TriangleAlert className="mt-0.5 h-4 w-4 shrink-0" />{yam.stale ? "Showing the last known state. " : ""}{yam.error}</div>}
        {!yam.setup.requires_physical_validation ? <div className="flex items-start gap-3 border-b border-blue-100 bg-blue-50 px-5 py-4 text-xs leading-5 text-blue-900 sm:px-6"><CircleDashed className="mt-0.5 h-4 w-4 shrink-0" /><p>Mock mode uses a deterministic four-arm, two-pair cell. It does not discover, connect, calibrate, or validate physical hardware.</p></div> : <div className="flex items-start gap-3 border-b border-amber-100 bg-amber-50 px-5 py-4 text-xs leading-5 text-amber-950 sm:px-6"><ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" /><p>Physical directions, frame maps, limits, bus behavior, and emergency-stop operation remain unvalidated until the field gates pass. Passive discovery and preflight never open a device or enable a motor.</p></div>}
        {yam.setup.config && !isYamCellConfig(yam.setup.config) && <div className="border-b border-amber-200 bg-amber-50 px-5 py-3 text-xs text-amber-900 sm:px-6">A V1.1 serial-GELLO setup remains readable. Discover and preflight a cell topology before replacing it; ctrl-π will not fabricate USB-CAN serial identities from its saved canN.</div>}

        <div className="space-y-8 px-5 py-6 sm:px-6">
          <section>
            <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between"><div><h3 className="text-sm font-semibold text-slate-900">1. Discover physical transports</h3><p className="mt-1 text-xs leading-5 text-slate-500">Read-only OS inspection reports durable adapter identity separately from its current runtime interface. Roles are never inferred.</p></div><button type="button" onClick={() => void discover()} disabled={busy} className="inline-flex shrink-0 items-center gap-2 rounded-lg border border-slate-200 px-3 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-50">{yam.operation === "discover" ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : <Radar className="h-3.5 w-3.5" />}{yam.discovery ? "Rescan passively" : "Discover hardware"}</button></div>
            {yam.discovery && <div className="mt-4 space-y-3"><DeviceList devices={yam.discovery.devices ?? []} assigned={assigned} onAssign={(device) => edit({ ...config, arms: [...config.arms, blankArm(device)] })} /><p className="text-[11px] leading-5 text-slate-500">{yam.discovery.detail}</p></div>}
          </section>

          <section className="border-t border-slate-100 pt-7">
            <h3 className="text-sm font-semibold text-slate-900">2. Define topology and pinned runtime</h3><p className="mt-1 text-xs leading-5 text-slate-500">The i2rt path must be an operator-provided read-only checkout. The backend verifies the exact commit; it never substitutes public latest.</p>
            <div className="mt-4 grid gap-4 sm:grid-cols-3"><TextField label="Cell name" value={config.name} onChange={(value) => edit({ ...config, name: value })} /><TextField label="Read-only i2rt checkout" value={config.i2rt_root} onChange={(value) => edit({ ...config, i2rt_root: value })} placeholder="/opt/i2rt" mono /><TextField label="Pinned i2rt commit" value={config.i2rt_commit} onChange={(value) => edit({ ...config, i2rt_commit: value })} placeholder="40 lowercase hex characters" mono /></div>
            <div className="mt-5 space-y-4">{config.arms.map((arm, index) => <ArmEditor key={`${index}:${arm.stable_identity}`} arm={arm} runtimeInterface={yam.discovery?.devices.find((device) => device.transport_kind === arm.transport_kind && device.stable_identity === arm.stable_identity)?.runtime_interface ?? yam.setup?.arms.find((status) => status.arm_id === arm.logical_id)?.runtime_interface ?? null} onChange={(next) => edit({ ...config, arms: config.arms.map((item, itemIndex) => itemIndex === index ? next : item) })} onRemove={() => edit({ ...config, arms: config.arms.filter((_, itemIndex) => itemIndex !== index) })} />)}</div>
            <button type="button" onClick={() => edit({ ...config, arms: [...config.arms, blankArm()] })} className="mt-4 inline-flex items-center gap-2 rounded-lg border border-slate-200 px-3 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50"><Plus className="h-3.5 w-3.5" />Add arm manually</button>
            {pairIds.length > 0 && <div className="mt-5 rounded-xl border border-slate-200 p-4"><h4 className="text-xs font-semibold text-slate-800">Pair-specific ports</h4><p className="mt-1 text-[10px] leading-4 text-slate-400">Distinct ports keep pair routes isolated. Ports are cell configuration, never global defaults.</p><div className="mt-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">{pairIds.map((pairId) => <label key={pairId} className="text-[11px] font-medium text-slate-700">{pairId}<input type="number" min={1024} max={65535} value={config.pair_ports[pairId] ?? ""} onChange={(event) => { const value = Number(event.target.value); const ports = { ...config.pair_ports }; if (event.target.value) ports[pairId] = value; else delete ports[pairId]; edit({ ...config, pair_ports: ports }); }} placeholder="optional" className="mt-1.5 w-full rounded-lg border border-slate-200 px-3 py-2 font-mono text-xs" /></label>)}</div></div>}
            {issue && <p className="mt-3 text-[11px] leading-5 text-amber-700">{issue}</p>}
          </section>

          <section className="border-t border-slate-100 pt-7">
            <div className="flex flex-wrap items-center justify-between gap-3"><div><h3 className="text-sm font-semibold text-slate-900">3. Passive preflight and save</h3><p className="mt-1 text-xs leading-5 text-slate-500">Validates identity resolution, link state, topology, pinned checkout, maps, and limits without opening hardware.</p></div><button type="button" onClick={() => void yam.check(normalized)} disabled={busy || issue !== null} className="inline-flex items-center gap-2 rounded-lg border border-slate-300 px-3.5 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-40">{yam.operation === "preflight" ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : <FileCheck2 className="h-3.5 w-3.5" />}Run passive preflight</button></div>
            {!hasArms && <p className="mt-3 rounded-lg border border-blue-200 bg-blue-50 px-3 py-2 text-[11px] leading-5 text-blue-900">An empty cell can be preflighted and saved as a topology draft after the exact i2rt checkout is verified. It is never connect-ready.</p>}
            {yam.preflight && <div className={`mt-4 rounded-xl border p-4 text-xs ${yam.preflight.ready ? "border-emerald-200 bg-emerald-50 text-emerald-900" : "border-amber-200 bg-amber-50 text-amber-950"}`}><p className="font-semibold">{yam.preflight.diagnostic.detail}</p><div className="mt-3 grid gap-2 sm:grid-cols-2">{yam.preflight.arms.map((arm) => <div key={arm.arm_id} className="rounded-lg border border-current/10 bg-white/70 px-3 py-2"><div className="flex items-center justify-between gap-2"><span className="font-mono font-semibold">{arm.arm_id}</span><span className="font-semibold">{arm.ready ? "Ready" : "Blocked"}</span></div><p className="mt-1 text-[10px]">runtime {arm.runtime_interface ?? "unresolved"} · map {arm.frame_map_status} · limits {arm.soft_limits_status} · handle {arm.handle_status}</p>{arm.warnings.map((warning) => <p key={warning} className={`mt-1 text-[10px] font-bold ${warning.includes("NO SASH GUARD") ? "text-rose-700" : ""}`}>{warning}</p>)}</div>)}</div></div>}
            {yam.setup.mode === "hardware" && <label className="mt-4 flex items-start gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3 text-xs leading-5 text-slate-700"><input type="checkbox" checked={hasArms && autoRestore} onChange={(event) => { setAutoRestore(event.target.checked); setDirty(true); }} disabled={busy || !hasArms} className="mt-0.5 h-4 w-4 rounded border-slate-300 text-brand-600 disabled:opacity-40" /><span><strong>Enable automatic connection.</strong> This is persistent consent to energize configured arms on boot/hot-plug. Calibrated <code>linear_4310</code> and <code>crank_4310</code> followers can also move their jaws during connection.{!hasArms && <span className="mt-1 block font-semibold text-amber-700">Add at least one arm before automatic connection can be enabled.</span>}</span></label>}
            {revokingAutoRestoreOnly && <p className="mt-3 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-[11px] leading-5 text-emerald-900">Revoking automatic connection is a database-only consent change. It does not require preflight, disconnect, or hardware access.</p>}
            {yam.setup.mode === "hardware" && autoRestore && normalized.arms.some(requiresJawCalibrationConsent) && <label className="mt-3 flex items-start gap-3 rounded-lg border border-amber-300 bg-amber-50 p-3 text-xs leading-5 text-amber-950"><input type="checkbox" checked={jawsConfirmed} onChange={(event) => setJawsConfirmed(event.target.checked)} className="mt-0.5 h-4 w-4 rounded border-amber-400 text-amber-600" /><span>I understand automatic connection can calibrate each configured 4310 follower (<code>linear_4310</code> or <code>crank_4310</code>). The jaws will move; I will keep them and the arm workspace clear.</span></label>}
            <div className="mt-4 flex flex-wrap gap-2"><button type="button" onClick={() => void save()} disabled={busy || !preflightAllowsSave || (yam.setup.mode === "hardware" && autoRestore && normalized.arms.some(requiresJawCalibrationConsent) && !jawsConfirmed)} className="inline-flex items-center gap-2 rounded-lg bg-ink px-3.5 py-2 text-xs font-semibold text-white hover:bg-slate-700 disabled:opacity-40"><Save className="h-3.5 w-3.5" />{yam.operation === "save" ? "Saving…" : yam.setup.mode === "mock" ? "Apply mock cell" : "Save cell"}</button>{dirty && isYamCellConfig(yam.setup.config) && <button type="button" onClick={() => { setConfig(cloneCell(yam.setup!.config as YamCellConfig)); setDirty(false); yam.clearPreflight(); }} disabled={busy} className="rounded-lg border border-slate-200 px-3.5 py-2 text-xs font-semibold text-slate-600">Discard edits</button>}</div>
          </section>

          <section className="border-t border-slate-100 pt-7">
            <h3 className="text-sm font-semibold text-slate-900">4. Connect or disconnect selected arms</h3><p className="mt-1 text-xs leading-5 text-slate-500">Every CAN arm connect is motion-capable and may energize or resist motion. Disconnect stops writers and requests the safest supported release; wait for the displayed disconnected/non-energized state before treating an arm as limp.</p>
            <div className="mt-4 overflow-hidden rounded-xl border border-slate-200">{yam.setup.arms.length === 0 ? <p className="px-4 py-5 text-xs leading-5 text-slate-500">This saved cell is an empty topology draft. Add and save at least one arm before connecting.</p> : yam.setup.arms.map((arm) => <ArmConnectionRow key={arm.arm_id} arm={arm} config={configById.get(arm.arm_id)} checked={selectedArms.has(arm.arm_id)} onChecked={(checked) => setSelectedArms((current) => { const next = new Set(current); if (checked) next.add(arm.arm_id); else next.delete(arm.arm_id); return next; })} />)}</div>
            <div className="mt-3 flex flex-wrap gap-2"><button type="button" onClick={() => setSelectedArms(new Set(yam.setup!.arms.map((arm) => arm.arm_id)))} className="text-[11px] font-semibold text-brand-700">Select all</button><button type="button" onClick={() => setSelectedArms(new Set())} className="text-[11px] font-semibold text-slate-500">Clear selection</button></div>
            {yam.setup.requires_physical_validation && <label className="mt-4 flex items-start gap-3 rounded-lg border border-amber-300 bg-amber-50 p-3 text-xs leading-5 text-amber-950"><input type="checkbox" checked={motionConfirmed} onChange={(event) => setMotionConfirmed(event.target.checked)} className="mt-0.5 h-4 w-4 rounded border-amber-400 text-amber-600" /><span>I secured the workspace, verified the emergency stop, and understand Connect can enable motors, start gravity compensation, and make every selected CAN arm resist manual motion.</span></label>}
            {yam.setup.requires_physical_validation && selectedNeedJawAck && <label className="mt-3 flex items-start gap-3 rounded-lg border border-rose-300 bg-rose-50 p-3 text-xs leading-5 text-rose-950"><input type="checkbox" checked={jawsConfirmed} onChange={(event) => setJawsConfirmed(event.target.checked)} className="mt-0.5 h-4 w-4 rounded border-rose-400 text-rose-600" /><span><strong>Connecting a selected <code>linear_4310</code> or <code>crank_4310</code> follower will enable its arm controller and calibrate the gripper. The jaws will move. Clear every selected jaw and arm workspace.</strong></span></label>}
            <div className="mt-4 flex flex-wrap gap-2"><button type="button" onClick={() => void connect()} disabled={busy || dirty || !hasArms || selectedArms.size === 0 || !yam.setup.calibration_ready || (yam.setup.requires_physical_validation && (!motionConfirmed || (selectedNeedJawAck && !jawsConfirmed)))} className="inline-flex items-center gap-2 rounded-lg bg-brand-600 px-3.5 py-2.5 text-xs font-semibold text-white hover:bg-brand-700 disabled:opacity-40">{yam.operation === "connect" ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : <PlugZap className="h-3.5 w-3.5" />}Connect selected</button><button type="button" onClick={() => void disconnect()} disabled={busy || selectedArms.size === 0 || !yam.setup.arms.some((arm) => selectedArms.has(arm.arm_id) && arm.connected)} className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-3.5 py-2.5 text-xs font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-40"><PowerOff className="h-3.5 w-3.5" />{yam.operation === "disconnect" ? "Disconnecting…" : "Disconnect selected"}</button></div>
          </section>

          <section className="border-t border-slate-100 pt-7">
            <div className="flex items-center gap-2"><Grip className="h-4 w-4 text-slate-500" /><h3 className="text-sm font-semibold text-slate-900">5. Teaching-handle range check</h3></div><p className="mt-1 text-xs leading-5 text-slate-500">CAN presence is not handle health. This separate active diagnostic observes trigger travel for 10 seconds; squeeze and release fully. It never re-zeros the encoder.</p>
            <div className="mt-3 grid gap-3 sm:grid-cols-[minmax(0,20rem)_1fr]"><select value={handleArmId} onChange={(event) => setHandleArmId(event.target.value)} className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs"><option value="">No configured leader</option>{yam.setup.arms.filter((arm) => arm.role === "leader").map((arm) => <option key={arm.arm_id} value={arm.arm_id}>{configById.get(arm.arm_id)?.name ?? arm.arm_id} · {arm.pair_id ?? "unpaired"}</option>)}</select>{yam.setup.requires_physical_validation && <label className="flex items-start gap-2 text-[11px] leading-5 text-slate-600"><input type="checkbox" checked={handleConfirmed} onChange={(event) => setHandleConfirmed(event.target.checked)} className="mt-0.5 h-4 w-4 rounded border-slate-300 text-brand-600" />I understand this is an active CAN input diagnostic and will operate only the selected handle.</label>}</div>
            <button type="button" onClick={() => void yam.checkHandle(handleArmId, 10, yam.setup?.mode === "hardware" && handleConfirmed)} disabled={busy || !handleArmId || (yam.setup.requires_physical_validation && !handleConfirmed)} className="mt-3 inline-flex items-center gap-2 rounded-lg border border-slate-300 px-3.5 py-2 text-xs font-semibold text-slate-700 disabled:opacity-40">{yam.operation === "handle-check" ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : <Activity className="h-3.5 w-3.5" />}Run 10-second range check</button>
            {yam.handleResult && <div className={`mt-3 rounded-lg border px-3 py-3 text-xs ${yam.handleResult.healthy ? "border-emerald-200 bg-emerald-50 text-emerald-900" : "border-rose-200 bg-rose-50 text-rose-900"}`}><p className="flex items-center gap-2 font-semibold">{yam.handleResult.healthy ? <CheckCircle2 className="h-4 w-4" /> : <TriangleAlert className="h-4 w-4" />}{yam.handleResult.detail}</p><p className="mt-1 font-mono text-[10px]">observed {yam.handleResult.observed_minimum?.toFixed(3) ?? "—"} … {yam.handleResult.observed_maximum?.toFixed(3) ?? "—"}</p>{!yam.handleResult.healthy && <p className="mt-2 text-[10px] leading-4">Keep ctrl-π disconnected and use the documented i2rt CLI maintenance procedure with the trigger mechanically released. There is intentionally no in-product re-zero action.</p>}</div>}
          </section>

          {yam.setup.saved && <section className="border-t border-slate-100 pt-6">{!confirmForget ? <button type="button" onClick={() => setConfirmForget(true)} disabled={busy} className="inline-flex items-center gap-2 text-xs font-semibold text-rose-700"><Trash2 className="h-3.5 w-3.5" />Forget saved YAM cell</button> : <div className="rounded-lg border border-rose-200 bg-rose-50 p-3 text-xs text-rose-900"><p className="font-semibold">Forget the saved cell and disconnect its arms?</p><div className="mt-3 flex gap-2"><button type="button" onClick={() => void forget()} className="rounded-lg bg-rose-700 px-3 py-2 font-semibold text-white">Confirm forget</button><button type="button" onClick={() => setConfirmForget(false)} className="rounded-lg border border-rose-200 bg-white px-3 py-2 font-semibold">Cancel</button></div></div>}</section>}
        </div>
      </>}
    </section>
  );
}
