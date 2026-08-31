import {
  Activity,
  CheckCircle2,
  FileCheck2,
  Plus,
  PlugZap,
  PowerOff,
  Radar,
  RefreshCw,
  Save,
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
import { Alert } from "./ui/Alert";
import { Badge, Mono } from "./ui/Badge";
import { Button, IconButton } from "./ui/Button";
import { InlineCode } from "./ui/Code";
import { Checkbox, Field, Select, TextInput } from "./ui/Form";
import { Panel, PanelHeader, SectionHeading } from "./ui/Panel";
import { Stat, StatGrid } from "./ui/Stat";
import type { Tone } from "./ui/Badge";

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

function statusTone(setup: YamSetupStatus): Tone {
  if (setup.state === "error") return "danger";
  if (setup.all_connected) return "success";
  if (setup.mode === "mock") return "info";
  return "warning";
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
    {
      label: "Topology",
      value: `${setup.configured_arm_count} arm${setup.configured_arm_count === 1 ? "" : "s"}`,
      ready: setup.configured,
      hint: "Configured",
    },
    {
      label: "Saved cell",
      value: setup.mode === "mock" ? "Built-in" : setup.saved ? "Stored" : "Not saved",
      ready: setup.mode === "mock" || setup.saved,
      hint: setup.mode === "mock" ? "Deterministic fixture" : "PostgreSQL",
    },
    {
      label: "Preflight",
      value: setup.calibration_ready ? "Ready" : "Pending",
      ready: setup.calibration_ready,
      hint: "Passive validation",
    },
    {
      label: "Connections",
      value: `${setup.connected_arm_count}/${setup.configured_arm_count}`,
      ready: setup.all_connected,
      hint: "Connected arms",
    },
  ];
  return (
    <StatGrid columns={4}>
      {items.map((item) => (
        <Stat
          key={item.label}
          label={item.label}
          value={
            <span className={`text-base ${item.ready ? "text-ink" : "text-ink-muted"}`}>
              {item.value}
            </span>
          }
          hint={item.hint}
        />
      ))}
    </StatGrid>
  );
}

function DeviceList({
  devices,
  assigned,
  onAssign,
}: {
  devices: YamDiscoveryDevice[];
  assigned: Set<string>;
  onAssign: (device: YamDiscoveryDevice) => void;
}) {
  if (devices.length === 0) {
    return (
      <p className="rounded-lg border border-dashed border-line px-4 py-4 text-xs text-ink-muted">
        No durable transport identities were found.
      </p>
    );
  }
  return (
    <ul className="divide-y divide-line-subtle overflow-hidden rounded-lg border border-line">
      {devices.map((device) => {
        const key = `${device.transport_kind}:${device.stable_identity}`;
        return (
          <li
            key={key}
            className="flex flex-col gap-3 px-4 py-3 sm:flex-row sm:items-center sm:justify-between"
          >
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <Badge tone="neutral">{device.transport_kind}</Badge>
                <Badge tone={device.link_state === "up" ? "success" : "warning"}>
                  {device.link_state}
                </Badge>
                {device.duplicate_identity && <Badge tone="danger">Duplicate identity</Badge>}
              </div>
              <p className="mt-1.5 break-all font-mono text-xs font-medium text-ink">
                {device.stable_identity}
              </p>
              <p className="mt-1 text-2xs text-ink-muted">
                {device.product ?? "Unlabeled adapter"} · current interface{" "}
                <span className="font-mono">{device.runtime_interface ?? "unresolved"}</span>
              </p>
            </div>
            <Button
              size="sm"
              disabled={assigned.has(key) || device.duplicate_identity}
              onClick={() => onAssign(device)}
            >
              {assigned.has(key) ? "Assigned" : "Assign"}
            </Button>
          </li>
        );
      })}
    </ul>
  );
}

function ArmEditor({
  arm,
  runtimeInterface,
  onChange,
  onRemove,
}: {
  arm: YamCellArmConfig;
  runtimeInterface: string | null;
  onChange: (arm: YamCellArmConfig) => void;
  onRemove: () => void;
}) {
  const update = <K extends keyof YamCellArmConfig>(field: K, value: YamCellArmConfig[K]) =>
    onChange({ ...arm, [field]: value });
  const endEffectors =
    arm.transport_kind === "serial"
      ? (["gello"] as YamEndEffectorKind[])
      : END_EFFECTORS.filter((kind) => kind !== "gello");

  return (
    <Panel as="article">
      <PanelHeader
        title={arm.name || arm.logical_id || "Unassigned arm"}
        description="Durable identity and logical routing metadata"
        actions={<IconButton icon={Trash2} label="Remove arm" size="sm" variant="ghost" onClick={onRemove} />}
      />
      <div className="grid gap-5 px-5 py-5 sm:grid-cols-2 xl:grid-cols-3">
        <Field label="Logical ID">
          <TextInput
            className="font-mono"
            value={arm.logical_id}
            placeholder="yam-follower-right"
            onChange={(event) => update("logical_id", event.target.value)}
          />
        </Field>
        <Field label="Display name">
          <TextInput
            value={arm.name}
            placeholder="Right follower"
            onChange={(event) => update("name", event.target.value)}
          />
        </Field>
        <Field label="Role">
          <Select
            value={arm.role}
            onChange={(event) => {
              const role = event.target.value as YamCellArmConfig["role"];
              onChange({
                ...arm,
                role,
                end_effector_kind:
                  arm.transport_kind === "serial"
                    ? "gello"
                    : role === "leader"
                      ? "yam_teaching_handle"
                      : "linear_4310",
              });
            }}
          >
            <option value="leader">Leader</option>
            <option value="follower">Follower</option>
          </Select>
        </Field>
        <Field label="Pair">
          <TextInput
            value={arm.pair_id ?? ""}
            placeholder="right"
            onChange={(event) => update("pair_id", event.target.value || null)}
          />
        </Field>
        <Field label="Group" optional>
          <TextInput
            value={arm.group_id ?? ""}
            placeholder="bimanual"
            onChange={(event) => update("group_id", event.target.value || null)}
          />
        </Field>
        <Field label="Side" optional>
          <TextInput
            value={arm.side ?? ""}
            placeholder="right"
            onChange={(event) => update("side", event.target.value || null)}
          />
        </Field>
        <Field label="Transport">
          <Select
            value={arm.transport_kind}
            onChange={(event) => {
              const transport = event.target.value as YamCellArmConfig["transport_kind"];
              onChange({
                ...arm,
                transport_kind: transport,
                end_effector_kind:
                  transport === "serial"
                    ? "gello"
                    : arm.role === "leader"
                      ? "yam_teaching_handle"
                      : "linear_4310",
              });
            }}
          >
            <option value="socketcan">SocketCAN</option>
            <option value="serial">Serial GELLO</option>
          </Select>
        </Field>
        <Field
          label={arm.transport_kind === "socketcan" ? "USB-CAN serial (stable)" : "Serial by-id path (stable)"}
          hint={
            arm.transport_kind === "socketcan"
              ? `Current interface: ${runtimeInterface ?? "unresolved"}. Never save canN here.`
              : "Use /dev/serial/by-id/…, not ttyUSB indices."
          }
        >
          <TextInput
            className="font-mono"
            value={arm.stable_identity}
            onChange={(event) => update("stable_identity", event.target.value)}
          />
        </Field>
        <Field label="End effector">
          <Select
            value={arm.end_effector_kind}
            onChange={(event) => update("end_effector_kind", event.target.value as YamEndEffectorKind)}
          >
            {endEffectors.map((kind) => (
              <option key={kind} value={kind}>
                {kind}
              </option>
            ))}
          </Select>
        </Field>
        {arm.role === "follower" && (
          <>
            <Field label="Frame map path" hint="Blank = identity map">
              <TextInput
                className="font-mono"
                value={arm.frame_map_path ?? ""}
                onChange={(event) => update("frame_map_path", event.target.value || null)}
              />
            </Field>
            <Field label="Soft-limit path" hint="Blank = NO SASH GUARD">
              <TextInput
                className="font-mono"
                value={arm.soft_limits_path ?? ""}
                onChange={(event) => update("soft_limits_path", event.target.value || null)}
              />
            </Field>
          </>
        )}
        {arm.transport_kind === "serial" && (
          <>
            <Field label="MuJoCo model XML">
              <TextInput
                className="font-mono"
                value={arm.mujoco_xml_path ?? ""}
                onChange={(event) => update("mujoco_xml_path", event.target.value || null)}
              />
            </Field>
            <Field label="Calibration ID">
              <TextInput
                value={arm.calibration_id ?? ""}
                onChange={(event) => update("calibration_id", event.target.value || null)}
              />
            </Field>
            <Field label="Calibration directory">
              <TextInput
                className="font-mono"
                value={arm.calibration_dir ?? ""}
                onChange={(event) => update("calibration_dir", event.target.value || null)}
              />
            </Field>
          </>
        )}
      </div>
      {arm.role === "follower" && !arm.soft_limits_path && (
        <div className="px-5 pb-5">
          <Alert tone="danger" title="NO SASH GUARD">
            This follower has no soft-limit file. ctrl-π never invents limits.
          </Alert>
        </div>
      )}
    </Panel>
  );
}

function ArmConnectionRow({
  arm,
  config,
  checked,
  onChecked,
}: {
  arm: YamSetupArmStatus;
  config: YamCellArmConfig | undefined;
  checked: boolean;
  onChecked: (checked: boolean) => void;
}) {
  return (
    <label className="flex items-start gap-3 px-4 py-3">
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChecked(event.target.checked)}
        className="mt-0.5 h-4 w-4 shrink-0 rounded border-line-strong text-accent-600 focus:ring-accent-500"
      />
      <span className="min-w-0 flex-1">
        <span className="flex flex-wrap items-center gap-2">
          <strong className="text-[13px] font-medium text-ink">{config?.name ?? arm.arm_id}</strong>
          <Badge tone="neutral">{arm.role}</Badge>
          {arm.pair_id && <span className="text-2xs text-ink-muted">pair {arm.pair_id}</span>}
        </span>
        <span className="mt-1 block text-2xs text-ink-muted">
          {arm.connected
            ? `${arm.control_state.replaceAll("_", " ")} · ${arm.energized ? "energized" : "not energized"}${arm.holding ? " · holding" : ""}`
            : "disconnected"}{" "}
          · runtime <span className="font-mono">{arm.runtime_interface ?? "unresolved"}</span>
        </span>
        {arm.error && <span className="mt-1 block text-2xs text-critical-700">{arm.error}</span>}
      </span>
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
    <section id="yam-setup" tabIndex={-1} aria-labelledby="yam-cell-heading" className="scroll-mt-24 focus:outline-none">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <h2 id="yam-cell-heading" className="text-sm font-semibold tracking-tight text-ink">
            Configure a multi-arm YAM cell
          </h2>
          <p className="mt-1 max-w-prose text-xs leading-5 text-ink-muted">
            Assign stable physical identities to logical arms, validate the topology passively, then
            connect only the arms you select.
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {yam.setup && <Badge tone={statusTone(yam.setup)} dot>{setupLabel(yam.setup)}</Badge>}
          <Button
            size="sm"
            icon={RefreshCw}
            disabled={busy}
            loading={yam.operation === "refresh"}
            onClick={() => void yam.refresh()}
          >
            Recheck
          </Button>
        </div>
      </div>

      {yam.loading && !yam.setup ? (
        <p className="mt-8 text-sm text-ink-muted">Loading YAM cell…</p>
      ) : !yam.setup ? (
        <Alert tone="danger" className="mt-6">
          {yam.error ?? "The YAM cell API is unavailable."}
        </Alert>
      ) : (
        <div className="mt-6 space-y-6">
          <Milestones setup={yam.setup} />

          {yam.error && (
            <Alert tone="danger" title="YAM cell error">
              {yam.stale ? "Showing the last known state. " : ""}
              {yam.error}
            </Alert>
          )}

          {!yam.setup.requires_physical_validation ? (
            <Alert tone="info" title="Deterministic mock cell">
              Mock mode uses a four-arm, two-pair cell. It does not discover, connect, calibrate, or
              validate physical hardware.
            </Alert>
          ) : (
            <Alert tone="warning" title="Physical behavior is unvalidated until the field gates pass">
              Directions, frame maps, limits, bus behavior, and emergency-stop operation remain
              unvalidated. Passive discovery and preflight never open a device or enable a motor.
            </Alert>
          )}

          {yam.setup.config && !isYamCellConfig(yam.setup.config) && (
            <Alert tone="warning" title="A V1.1 serial-GELLO setup remains readable">
              Discover and preflight a cell topology before replacing it; ctrl-π will not fabricate
              USB-CAN serial identities from its saved canN.
            </Alert>
          )}

          <section className="border-t border-line pt-6">
            <SectionHeading
              level={3}
              title="1 · Discover physical transports"
              description="Read-only OS inspection reports durable adapter identity separately from its current runtime interface. Roles are never inferred."
              actions={
                <Button
                  icon={Radar}
                  loading={yam.operation === "discover"}
                  disabled={busy}
                  onClick={() => void discover()}
                >
                  {yam.discovery ? "Rescan passively" : "Discover hardware"}
                </Button>
              }
            />
            {yam.discovery && (
              <div className="mt-4 space-y-3">
                <DeviceList
                  devices={yam.discovery.devices ?? []}
                  assigned={assigned}
                  onAssign={(device) => edit({ ...config, arms: [...config.arms, blankArm(device)] })}
                />
                <p className="text-2xs leading-5 text-ink-muted">{yam.discovery.detail}</p>
              </div>
            )}
          </section>

          <section className="border-t border-line pt-6">
            <SectionHeading
              level={3}
              title="2 · Define topology and pinned runtime"
              description="The i2rt path must be an operator-provided read-only checkout. The backend verifies the exact commit; it never substitutes public latest."
            />
            <div className="mt-4 grid gap-4 sm:grid-cols-3">
              <Field label="Cell name">
                <TextInput value={config.name} onChange={(event) => edit({ ...config, name: event.target.value })} />
              </Field>
              <Field label="Read-only i2rt checkout">
                <TextInput
                  className="font-mono"
                  value={config.i2rt_root}
                  placeholder="/opt/i2rt"
                  onChange={(event) => edit({ ...config, i2rt_root: event.target.value })}
                />
              </Field>
              <Field label="Pinned i2rt commit">
                <TextInput
                  className="font-mono"
                  value={config.i2rt_commit}
                  placeholder="40 lowercase hex characters"
                  onChange={(event) => edit({ ...config, i2rt_commit: event.target.value })}
                />
              </Field>
            </div>

            <div className="mt-5 space-y-4">
              {config.arms.map((arm, index) => (
                <ArmEditor
                  key={`${index}:${arm.stable_identity}`}
                  arm={arm}
                  runtimeInterface={
                    yam.discovery?.devices.find(
                      (device) =>
                        device.transport_kind === arm.transport_kind &&
                        device.stable_identity === arm.stable_identity,
                    )?.runtime_interface ??
                    yam.setup?.arms.find((status) => status.arm_id === arm.logical_id)?.runtime_interface ??
                    null
                  }
                  onChange={(next) =>
                    edit({
                      ...config,
                      arms: config.arms.map((item, itemIndex) => (itemIndex === index ? next : item)),
                    })
                  }
                  onRemove={() =>
                    edit({ ...config, arms: config.arms.filter((_, itemIndex) => itemIndex !== index) })
                  }
                />
              ))}
            </div>

            <Button
              className="mt-4"
              icon={Plus}
              onClick={() => edit({ ...config, arms: [...config.arms, blankArm()] })}
            >
              Add arm manually
            </Button>

            {pairIds.length > 0 && (
              <div className="mt-6 rounded-xl border border-line p-5">
                <h4 className="text-xs font-medium text-ink">Pair-specific ports</h4>
                <p className="mt-1 text-2xs leading-4 text-ink-muted">
                  Distinct ports keep pair routes isolated. Ports are cell configuration, never
                  global defaults.
                </p>
                <div className="mt-4 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                  {pairIds.map((pairId) => (
                    <Field key={pairId} label={pairId}>
                      <TextInput
                        type="number"
                        min={1024}
                        max={65535}
                        className="font-mono"
                        placeholder="optional"
                        value={config.pair_ports[pairId] ?? ""}
                        onChange={(event) => {
                          const value = Number(event.target.value);
                          const ports = { ...config.pair_ports };
                          if (event.target.value) ports[pairId] = value;
                          else delete ports[pairId];
                          edit({ ...config, pair_ports: ports });
                        }}
                      />
                    </Field>
                  ))}
                </div>
              </div>
            )}

            {issue && (
              <p className="mt-4 text-xs leading-5 text-caution-700">{issue}</p>
            )}
          </section>

          <section className="border-t border-line pt-6">
            <SectionHeading
              level={3}
              title="3 · Passive preflight and save"
              description="Validates identity resolution, link state, topology, pinned checkout, maps, and limits without opening hardware."
              actions={
                <Button
                  icon={FileCheck2}
                  loading={yam.operation === "preflight"}
                  disabled={busy || issue !== null}
                  onClick={() => void yam.check(normalized)}
                >
                  Run passive preflight
                </Button>
              }
            />

            {!hasArms && (
              <Alert tone="info" className="mt-4">
                An empty cell can be preflighted and saved as a topology draft after the exact i2rt
                checkout is verified. It is never connect-ready.
              </Alert>
            )}

            {yam.preflight && (
              <div className="mt-4 space-y-3">
                <Alert
                  tone={yam.preflight.ready ? "success" : "warning"}
                  title={yam.preflight.diagnostic.detail}
                />
                <div className="grid gap-3 sm:grid-cols-2">
                  {yam.preflight.arms.map((arm) => (
                    <div key={arm.arm_id} className="rounded-lg border border-line bg-surface px-4 py-3">
                      <div className="flex items-center justify-between gap-2">
                        <Mono>{arm.arm_id}</Mono>
                        <Badge tone={arm.ready ? "success" : "warning"}>
                          {arm.ready ? "Ready" : "Blocked"}
                        </Badge>
                      </div>
                      <p className="mt-2 text-2xs text-ink-muted">
                        runtime {arm.runtime_interface ?? "unresolved"} · map {arm.frame_map_status} ·
                        limits {arm.soft_limits_status} · handle {arm.handle_status}
                      </p>
                      {arm.warnings.map((warning) => (
                        <p
                          key={warning}
                          className={`mt-1 text-2xs font-medium ${
                            warning.includes("NO SASH GUARD") ? "text-critical-700" : "text-caution-700"
                          }`}
                        >
                          {warning}
                        </p>
                      ))}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {yam.setup.mode === "hardware" && (
              <Checkbox
                className="mt-4"
                checked={hasArms && autoRestore}
                disabled={busy || !hasArms}
                onChange={(event) => {
                  setAutoRestore(event.target.checked);
                  setDirty(true);
                }}
                label="Enable automatic connection"
                description={
                  <>
                    This is persistent consent to energize configured arms on boot/hot-plug.
                    Calibrated <InlineCode>linear_4310</InlineCode> and{" "}
                    <InlineCode>crank_4310</InlineCode> followers can also move their jaws during
                    connection.
                    {!hasArms && " Add at least one arm before automatic connection can be enabled."}
                  </>
                }
              />
            )}

            {revokingAutoRestoreOnly && (
              <Alert tone="success" className="mt-3">
                Revoking automatic connection is a database-only consent change. It does not require
                preflight, disconnect, or hardware access.
              </Alert>
            )}

            {yam.setup.mode === "hardware" && autoRestore && normalized.arms.some(requiresJawCalibrationConsent) && (
              <Checkbox
                className="mt-3"
                tone="warning"
                checked={jawsConfirmed}
                onChange={(event) => setJawsConfirmed(event.target.checked)}
                label="I understand automatic connection can calibrate each configured 4310 follower."
                description="The jaws will move; I will keep them and the arm workspace clear."
              />
            )}

            <div className="mt-5 flex flex-wrap gap-2">
              <Button
                variant="primary"
                icon={Save}
                loading={yam.operation === "save"}
                disabled={
                  busy ||
                  !preflightAllowsSave ||
                  (yam.setup.mode === "hardware" &&
                    autoRestore &&
                    normalized.arms.some(requiresJawCalibrationConsent) &&
                    !jawsConfirmed)
                }
                onClick={() => void save()}
              >
                {yam.setup.mode === "mock" ? "Apply mock cell" : "Save cell"}
              </Button>
              {dirty && isYamCellConfig(yam.setup.config) && (
                <Button
                  disabled={busy}
                  onClick={() => {
                    setConfig(cloneCell(yam.setup!.config as YamCellConfig));
                    setDirty(false);
                    yam.clearPreflight();
                  }}
                >
                  Discard edits
                </Button>
              )}
            </div>
          </section>

          <section className="border-t border-line pt-6">
            <SectionHeading
              level={3}
              title="4 · Connect or disconnect selected arms"
              description="Every CAN arm connect is motion-capable and may energize or resist motion. Disconnect stops writers and requests the safest supported release; wait for the displayed disconnected/non-energized state before treating an arm as limp."
            />

            <div className="mt-4 divide-y divide-line-subtle overflow-hidden rounded-xl border border-line">
              {yam.setup.arms.length === 0 ? (
                <p className="px-4 py-5 text-xs leading-5 text-ink-muted">
                  This saved cell is an empty topology draft. Add and save at least one arm before
                  connecting.
                </p>
              ) : (
                yam.setup.arms.map((arm) => (
                  <ArmConnectionRow
                    key={arm.arm_id}
                    arm={arm}
                    config={configById.get(arm.arm_id)}
                    checked={selectedArms.has(arm.arm_id)}
                    onChecked={(checked) =>
                      setSelectedArms((current) => {
                        const next = new Set(current);
                        if (checked) next.add(arm.arm_id);
                        else next.delete(arm.arm_id);
                        return next;
                      })
                    }
                  />
                ))
              )}
            </div>

            <div className="mt-3 flex flex-wrap gap-4">
              <button
                type="button"
                onClick={() => setSelectedArms(new Set(yam.setup!.arms.map((arm) => arm.arm_id)))}
                className="text-2xs font-medium text-accent-700 hover:text-accent-800"
              >
                Select all
              </button>
              <button
                type="button"
                onClick={() => setSelectedArms(new Set())}
                className="text-2xs font-medium text-ink-muted hover:text-ink"
              >
                Clear selection
              </button>
            </div>

            {yam.setup.requires_physical_validation && (
              <Checkbox
                className="mt-4"
                tone="warning"
                checked={motionConfirmed}
                onChange={(event) => setMotionConfirmed(event.target.checked)}
                label="I secured the workspace and verified the emergency stop."
                description="Connect can enable motors, start gravity compensation, and make every selected CAN arm resist manual motion."
              />
            )}

            {yam.setup.requires_physical_validation && selectedNeedJawAck && (
              <Checkbox
                className="mt-3"
                tone="danger"
                checked={jawsConfirmed}
                onChange={(event) => setJawsConfirmed(event.target.checked)}
                label="Connecting a selected linear_4310 or crank_4310 follower will calibrate its gripper."
                description="The arm controller is enabled and the jaws will move. Clear every selected jaw and arm workspace."
              />
            )}

            <div className="mt-5 flex flex-wrap gap-2">
              <Button
                variant="primary"
                icon={PlugZap}
                loading={yam.operation === "connect"}
                disabled={
                  busy ||
                  dirty ||
                  !hasArms ||
                  selectedArms.size === 0 ||
                  !yam.setup.calibration_ready ||
                  (yam.setup.requires_physical_validation &&
                    (!motionConfirmed || (selectedNeedJawAck && !jawsConfirmed)))
                }
                onClick={() => void connect()}
              >
                Connect selected
              </Button>
              <Button
                icon={PowerOff}
                loading={yam.operation === "disconnect"}
                disabled={
                  busy ||
                  selectedArms.size === 0 ||
                  !yam.setup.arms.some((arm) => selectedArms.has(arm.arm_id) && arm.connected)
                }
                onClick={() => void disconnect()}
              >
                Disconnect selected
              </Button>
            </div>
          </section>

          <section className="border-t border-line pt-6">
            <SectionHeading
              level={3}
              title="5 · Teaching-handle range check"
              description="CAN presence is not handle health. This separate active diagnostic observes trigger travel for 10 seconds; squeeze and release fully. It never re-zeros the encoder."
            />
            <div className="mt-4 grid gap-4 sm:grid-cols-[minmax(0,20rem)_1fr]">
              <Field label="Leader arm">
                <Select value={handleArmId} onChange={(event) => setHandleArmId(event.target.value)}>
                  <option value="">No configured leader</option>
                  {yam.setup.arms
                    .filter((arm) => arm.role === "leader")
                    .map((arm) => (
                      <option key={arm.arm_id} value={arm.arm_id}>
                        {configById.get(arm.arm_id)?.name ?? arm.arm_id} · {arm.pair_id ?? "unpaired"}
                      </option>
                    ))}
                </Select>
              </Field>
              {yam.setup.requires_physical_validation && (
                <Checkbox
                  checked={handleConfirmed}
                  onChange={(event) => setHandleConfirmed(event.target.checked)}
                  label="I understand this is an active CAN input diagnostic."
                  description="It will operate only the selected handle."
                />
              )}
            </div>
            <Button
              className="mt-4"
              icon={Activity}
              loading={yam.operation === "handle-check"}
              disabled={busy || !handleArmId || (yam.setup.requires_physical_validation && !handleConfirmed)}
              onClick={() =>
                void yam.checkHandle(handleArmId, 10, yam.setup?.mode === "hardware" && handleConfirmed)
              }
            >
              Run 10-second range check
            </Button>
            {yam.handleResult && (
              <Alert
                className="mt-4"
                tone={yam.handleResult.healthy ? "success" : "danger"}
                icon={yam.handleResult.healthy ? CheckCircle2 : TriangleAlert}
                title={yam.handleResult.detail}
              >
                <p className="font-mono text-2xs">
                  observed {yam.handleResult.observed_minimum?.toFixed(3) ?? "—"} …{" "}
                  {yam.handleResult.observed_maximum?.toFixed(3) ?? "—"}
                </p>
                {!yam.handleResult.healthy && (
                  <p className="mt-2 text-2xs leading-4">
                    Keep ctrl-π disconnected and use the documented i2rt CLI maintenance procedure
                    with the trigger mechanically released. There is intentionally no in-product
                    re-zero action.
                  </p>
                )}
              </Alert>
            )}
          </section>

          {yam.setup.saved && (
            <section className="border-t border-line pt-6">
              {!confirmForget ? (
                <Button variant="ghost" icon={Trash2} disabled={busy} onClick={() => setConfirmForget(true)}>
                  Forget saved YAM cell
                </Button>
              ) : (
                <Alert tone="danger" title="Forget the saved cell and disconnect its arms?">
                  <div className="mt-3 flex gap-2">
                    <Button variant="danger" size="sm" onClick={() => void forget()}>
                      Confirm forget
                    </Button>
                    <Button size="sm" onClick={() => setConfirmForget(false)}>
                      Cancel
                    </Button>
                  </div>
                </Alert>
              )}
            </section>
          )}
        </div>
      )}
    </section>
  );
}
