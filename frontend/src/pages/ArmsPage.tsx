import {
  Activity,
  AlertCircle,
  Bot,
  CheckCircle2,
  ChevronDown,
  CircleGauge,
  Gauge,
  Grip,
  Minus,
  Move3d,
  Network,
  Plus,
  RefreshCw,
  RotateCcw,
  Thermometer,
  Wifi,
  WifiOff,
  Zap,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { useArms } from "../hooks/useArms";
import type {
  ArmTelemetry,
  JogCommand,
  TelemetryConnectionState,
} from "../types/arms";

const jointStepOptions = [
  { label: "1°", value: Math.PI / 180 },
  { label: "5°", value: Math.PI / 36 },
  { label: "10°", value: Math.PI / 18 },
];

const translationAxes = ["x", "y", "z"] as const;
const rotationAxes = ["roll", "pitch", "yaw"] as const;

function degrees(radians: number): string {
  return `${((radians * 180) / Math.PI).toFixed(1)}°`;
}

function signed(value: number, digits = 2): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
}

function optionalSigned(
  value: number | null,
  digits: number,
  suffix: string,
): string {
  return value === null ? "—" : `${signed(value, digits)}${suffix}`;
}

function optionalFixed(
  value: number | null,
  digits: number,
  suffix: string,
): string {
  return value === null ? "—" : `${value.toFixed(digits)}${suffix}`;
}

function busRate(bitrate: number): string {
  return bitrate >= 1_000_000
    ? `${(bitrate / 1_000_000).toFixed(0)} Mbps`
    : `${(bitrate / 1_000).toFixed(1)} kbps`;
}

function telemetryTime(timestamp: string): string {
  const parsed = new Date(timestamp);
  return Number.isNaN(parsed.valueOf())
    ? "Waiting"
    : parsed.toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });
}

function StatusDot({ ready }: { ready: boolean }) {
  return (
    <span
      aria-hidden="true"
      className={`h-1.5 w-1.5 rounded-full ${ready ? "bg-emerald-500" : "bg-rose-500"}`}
    />
  );
}

function StreamBadge({ state }: { state: TelemetryConnectionState }) {
  const live = state === "live";
  const Icon = live ? Wifi : WifiOff;
  const labels: Record<TelemetryConnectionState, string> = {
    connecting: "Connecting",
    live: "Telemetry live",
    reconnecting: "Reconnecting",
    offline: "Offline",
  };

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-semibold ${
        live ? "bg-emerald-50 text-emerald-700" : "bg-amber-50 text-amber-700"
      }`}
    >
      <Icon className="h-3.5 w-3.5" strokeWidth={2} />
      {labels[state]}
    </span>
  );
}

function ArmSelector({
  arms,
  selectedId,
  onChange,
}: {
  arms: ArmTelemetry[];
  selectedId: string;
  onChange: (armId: string) => void;
}) {
  return (
    <div className="relative min-w-0 sm:w-72">
      <label htmlFor="arm-select" className="sr-only">
        Active arm
      </label>
      <Bot
        aria-hidden="true"
        className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400"
      />
      <select
        id="arm-select"
        value={selectedId}
        onChange={(event) => onChange(event.target.value)}
        className="w-full appearance-none rounded-lg border border-slate-200 bg-white py-2.5 pl-9 pr-9 text-sm font-medium text-slate-800 outline-none ring-brand-100 transition focus:border-brand-500 focus:ring-4"
      >
        {arms.map((arm) => (
          <option key={arm.id} value={arm.id}>
            {arm.name} · {arm.role}
          </option>
        ))}
      </select>
      <ChevronDown
        aria-hidden="true"
        className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400"
      />
    </div>
  );
}

function StatusStrip({ arm }: { arm: ArmTelemetry }) {
  const busHealthy = arm.can.state === "active";
  const items = [
    {
      label: "Connection",
      value: arm.connected ? "Connected" : "Disconnected",
      detail: arm.driver,
      ready: arm.connected,
      icon: Wifi,
    },
    {
      label: "Device bus",
      value: arm.can.state,
      detail: `${arm.can.interface} · ${busRate(arm.can.bitrate)}`,
      ready: busHealthy,
      icon: Network,
    },
    {
      label: "Role",
      value: arm.role,
      detail: "YAM arm",
      ready: true,
      icon: Bot,
    },
    {
      label: "Last update",
      value: telemetryTime(arm.timestamp),
      detail: "Live snapshot",
      ready: true,
      icon: Activity,
    },
  ];

  return (
    <section aria-label="Arm connection summary" className="grid divide-y divide-slate-100 rounded-xl border border-slate-200 bg-white shadow-panel sm:grid-cols-2 sm:divide-x sm:divide-y-0 xl:grid-cols-4">
      {items.map((item) => {
        const Icon = item.icon;
        return (
          <div key={item.label} className="flex items-start gap-3 px-4 py-4 sm:px-5">
            <div className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-slate-50 text-slate-500">
              <Icon className="h-4 w-4" strokeWidth={1.8} />
            </div>
            <div className="min-w-0">
              <p className="text-[11px] font-medium uppercase tracking-[0.1em] text-slate-400">
                {item.label}
              </p>
              <p className="mt-1 flex items-center gap-1.5 text-sm font-semibold capitalize text-slate-800">
                <StatusDot ready={item.ready} />
                <span className="truncate">{item.value}</span>
              </p>
              <p className="mt-0.5 truncate text-[11px] text-slate-400">{item.detail}</p>
            </div>
          </div>
        );
      })}
    </section>
  );
}

function JointState({ arm }: { arm: ArmTelemetry }) {
  return (
    <section className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-panel">
      <div className="flex items-center justify-between border-b border-slate-100 px-5 py-4 sm:px-6">
        <div>
          <h2 className="text-sm font-semibold text-slate-900">Joint state</h2>
          <p className="mt-1 text-xs text-slate-400">Position, velocity, effort, and motor temperature</p>
        </div>
        <span className="rounded-md bg-slate-100 px-2 py-1 font-mono text-[10px] text-slate-500">
          rad · SI
        </span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[620px] border-collapse text-left">
          <thead>
            <tr className="border-b border-slate-100 text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-400">
              <th className="px-5 py-3 sm:px-6">Joint</th>
              <th className="px-4 py-3">Position</th>
              <th className="px-4 py-3">Velocity</th>
              <th className="px-4 py-3">Effort</th>
              <th className="px-4 py-3 text-right sm:pr-6">Temp</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-50 font-mono text-xs text-slate-600">
            {arm.joints.map((joint, index) => (
              <tr key={joint.name} className="transition-colors hover:bg-slate-50/70">
                <td className="px-5 py-3 sm:px-6">
                  <div className="flex items-center gap-2 font-sans">
                    <span className="grid h-5 w-5 place-items-center rounded bg-slate-100 text-[9px] font-semibold text-slate-500">
                      J{index + 1}
                    </span>
                    <span className="font-medium capitalize text-slate-700">
                      {joint.name.replaceAll("_", " ")}
                    </span>
                  </div>
                </td>
                <td className="px-4 py-3">
                  <span className="font-semibold text-slate-800">{degrees(joint.position_radians)}</span>
                  <span className="ml-2 text-[10px] text-slate-400">
                    {signed(joint.position_radians)}
                  </span>
                </td>
                <td className="px-4 py-3">
                  {signed(joint.velocity_radians_per_second)}
                  <span className="ml-1 text-[10px] text-slate-400">rad/s</span>
                </td>
                <td className="px-4 py-3">
                  {optionalSigned(joint.effort_newton_meters, 2, " Nm")}
                </td>
                <td className="px-4 py-3 text-right sm:pr-6">
                  <span className="inline-flex items-center gap-1">
                    <Thermometer className="h-3 w-3 text-slate-300" />
                    {optionalFixed(joint.temperature_celsius, 1, "°C")}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function PoseCard({ arm }: { arm: ArmTelemetry }) {
  const values = [
    { label: "X", value: `${(arm.pose.x_m * 1_000).toFixed(1)} mm` },
    { label: "Y", value: `${(arm.pose.y_m * 1_000).toFixed(1)} mm` },
    { label: "Z", value: `${(arm.pose.z_m * 1_000).toFixed(1)} mm` },
    { label: "Roll", value: degrees(arm.pose.roll_radians) },
    { label: "Pitch", value: degrees(arm.pose.pitch_radians) },
    { label: "Yaw", value: degrees(arm.pose.yaw_radians) },
  ];

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-panel sm:p-6">
      <div className="flex items-center gap-2">
        <Move3d className="h-4 w-4 text-slate-400" strokeWidth={1.8} />
        <h2 className="text-sm font-semibold text-slate-900">End-effector pose</h2>
      </div>
      <div className="mt-4 grid grid-cols-3 gap-px overflow-hidden rounded-lg border border-slate-100 bg-slate-100">
        {values.map((item) => (
          <div key={item.label} className="bg-white px-3 py-3">
            <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">{item.label}</p>
            <p className="mt-1 whitespace-nowrap font-mono text-xs font-semibold text-slate-700">
              {item.value}
            </p>
          </div>
        ))}
      </div>
    </section>
  );
}

function GripperCard({ arm }: { arm: ArmTelemetry }) {
  const percent = Math.max(0, Math.min(100, arm.gripper.position * 100));

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-panel sm:p-6">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Grip className="h-4 w-4 text-slate-400" strokeWidth={1.8} />
          <h2 className="text-sm font-semibold text-slate-900">Gripper</h2>
        </div>
        <span className={`rounded-full px-2 py-1 text-[10px] font-semibold ${arm.gripper.is_closed ? "bg-blue-50 text-blue-700" : "bg-slate-100 text-slate-600"}`}>
          {arm.gripper.is_closed ? "Closed" : "Open"}
        </span>
      </div>
      <div className="mt-5">
        <div className="mb-2 flex items-center justify-between text-[11px] text-slate-400">
          <span>Closed</span>
          <span className="font-mono font-semibold text-slate-700">{percent.toFixed(0)}%</span>
          <span>Open</span>
        </div>
        <div className="h-2 overflow-hidden rounded-full bg-slate-100">
          <div className="h-full rounded-full bg-brand-500 transition-[width] duration-150" style={{ width: `${percent}%` }} />
        </div>
      </div>
      <div className="mt-5 grid grid-cols-2 gap-3">
        <div className="rounded-lg bg-slate-50 px-3 py-2.5">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Force</p>
          <p className="mt-1 font-mono text-xs font-semibold text-slate-700">
            {optionalFixed(arm.gripper.force_newtons, 2, " N")}
          </p>
        </div>
        <div className="rounded-lg bg-slate-50 px-3 py-2.5">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Velocity</p>
          <p className="mt-1 font-mono text-xs font-semibold text-slate-700">
            {signed(arm.gripper.velocity)} /s
          </p>
        </div>
      </div>
    </section>
  );
}

function Diagnostics({ arm }: { arm: ArmTelemetry }) {
  const loop = arm.control_loop;
  const targetDelta = Math.abs(loop.frequency_hz - loop.target_frequency_hz);
  const onTarget = targetDelta <= loop.target_frequency_hz * 0.05;
  const txErrors = arm.can.tx_error_count;
  const rxErrors = arm.can.rx_error_count;
  const errors =
    txErrors === null || rxErrors === null ? null : txErrors + rxErrors;
  const stats = [
    {
      label: "Loop frequency",
      value: `${loop.frequency_hz.toFixed(1)} Hz`,
      detail: `Target ${loop.target_frequency_hz.toFixed(0)} Hz`,
      icon: CircleGauge,
      healthy: onTarget,
    },
    {
      label: "Cycle time",
      value: `${loop.cycle_time_ms.toFixed(2)} ms`,
      detail: `${loop.jitter_ms.toFixed(2)} ms jitter`,
      icon: Gauge,
      healthy: loop.jitter_ms < 2,
    },
    {
      label: "Dropped cycles",
      value: loop.dropped_cycles.toLocaleString(),
      detail: "Since driver start",
      icon: Activity,
      healthy: loop.dropped_cycles === 0,
    },
    {
      label: "Bus errors",
      value: errors === null ? "—" : errors.toLocaleString(),
      detail: errors !== null
        ? `${txErrors} TX · ${rxErrors} RX`
        : "Counters unavailable from driver",
      icon: Zap,
      healthy: errors === null ? null : errors === 0,
    },
  ];

  return (
    <section className="rounded-xl border border-slate-200 bg-white shadow-panel">
      <div className="border-b border-slate-100 px-5 py-4 sm:px-6">
        <h2 className="text-sm font-semibold text-slate-900">Loop diagnostics</h2>
        <p className="mt-1 text-xs text-slate-400">Driver timing and bus health</p>
      </div>
      <div className="grid divide-y divide-slate-100 sm:grid-cols-2 sm:divide-x sm:divide-y-0 xl:grid-cols-4">
        {stats.map((stat) => {
          const Icon = stat.icon;
          return (
            <div key={stat.label} className="px-5 py-4 sm:px-6">
              <div className="flex items-center justify-between">
                <Icon className="h-4 w-4 text-slate-400" strokeWidth={1.8} />
                {stat.healthy === null ? (
                  <span
                    aria-label="Unavailable"
                    className="font-mono text-xs text-slate-300"
                  >
                    —
                  </span>
                ) : stat.healthy ? (
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
                ) : (
                  <AlertCircle className="h-3.5 w-3.5 text-amber-500" />
                )}
              </div>
              <p className="mt-3 font-mono text-lg font-semibold tracking-tight text-slate-800">
                {stat.value}
              </p>
              <p className="mt-1 text-[11px] font-medium text-slate-500">{stat.label}</p>
              <p className="mt-0.5 text-[10px] text-slate-400">{stat.detail}</p>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function JogButton({
  label,
  direction,
  disabled,
  onClick,
}: {
  label: string;
  direction: "negative" | "positive";
  disabled: boolean;
  onClick: () => void;
}) {
  const Icon = direction === "negative" ? Minus : Plus;
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      aria-label={label}
      className="inline-flex h-9 min-w-10 items-center justify-center rounded-lg border border-slate-200 bg-white px-3 text-slate-600 shadow-sm transition hover:border-slate-300 hover:bg-slate-50 hover:text-slate-900 active:translate-y-px disabled:cursor-not-allowed disabled:opacity-40"
    >
      <Icon className="h-4 w-4" strokeWidth={2} />
    </button>
  );
}

function ManualControls({
  arm,
  disabled,
  pending,
  error,
  lastCommandAt,
  onJog,
}: {
  arm: ArmTelemetry;
  disabled: boolean;
  pending: boolean;
  error: string | null;
  lastCommandAt: string | null;
  onJog: (command: JogCommand) => void;
}) {
  const [jointName, setJointName] = useState(arm.joints[0]?.name ?? "");
  const [jointStep, setJointStep] = useState(Math.PI / 36);

  useEffect(() => {
    if (!arm.joints.some((joint) => joint.name === jointName)) {
      setJointName(arm.joints[0]?.name ?? "");
    }
  }, [arm.id, arm.joints, jointName]);

  const controlsDisabled = disabled || pending;

  return (
    <section className="rounded-xl border border-slate-200 bg-white shadow-panel">
      <div className="flex flex-col gap-3 border-b border-slate-100 px-5 py-4 sm:flex-row sm:items-center sm:justify-between sm:px-6">
        <div>
          <h2 className="text-sm font-semibold text-slate-900">Manual jog</h2>
          <p className="mt-1 text-xs text-slate-400">Incremental commands apply to {arm.name}</p>
        </div>
        <div aria-live="polite" className="text-[11px] font-medium">
          {error ? (
            <span className="inline-flex items-center gap-1.5 text-rose-600">
              <AlertCircle className="h-3.5 w-3.5" /> {error}
            </span>
          ) : pending ? (
            <span className="inline-flex items-center gap-1.5 text-brand-600">
              <RefreshCw className="h-3.5 w-3.5 animate-spin" /> Sending command
            </span>
          ) : lastCommandAt ? (
            <span className="inline-flex items-center gap-1.5 text-emerald-600">
              <CheckCircle2 className="h-3.5 w-3.5" /> Command applied
            </span>
          ) : disabled ? (
            <span className="text-amber-600">Controls require live telemetry</span>
          ) : (
            <span className="text-slate-400">Ready</span>
          )}
        </div>
      </div>

      <div className="grid divide-y divide-slate-100 xl:grid-cols-[1fr_1.35fr_0.8fr] xl:divide-x xl:divide-y-0">
        <div className="p-5 sm:p-6">
          <div className="flex items-center gap-2">
            <RotateCcw className="h-4 w-4 text-slate-400" strokeWidth={1.8} />
            <h3 className="text-xs font-semibold uppercase tracking-[0.1em] text-slate-600">Joint</h3>
          </div>
          <div className="mt-4 grid grid-cols-[1fr_auto] gap-2">
            <label className="block text-[10px] font-medium uppercase tracking-wider text-slate-400">
              Axis
              <select
                value={jointName}
                onChange={(event) => setJointName(event.target.value)}
                disabled={controlsDisabled}
                className="mt-1.5 w-full rounded-lg border border-slate-200 bg-white px-2.5 py-2 text-xs font-medium capitalize text-slate-700 outline-none focus:border-brand-500 disabled:bg-slate-50"
              >
                {arm.joints.map((joint) => (
                  <option key={joint.name} value={joint.name}>
                    {joint.name.replaceAll("_", " ")}
                  </option>
                ))}
              </select>
            </label>
            <label className="block text-[10px] font-medium uppercase tracking-wider text-slate-400">
              Step
              <select
                value={jointStep}
                onChange={(event) => setJointStep(Number(event.target.value))}
                disabled={controlsDisabled}
                className="mt-1.5 rounded-lg border border-slate-200 bg-white px-2.5 py-2 text-xs font-medium text-slate-700 outline-none focus:border-brand-500 disabled:bg-slate-50"
              >
                {jointStepOptions.map((step) => (
                  <option key={step.label} value={step.value}>
                    {step.label}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="mt-3 grid grid-cols-2 gap-2">
            <JogButton
              label={`Jog ${jointName} negative`}
              direction="negative"
              disabled={controlsDisabled || !jointName}
              onClick={() => onJog({ kind: "joint", axis: jointName, delta: -jointStep })}
            />
            <JogButton
              label={`Jog ${jointName} positive`}
              direction="positive"
              disabled={controlsDisabled || !jointName}
              onClick={() => onJog({ kind: "joint", axis: jointName, delta: jointStep })}
            />
          </div>
        </div>

        <div className="p-5 sm:p-6">
          <div className="flex items-center gap-2">
            <Move3d className="h-4 w-4 text-slate-400" strokeWidth={1.8} />
            <h3 className="text-xs font-semibold uppercase tracking-[0.1em] text-slate-600">Cartesian</h3>
          </div>
          <div className="mt-4 grid gap-2 sm:grid-cols-2">
            {translationAxes.map((axis) => (
              <div key={axis} className="grid grid-cols-[2.25rem_1fr_1fr] items-center gap-1.5">
                <span className="font-mono text-xs font-semibold uppercase text-slate-500">{axis}</span>
                <JogButton
                  label={`Jog ${axis} negative 5 millimeters`}
                  direction="negative"
                  disabled={controlsDisabled}
                  onClick={() => onJog({ kind: "cartesian", axis, delta: -0.005 })}
                />
                <JogButton
                  label={`Jog ${axis} positive 5 millimeters`}
                  direction="positive"
                  disabled={controlsDisabled}
                  onClick={() => onJog({ kind: "cartesian", axis, delta: 0.005 })}
                />
              </div>
            ))}
            {rotationAxes.map((axis) => (
              <div key={axis} className="grid grid-cols-[2.25rem_1fr_1fr] items-center gap-1.5">
                <span className="font-mono text-[10px] font-semibold uppercase text-slate-500">
                  {axis.slice(0, 1)}{axis === "yaw" ? "y" : axis === "pitch" ? "p" : "r"}
                </span>
                <JogButton
                  label={`Jog ${axis} negative 5 degrees`}
                  direction="negative"
                  disabled={controlsDisabled}
                  onClick={() => onJog({ kind: "cartesian", axis, delta: -Math.PI / 36 })}
                />
                <JogButton
                  label={`Jog ${axis} positive 5 degrees`}
                  direction="positive"
                  disabled={controlsDisabled}
                  onClick={() => onJog({ kind: "cartesian", axis, delta: Math.PI / 36 })}
                />
              </div>
            ))}
          </div>
          <p className="mt-3 text-[10px] text-slate-400">Translation 5 mm · rotation 5°</p>
        </div>

        <div className="p-5 sm:p-6">
          <div className="flex items-center gap-2">
            <Grip className="h-4 w-4 text-slate-400" strokeWidth={1.8} />
            <h3 className="text-xs font-semibold uppercase tracking-[0.1em] text-slate-600">Gripper</h3>
          </div>
          <div className="mt-4 grid grid-cols-2 gap-2">
            <button
              type="button"
              disabled={controlsDisabled}
              onClick={() => onJog({ kind: "gripper", axis: "position", delta: 0.1 })}
              className="rounded-lg border border-slate-200 bg-white px-3 py-2.5 text-xs font-semibold text-slate-600 shadow-sm transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Open
            </button>
            <button
              type="button"
              disabled={controlsDisabled}
              onClick={() => onJog({ kind: "gripper", axis: "position", delta: -0.1 })}
              className="rounded-lg bg-ink px-3 py-2.5 text-xs font-semibold text-white shadow-sm transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Close
            </button>
          </div>
          <p className="mt-3 text-[10px] text-slate-400">10% incremental travel per command</p>
        </div>
      </div>
    </section>
  );
}

function EmptyArms({ loading, error, refresh }: { loading: boolean; error: string | null; refresh: () => void }) {
  return (
    <section className="mt-8 grid min-h-80 place-items-center rounded-xl border border-slate-200 bg-white px-6 text-center shadow-panel">
      <div>
        <div className="mx-auto grid h-11 w-11 place-items-center rounded-xl bg-slate-100 text-slate-500">
          {error ? <WifiOff className="h-5 w-5" /> : <Bot className="h-5 w-5" strokeWidth={1.7} />}
        </div>
        <p className="mt-4 text-sm font-semibold text-slate-800">
          {loading ? "Loading arms…" : error ? "Arms API unavailable" : "No arms found"}
        </p>
        <p className="mx-auto mt-1 max-w-sm text-sm leading-6 text-slate-400">
          {error ?? "No arms are exposed by the configured driver."}
        </p>
        {!loading && (
          <button
            type="button"
            onClick={refresh}
            className="mt-4 inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-600 shadow-sm hover:bg-slate-50"
          >
            <RefreshCw className="h-3.5 w-3.5" /> Retry
          </button>
        )}
      </div>
    </section>
  );
}

export function ArmsPage() {
  const {
    arms,
    loading,
    error,
    refresh,
    connectionState,
    commandPending,
    commandError,
    lastCommandAt,
    sendJog,
  } = useArms();
  const [selectedId, setSelectedId] = useState("");

  useEffect(() => {
    if (!arms.some((arm) => arm.id === selectedId)) {
      setSelectedId(arms[0]?.id ?? "");
    }
  }, [arms, selectedId]);

  const arm = useMemo(
    () => arms.find((candidate) => candidate.id === selectedId) ?? arms[0],
    [arms, selectedId],
  );
  const controlsDisabled = connectionState !== "live" || !arm?.connected;

  return (
    <div className="mx-auto max-w-7xl px-5 py-8 sm:px-8 lg:px-10 lg:py-10">
      <header className="flex flex-col gap-5 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-brand-600">Hardware</p>
          <h1 className="mt-2 text-2xl font-semibold tracking-tight text-slate-950 sm:text-3xl">Arms</h1>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">
            Inspect live YAM telemetry and apply small manual jog commands.
          </p>
        </div>
        <div className="flex items-center gap-2 self-start">
          <StreamBadge state={connectionState} />
          <button
            type="button"
            aria-label="Refresh arms"
            title="Refresh arms"
            onClick={() => void refresh()}
            disabled={loading}
            className="grid h-8 w-8 place-items-center rounded-lg border border-slate-200 bg-white text-slate-500 shadow-sm transition hover:bg-slate-50 disabled:opacity-50"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
          </button>
        </div>
      </header>

      {arm ? (
        <>
          <div className="mt-7 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-400">Active arm</p>
              <p className="mt-1 text-xs text-slate-500">
                {arms.length} arm{arms.length === 1 ? "" : "s"} available
              </p>
            </div>
            <ArmSelector arms={arms} selectedId={arm.id} onChange={setSelectedId} />
          </div>

          <div className="mt-4 space-y-5">
            <StatusStrip arm={arm} />
            <div className="grid gap-5 xl:grid-cols-[minmax(0,1.5fr)_minmax(300px,0.8fr)]">
              <JointState arm={arm} />
              <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-1">
                <PoseCard arm={arm} />
                <GripperCard arm={arm} />
              </div>
            </div>
            <Diagnostics arm={arm} />
            <ManualControls
              key={arm.id}
              arm={arm}
              disabled={controlsDisabled}
              pending={commandPending}
              error={commandError}
              lastCommandAt={lastCommandAt}
              onJog={(command) => void sendJog(arm.id, command)}
            />
          </div>
        </>
      ) : (
        <EmptyArms loading={loading} error={error} refresh={() => void refresh()} />
      )}
    </div>
  );
}
