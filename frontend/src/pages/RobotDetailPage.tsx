import { Bot, Minus, Plus, RefreshCw } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { Page, PageHeader, PageSection } from "../components/layout/Page";
import { Alert } from "../components/ui/Alert";
import { Badge, Mono, StatusDot } from "../components/ui/Badge";
import { Button, buttonClass, IconButton } from "../components/ui/Button";
import { DescriptionList } from "../components/ui/DescriptionList";
import { Disclosure, DisclosureGroup } from "../components/ui/Disclosure";
import { EmptyState } from "../components/ui/EmptyState";
import { Select } from "../components/ui/Form";
import { SectionHeading } from "../components/ui/Panel";
import { Skeleton } from "../components/ui/Skeleton";
import { Meter, Stat, StatGrid } from "../components/ui/Stat";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
} from "../components/ui/Table";
import { useArms } from "../hooks/useArms";
import {
  degrees,
  formatCount,
  formatTime,
  humanize,
  optionalFixed,
  optionalSigned,
  sentenceCase,
  signed,
} from "../lib/format";
import { controlStateTone, isBusyStatus } from "../lib/status";
import type { ArmTelemetry, JogCommand } from "../types/arms";
import { TelemetryBadge, robotSubtitle } from "./RobotsPage";

const jointStepOptions = [
  { label: "1°", value: Math.PI / 180 },
  { label: "5°", value: Math.PI / 36 },
  { label: "10°", value: Math.PI / 18 },
];

const translationAxes = ["x", "y", "z"] as const;
const rotationAxes = ["roll", "pitch", "yaw"] as const;

function busRate(bitrate: number | null): string {
  if (bitrate === null) return "bitrate unavailable";
  return bitrate >= 1_000_000
    ? `${(bitrate / 1_000_000).toFixed(0)} Mbps`
    : `${(bitrate / 1_000).toFixed(1)} kbps`;
}

function JointState({ arm }: { arm: ArmTelemetry }) {
  return (
    <Table label="Joint state" minWidth="42rem">
      <TableHead>
        <TableHeaderCell>Joint</TableHeaderCell>
        <TableHeaderCell align="right">Position</TableHeaderCell>
        <TableHeaderCell align="right">Radians</TableHeaderCell>
        <TableHeaderCell align="right">Velocity</TableHeaderCell>
        <TableHeaderCell align="right">Effort</TableHeaderCell>
        <TableHeaderCell align="right">Temp</TableHeaderCell>
      </TableHead>
      <TableBody>
        {arm.joints.map((joint, index) => (
          <TableRow key={joint.name}>
            <TableCell>
              <span className="flex items-center gap-2.5">
                <span className="grid h-5 w-5 place-items-center rounded bg-line-subtle text-2xs font-medium text-ink-muted">
                  {index + 1}
                </span>
                <span className="text-ink">{sentenceCase(humanize(joint.name))}</span>
              </span>
            </TableCell>
            <TableCell align="right" className="font-medium text-ink">
              {degrees(joint.position_radians)}
            </TableCell>
            <TableCell align="right" mono muted>
              {signed(joint.position_radians, 3)}
            </TableCell>
            <TableCell align="right" mono>
              {signed(joint.velocity_radians_per_second)} rad/s
            </TableCell>
            <TableCell align="right" mono>
              {optionalSigned(joint.effort_newton_meters, 2, " Nm")}
            </TableCell>
            <TableCell align="right" mono>
              {optionalFixed(joint.temperature_celsius, 1, "°C")}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

function PairAlignment({ arm, mate }: { arm: ArmTelemetry; mate: ArmTelemetry }) {
  const leader = arm.role === "leader" ? arm : mate;
  const follower = arm.role === "follower" ? arm : mate;
  if (follower.frame_map_active) {
    return (
      <Alert tone="warning" title="Mapped pair alignment">
        This pair has an active follower frame map, so raw leader and follower joint values are not
        directly comparable. Start observation-only teleop in Record and inspect its driver-prepared
        mapped deltas before approving synchronization.
      </Alert>
    );
  }
  const followerByName = new Map(
    follower.joints.map((joint) => [joint.name, joint.position_radians]),
  );
  return (
    <div>
      <p className="text-xs leading-5 text-ink-muted">
        Identity-mapped leader − follower physical joint deltas for pair {arm.pair_id}. Inspect
        before explicitly enabling sync in Record.
      </p>
      <div className="mt-4 grid grid-cols-3 gap-x-6 gap-y-4 sm:grid-cols-6">
        {leader.joints.map((joint, index) => {
          const delta =
            joint.position_radians - (followerByName.get(joint.name) ?? joint.position_radians);
          return (
            <div key={joint.name}>
              <p className="text-2xs font-medium uppercase tracking-[0.08em] text-ink-faint">
                J{index + 1}
              </p>
              <p className="mt-1 font-mono text-sm font-medium text-ink">{degrees(delta)}</p>
              <p className="mt-0.5 font-mono text-2xs text-ink-faint">{signed(delta, 3)} rad</p>
            </div>
          );
        })}
      </div>
    </div>
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
    <IconButton icon={Icon} label={label} size="sm" disabled={disabled} onClick={onClick} />
  );
}

function ManualJog({
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
    <div className="space-y-6">
      <div aria-live="polite" className="text-2xs font-medium">
        {error ? (
          <span className="text-critical-700">{error}</span>
        ) : pending ? (
          <span className="text-accent-700">Sending command…</span>
        ) : lastCommandAt ? (
          <span className="text-positive-700">Command applied</span>
        ) : disabled ? (
          <span className="text-caution-700">Controls require live telemetry</span>
        ) : (
          <span className="text-ink-muted">Ready · incremental commands apply to {arm.name}</span>
        )}
      </div>

      <div className="grid gap-8 lg:grid-cols-3">
        <div>
          <h4 className="text-xs font-medium text-ink">Joint</h4>
          <div className="mt-3 grid grid-cols-[1fr_5rem] gap-2">
            <Select
              aria-label="Jog axis"
              value={jointName}
              disabled={controlsDisabled}
              onChange={(event) => setJointName(event.target.value)}
              className="capitalize"
            >
              {arm.joints.map((joint) => (
                <option key={joint.name} value={joint.name}>
                  {humanize(joint.name)}
                </option>
              ))}
            </Select>
            <Select
              aria-label="Jog step"
              value={jointStep}
              disabled={controlsDisabled}
              onChange={(event) => setJointStep(Number(event.target.value))}
            >
              {jointStepOptions.map((step) => (
                <option key={step.label} value={step.value}>
                  {step.label}
                </option>
              ))}
            </Select>
          </div>
          <div className="mt-3 flex gap-2">
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

        <div>
          <h4 className="text-xs font-medium text-ink">Cartesian</h4>
          <p className="mt-1 text-2xs text-ink-muted">Translation 5 mm · rotation 5°</p>
          <div className="mt-3 grid gap-2 sm:grid-cols-2">
            {translationAxes.map((axis) => (
              <div key={axis} className="flex items-center gap-2">
                <span className="w-8 font-mono text-2xs uppercase text-ink-muted">{axis}</span>
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
              <div key={axis} className="flex items-center gap-2">
                <span className="w-8 font-mono text-2xs uppercase text-ink-muted">
                  {axis.slice(0, 3)}
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
        </div>

        <div>
          <h4 className="text-xs font-medium text-ink">Gripper</h4>
          <p className="mt-1 text-2xs text-ink-muted">10% incremental travel per command</p>
          <div className="mt-3 flex gap-2">
            <Button
              size="sm"
              disabled={controlsDisabled}
              onClick={() => onJog({ kind: "gripper", axis: "position", delta: 0.1 })}
            >
              Open
            </Button>
            <Button
              size="sm"
              variant="primary"
              disabled={controlsDisabled}
              onClick={() => onJog({ kind: "gripper", axis: "position", delta: -0.1 })}
            >
              Close
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

export function RobotDetailPage() {
  const { robotId = "" } = useParams<{ robotId: string }>();
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

  const arm = useMemo(() => arms.find((candidate) => candidate.id === robotId), [arms, robotId]);
  const pairMate = useMemo(
    () =>
      arm?.pair_id
        ? arms.find(
            (candidate) =>
              candidate.id !== arm.id &&
              candidate.pair_id === arm.pair_id &&
              candidate.role !== arm.role,
          )
        : undefined,
    [arm, arms],
  );

  if (!arm) {
    return (
      <Page>
        <PageHeader back={{ to: "/robots", label: "Robots" }} title={robotId || "Robot"} />
        <PageSection>
          {loading ? (
            <Skeleton className="h-64 w-full" />
          ) : (
            <EmptyState
              icon={Bot}
              title="Robot not found"
              description={
                error ?? "This robot is not exposed by the configured driver in the current cell."
              }
              action={
                <Link to="/robots" className={buttonClass("primary", "md")}>
                  Back to robots
                </Link>
              }
            />
          )}
        </PageSection>
      </Page>
    );
  }

  const controlsDisabled = connectionState !== "live" || !arm.connected;
  const loop = arm.control_loop;
  const txErrors = arm.can?.tx_error_count ?? null;
  const rxErrors = arm.can?.rx_error_count ?? null;
  const busErrors = txErrors === null || rxErrors === null ? null : txErrors + rxErrors;
  const gripperPercent = Math.max(0, Math.min(100, arm.gripper.position * 100));

  return (
    <Page>
      <PageHeader
        back={{ to: "/robots", label: "Robots" }}
        title={arm.name}
        description={`${arm.role === "leader" ? "Leader" : "Follower"} arm${
          robotSubtitle(arm) ? ` · ${robotSubtitle(arm)}` : ""
        }`}
        meta={
          <>
            <Badge
              tone={controlStateTone[arm.control_state]}
              dot
              pulse={isBusyStatus(arm.control_state)}
            >
              {humanize(arm.control_state)}
            </Badge>
            <TelemetryBadge state={connectionState} />
            <Mono title={arm.id}>{arm.id}</Mono>
          </>
        }
        actions={
          <IconButton
            icon={RefreshCw}
            label="Refresh telemetry"
            spinning={loading}
            disabled={loading}
            onClick={() => void refresh()}
          />
        }
      />

      {arm.warnings.length > 0 && (
        <PageSection className="mt-8 space-y-3">
          {arm.warnings.map((warning) => (
            <Alert
              key={warning}
              tone={warning.includes("NO SASH GUARD") ? "danger" : "warning"}
              title={warning}
            />
          ))}
        </PageSection>
      )}

      <PageSection className="mt-8">
        <StatGrid columns={4}>
          <Stat
            label="Connection"
            value={
              <span className="flex items-center gap-2 text-base">
                <StatusDot
                  tone={controlStateTone[arm.control_state]}
                  pulse={isBusyStatus(arm.control_state)}
                />
                {sentenceCase(humanize(arm.control_state))}
              </span>
            }
            hint={
              arm.energized
                ? arm.holding
                  ? "Energized · holding position"
                  : "Energized"
                : "Not energized"
            }
          />
          <Stat
            label="Device bus"
            value={
              <span className="text-base">
                {sentenceCase(arm.can ? humanize(arm.can.state) : arm.transport_kind)}
              </span>
            }
            hint={
              arm.can
                ? `${arm.can.interface} · ${busRate(arm.can.bitrate)}`
                : arm.transport_kind === "serial"
                  ? "Stable serial transport"
                  : "CAN unresolved"
            }
          />
          <Stat
            label="Control loop"
            value={`${loop.frequency_hz.toFixed(1)} Hz`}
            hint={`${loop.cycle_time_ms.toFixed(2)} ms cycle · ${loop.source} observation`}
          />
          <Stat
            label="Last update"
            value={<span className="text-base">{formatTime(arm.timestamp)}</span>}
            hint="Live telemetry snapshot"
          />
        </StatGrid>
      </PageSection>

      <PageSection>
        <SectionHeading
          title="Joint state"
          description="Position, velocity, effort, and motor temperature reported by the driver."
          className="mb-4"
        />
        <JointState arm={arm} />
      </PageSection>

      <PageSection>
        <SectionHeading title="Details" className="mb-3" />
        <DisclosureGroup>
          <Disclosure title="End-effector pose" meta="Cartesian position and orientation">
            <DescriptionList
              columns={3}
              items={[
                { label: "X", value: `${(arm.pose.x_m * 1_000).toFixed(1)} mm`, mono: true },
                { label: "Y", value: `${(arm.pose.y_m * 1_000).toFixed(1)} mm`, mono: true },
                { label: "Z", value: `${(arm.pose.z_m * 1_000).toFixed(1)} mm`, mono: true },
                { label: "Roll", value: degrees(arm.pose.roll_radians), mono: true },
                { label: "Pitch", value: degrees(arm.pose.pitch_radians), mono: true },
                { label: "Yaw", value: degrees(arm.pose.yaw_radians), mono: true },
              ]}
            />
          </Disclosure>

          {arm.handle ? (
            <Disclosure
              title="Teaching handle"
              meta={humanize(arm.handle.range_status)}
            >
              <DescriptionList
                items={[
                  {
                    label: "Trigger",
                    value: arm.handle.trigger_position?.toFixed(3) ?? "—",
                    mono: true,
                  },
                  {
                    label: "Encoder",
                    value: arm.handle.reachable ? "Reachable" : "Not verified",
                  },
                  {
                    label: "Observed range",
                    value: `${arm.handle.observed_minimum?.toFixed(3) ?? "—"} … ${
                      arm.handle.observed_maximum?.toFixed(3) ?? "—"
                    }`,
                    mono: true,
                  },
                  { label: "Range status", value: humanize(arm.handle.range_status) },
                ]}
              />
              <p className="mt-4 text-2xs leading-5 text-ink-muted">
                CAN connectivity and handle health are separate. Run the explicit range check in
                Settings; ctrl-π does not re-zero handles.
              </p>
              {arm.handle.calibration_warning && (
                <Alert tone="danger" className="mt-3">
                  {arm.handle.calibration_warning}
                </Alert>
              )}
            </Disclosure>
          ) : (
            <Disclosure
              title={arm.end_effector_kind === "linear_4310" ? "Linear jaw" : "Gripper"}
              meta={arm.gripper.is_closed ? "Closed" : "Open"}
            >
              <Meter
                value={gripperPercent}
                leadingLabel="Closed"
                trailingLabel="Open"
                label={`${gripperPercent.toFixed(0)}%`}
              />
              <DescriptionList
                className="mt-5"
                items={[
                  {
                    label: "Force",
                    value: optionalFixed(arm.gripper.force_newtons, 2, " N"),
                    mono: true,
                  },
                  { label: "Velocity", value: `${signed(arm.gripper.velocity)} /s`, mono: true },
                ]}
              />
              <p className="mt-4 text-2xs text-ink-muted">Normalized command: 0 closed · 1 open</p>
            </Disclosure>
          )}

          {pairMate && (
            <Disclosure title="Pair alignment" meta={`with ${pairMate.name}`}>
              <PairAlignment arm={arm} mate={pairMate} />
            </Disclosure>
          )}

          <Disclosure title="Loop diagnostics" meta={`${loop.frequency_hz.toFixed(1)} Hz`}>
            <DescriptionList
              columns={2}
              items={[
                {
                  label: "Loop frequency",
                  value: `${loop.frequency_hz.toFixed(1)} Hz`,
                  hint: `${loop.source} observation · reference only`,
                },
                {
                  label: "Cycle time",
                  value: `${loop.cycle_time_ms.toFixed(2)} ms`,
                  hint: `${loop.jitter_ms.toFixed(2)} ms jitter`,
                },
                {
                  label: "Dropped cycles",
                  value: formatCount(loop.dropped_cycles),
                  hint: "Since driver start",
                },
                {
                  label: arm.can ? "Bus errors" : "Bus counters",
                  value: busErrors === null ? "—" : formatCount(busErrors),
                  hint:
                    busErrors !== null
                      ? `${txErrors} TX · ${rxErrors} RX`
                      : arm.can
                        ? "Counters unavailable from driver"
                        : "Not a CAN transport",
                },
              ]}
            />
          </Disclosure>

          <Disclosure title="Identity and transport" meta={arm.transport_kind}>
            <DescriptionList
              items={[
                { label: "Logical ID", value: arm.id, mono: true },
                {
                  label: "Stable identity",
                  value: arm.stable_identity ?? "Not exposed",
                  mono: true,
                },
                {
                  label: "Transport",
                  value: `${arm.transport_kind} · runtime ${arm.can?.interface ?? "n/a"}`,
                },
                { label: "Driver", value: arm.driver, mono: true },
                { label: "End effector", value: arm.end_effector_kind, mono: true },
                {
                  label: "Frame map",
                  value: arm.frame_map_active ? "Active" : "Identity mapping",
                },
                {
                  label: "Soft limits",
                  value: arm.soft_limits_active ? "Active" : "NO SASH GUARD",
                },
                { label: "Pair", value: arm.pair_id ?? "Unpaired" },
              ]}
            />
          </Disclosure>

          <Disclosure title="Manual jog" meta={arm.driver === "i2rt-worker" ? "Unavailable" : "Bounded"}>
            {arm.driver === "i2rt-worker" ? (
              <section aria-label="Manual jog unavailable">
                <Alert tone="warning" title="Manual jog is unavailable for supervised i2rt cell arms">
                  One-shot jog commands are intentionally disabled because they cannot maintain the
                  worker&apos;s fresh-command watchdog. Use Record with its separate synchronization
                  boundary, or a supervised inference session, for continuously refreshed follower
                  motion. Mock and legacy drivers retain manual jog.
                </Alert>
              </section>
            ) : (
              <ManualJog
                key={arm.id}
                arm={arm}
                disabled={controlsDisabled}
                pending={commandPending}
                error={commandError}
                lastCommandAt={lastCommandAt}
                onJog={(command) => void sendJog(arm.id, command)}
              />
            )}
          </Disclosure>
        </DisclosureGroup>
      </PageSection>
    </Page>
  );
}
