export type ArmRole = "leader" | "follower";

export type CanState = "active" | "warning" | "bus_off" | "disconnected";

export type JointTelemetry = {
  name: string;
  position_radians: number;
  velocity_radians_per_second: number;
  effort_newton_meters: number | null;
  temperature_celsius: number | null;
};

export type EndEffectorPose = {
  x_m: number;
  y_m: number;
  z_m: number;
  roll_radians: number;
  pitch_radians: number;
  yaw_radians: number;
};

export type GripperTelemetry = {
  position: number;
  velocity: number;
  force_newtons: number | null;
  is_closed: boolean;
};

export type CanTelemetry = {
  interface: string;
  state: CanState;
  bitrate: number;
  tx_error_count: number | null;
  rx_error_count: number | null;
};

export type ControlLoopTelemetry = {
  target_frequency_hz: number;
  frequency_hz: number;
  cycle_time_ms: number;
  jitter_ms: number;
  dropped_cycles: number;
};

export type ArmTelemetry = {
  id: string;
  name: string;
  role: ArmRole;
  driver: string;
  connected: boolean;
  timestamp: string;
  joints: JointTelemetry[];
  pose: EndEffectorPose;
  gripper: GripperTelemetry;
  can: CanTelemetry;
  control_loop: ControlLoopTelemetry;
};

export type ArmsResponse = {
  arms: ArmTelemetry[];
};

export type ArmsTelemetryMessage = ArmsResponse & {
  type: "telemetry";
  timestamp: string;
};

export type JogKind = "joint" | "cartesian" | "gripper";

export type JogCommand = {
  kind: JogKind;
  axis: string;
  delta: number;
};

export type TelemetryConnectionState =
  | "connecting"
  | "live"
  | "reconnecting"
  | "offline";
