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
  bitrate: number | null;
  tx_error_count: number | null;
  rx_error_count: number | null;
};

export type ControlLoopTelemetry = {
  target_frequency_hz: number;
  frequency_hz: number;
  cycle_time_ms: number;
  jitter_ms: number;
  dropped_cycles: number;
  source: string;
};

export type TeachingHandleTelemetry = {
  reachable: boolean;
  trigger_position: number | null;
  buttons: boolean[];
  range_status: "not_tested" | "healthy" | "unhealthy";
  observed_minimum: number | null;
  observed_maximum: number | null;
  calibration_warning: string | null;
};

export type ArmControlState =
  | "disconnected"
  | "connecting"
  | "gravity_comp"
  | "position_control"
  | "stopping"
  | "error";

export type ArmTelemetry = {
  id: string;
  name: string;
  role: ArmRole;
  pair_id: string | null;
  group_id: string | null;
  side: string | null;
  transport_kind: "socketcan" | "serial" | "mock";
  stable_identity: string | null;
  end_effector_kind:
    | "yam_teaching_handle"
    | "linear_4310"
    | "crank_4310"
    | "gello"
    | "none";
  driver: string;
  connected: boolean;
  control_state: ArmControlState;
  energized: boolean;
  holding: boolean;
  timestamp: string;
  joints: JointTelemetry[];
  pose: EndEffectorPose;
  gripper: GripperTelemetry;
  can: CanTelemetry | null;
  control_loop: ControlLoopTelemetry;
  handle: TeachingHandleTelemetry | null;
  frame_map_active: boolean;
  soft_limits_active: boolean;
  warnings: string[];
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
