export type YamArmRole = "leader" | "follower";
export type YamTransportKind = "socketcan" | "serial";
export type YamEndEffectorKind =
  | "yam_teaching_handle"
  | "linear_4310"
  | "crank_4310"
  | "gello"
  | "none";

/** V1.1 compatibility shape. New configuration should use YamCellConfig. */
export type LegacyYamSetupConfig = {
  kind?: "legacy_pair";
  can_interface: string;
  leader_port: string;
  mujoco_xml_path: string;
  leader_calibration_id: string;
  leader_calibration_dir: string;
};

export type YamCellArmConfig = {
  logical_id: string;
  name: string;
  role: YamArmRole;
  pair_id: string | null;
  group_id: string | null;
  side: string | null;
  transport_kind: YamTransportKind;
  stable_identity: string;
  end_effector_kind: YamEndEffectorKind;
  frame_map_path: string | null;
  soft_limits_path: string | null;
  mujoco_xml_path: string | null;
  calibration_id: string | null;
  calibration_dir: string | null;
};

export type YamCellConfig = {
  kind: "cell";
  name: string;
  i2rt_root: string;
  i2rt_commit: string;
  arms: YamCellArmConfig[];
  pair_ports: Record<string, number>;
};

export type YamSetupConfig = LegacyYamSetupConfig | YamCellConfig;

export function isYamCellConfig(config: YamSetupConfig | null): config is YamCellConfig {
  return config?.kind === "cell" && Array.isArray(config.arms);
}

export type YamSetupDiagnostic = {
  status: "connected" | "configured" | "missing" | "error";
  detail: string;
};

export type YamArmControlState =
  | "disconnected"
  | "connecting"
  | "gravity_comp"
  | "position_control"
  | "stopping"
  | "error";

export type YamSetupArmStatus = {
  arm_id: string;
  role: YamArmRole;
  pair_id: string | null;
  group_id: string | null;
  side: string | null;
  connected: boolean;
  control_state: YamArmControlState;
  energized: boolean;
  holding: boolean;
  runtime_interface: string | null;
  error: string | null;
};

export type YamSetupStatus = {
  mode: "mock" | "hardware";
  state:
    | "needs_setup"
    | "awaiting_hardware"
    | "ready_to_connect"
    | "partially_connected"
    | "ready"
    | "error";
  saved: boolean;
  configured: boolean;
  connected: boolean;
  any_connected: boolean;
  all_connected: boolean;
  configured_arm_count: number;
  connected_arm_count: number;
  arms: YamSetupArmStatus[];
  calibration_ready: boolean;
  auto_restore: boolean;
  restored_on_boot: boolean;
  config: YamSetupConfig | null;
  diagnostic: YamSetupDiagnostic;
  last_attempt_at: string | null;
  last_connected_at: string | null;
  requires_physical_validation: boolean;
};

export type YamSetupCandidate = {
  id: string;
  label: string;
};

export type YamDiscoveryDevice = {
  transport_kind: YamTransportKind;
  stable_identity: string;
  product: string | null;
  runtime_interface: string | null;
  link_state: "up" | "down" | "unknown" | "not_applicable";
  duplicate_identity: boolean;
};

export type YamArmResolution = {
  arm_id: string;
  transport_kind: YamTransportKind;
  stable_identity: string;
  runtime_interface: string | null;
  resolved: boolean;
  conflict: boolean;
  detail: string;
};

export type YamSetupDiscovery = {
  mode: "mock" | "hardware";
  devices: YamDiscoveryDevice[];
  resolutions: YamArmResolution[];
  /** Deprecated V1.1 discovery fields retained for old hardware adapters. */
  can_interfaces: YamSetupCandidate[];
  leader_ports: YamSetupCandidate[];
  suggested_config: YamSetupConfig | null;
  detail: string;
};

export type YamArmPreflight = {
  arm_id: string;
  ready: boolean;
  runtime_interface: string | null;
  link_state: "up" | "down" | "unknown" | "not_applicable";
  frame_map_status: "identity" | "active" | "error" | "not_applicable";
  soft_limits_status: "active" | "missing" | "error" | "not_applicable";
  handle_status: "not_checked" | "healthy" | "unhealthy" | "not_applicable";
  warnings: string[];
  diagnostic: YamSetupDiagnostic;
};

export type YamSetupPreflight = {
  ready: boolean;
  calibration_ready: boolean;
  diagnostic: YamSetupDiagnostic;
  i2rt_ready: boolean | null;
  arms: YamArmPreflight[];
  warnings: string[];
};

export type YamHandleRangeResult = {
  arm_id: string;
  reachable: boolean;
  observed_minimum: number | null;
  observed_maximum: number | null;
  healthy: boolean;
  detail: string;
};

export type SaveYamSetupRequest = {
  config: YamSetupConfig;
  auto_restore: boolean;
  acknowledge_automatic_motion_risk: boolean;
  /** Confirms jaw motion while calibrating configured linear_4310 or crank_4310 followers. */
  acknowledge_gripper_calibration_motion: boolean;
};

export type ConnectYamSetupRequest = {
  arm_ids: string[] | null;
  acknowledge_hardware_motion_risk: boolean;
  /** Confirms jaw motion while calibrating selected linear_4310 or crank_4310 followers. */
  acknowledge_gripper_calibration_motion: boolean;
};

export type DisconnectYamSetupRequest = {
  arm_ids: string[] | null;
};

export type YamHandleCheckRequest = {
  arm_id: string;
  duration_seconds: number;
  acknowledge_active_can_diagnostic: boolean;
};
