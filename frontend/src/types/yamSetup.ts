export type YamSetupConfig = {
  can_interface: string;
  leader_port: string;
  mujoco_xml_path: string;
  leader_calibration_id: string;
  leader_calibration_dir: string;
};

export type YamSetupDiagnostic = {
  status: "connected" | "configured" | "missing" | "error";
  detail: string;
};

export type YamSetupStatus = {
  mode: "mock" | "hardware";
  state: "needs_setup" | "awaiting_hardware" | "ready_to_connect" | "ready" | "error";
  saved: boolean;
  configured: boolean;
  connected: boolean;
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

export type YamSetupDiscovery = {
  mode: "mock" | "hardware";
  can_interfaces: YamSetupCandidate[];
  leader_ports: YamSetupCandidate[];
  suggested_config: YamSetupConfig | null;
  detail: string;
};

export type YamSetupPreflight = {
  ready: boolean;
  calibration_ready: boolean;
  diagnostic: YamSetupDiagnostic;
};

export type SaveYamSetupRequest = {
  config: YamSetupConfig;
  auto_restore: boolean;
  acknowledge_automatic_motion_risk: boolean;
};

export type ConnectYamSetupRequest = {
  acknowledge_hardware_motion_risk: boolean;
};
