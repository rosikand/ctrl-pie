import type { Tone } from "../components/ui/Badge";
import type { ArmControlState, CanState } from "../types/arms";
import type { DeploymentStatus, InferenceSessionStatus } from "../types/inference";
import type { RecordingStatus } from "../types/recordings";
import type { ManagedTrainingJobStatus, TrainingRunStatus } from "../types/training";

/**
 * Every status in the product resolves to one of five tones here, so green,
 * amber, and red always mean the same thing across screens.
 */
export const recordingTone: Record<RecordingStatus, Tone> = {
  draft: "neutral",
  teleop: "info",
  recording: "danger",
  ready: "success",
  uploading: "warning",
  uploaded: "success",
  failed: "danger",
};

export const trainingTone: Record<TrainingRunStatus, Tone> = {
  created: "neutral",
  running: "info",
  completed: "success",
  failed: "danger",
  cancelled: "warning",
};

export const managedJobTone: Record<ManagedTrainingJobStatus, Tone> = {
  created: "neutral",
  launching: "info",
  running: "info",
  finalizing: "info",
  cancelling: "warning",
  completed: "success",
  failed: "danger",
  cancelled: "warning",
};

export const deploymentTone: Record<DeploymentStatus, Tone> = {
  created: "neutral",
  deploying: "warning",
  running: "success",
  stopping: "warning",
  stopped: "neutral",
  failed: "danger",
};

export const sessionTone: Record<InferenceSessionStatus, Tone> = {
  idle: "neutral",
  starting: "warning",
  running: "info",
  stopping: "warning",
  stopped: "neutral",
  failed: "danger",
};

export const controlStateTone: Record<ArmControlState, Tone> = {
  disconnected: "neutral",
  connecting: "warning",
  gravity_comp: "success",
  position_control: "info",
  stopping: "warning",
  error: "danger",
};

export const canStateTone: Record<CanState, Tone> = {
  active: "success",
  warning: "warning",
  bus_off: "danger",
  disconnected: "neutral",
};

const ACTIVE_STATUSES = new Set<string>([
  "recording",
  "teleop",
  "uploading",
  "running",
  "deploying",
  "stopping",
  "starting",
  "launching",
  "finalizing",
  "cancelling",
  "connecting",
]);

/** Whether a status dot should pulse because work is in flight. */
export function isBusyStatus(status: string): boolean {
  return ACTIVE_STATUSES.has(status);
}
