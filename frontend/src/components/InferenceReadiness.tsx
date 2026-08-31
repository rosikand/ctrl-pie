import { Link } from "react-router-dom";

import { useSystemStatus } from "../state/systemStatus";
import { Alert } from "./ui/Alert";
import { Badge } from "./ui/Badge";
import { buttonClass } from "./ui/Button";
import { Skeleton } from "./ui/Skeleton";

/** True when a deployment may be created with the current server configuration. */
export function useInferenceReadiness() {
  const { status, loading, error, refresh } = useSystemStatus();
  const readiness = status?.inference ?? null;
  const ready = Boolean(
    readiness &&
      (readiness.mock_mode ||
        (readiness.hf_configured && readiness.modal_configured && readiness.modal_proxy_configured)),
  );
  return { readiness, ready, loading, error, refresh };
}

/**
 * One line about where policies actually run. Mock mode says so plainly; real
 * mode lists exactly which server-side credentials are still missing.
 */
export function InferenceReadinessNotice() {
  const { readiness, loading, error, refresh } = useInferenceReadiness();

  if (loading && !readiness) return <Skeleton className="h-14 w-full" />;

  if (!readiness) {
    return (
      <Alert
        tone="warning"
        title="Inference readiness is unavailable"
        action={
          <button
            type="button"
            onClick={refresh}
            className="text-xs font-medium underline underline-offset-2"
          >
            Retry
          </button>
        }
      >
        {error ?? "The backend did not report inference readiness."}
      </Alert>
    );
  }

  if (readiness.mock_mode) {
    return (
      <Alert tone="info" title="Mock runtime">
        LeRobot and OpenPI selections run through the deterministic local stub. No Hugging Face or
        Modal credentials are used, and no cloud resources are created.
      </Alert>
    );
  }

  const checks = [
    { label: "Hugging Face", ready: readiness.hf_configured },
    { label: "Modal API", ready: readiness.modal_configured },
    { label: "Proxy tokens", ready: readiness.modal_proxy_configured },
  ];
  const missing = checks.filter((check) => !check.ready);

  return (
    <Alert
      tone={missing.length ? "warning" : "neutral"}
      title={missing.length ? "Provider credentials are incomplete" : "Modal runtime ready"}
      action={
        <Link to="/settings" className={buttonClass("secondary", "sm")}>
          Settings
        </Link>
      }
    >
      <div className="flex flex-wrap items-center gap-1.5">
        {checks.map((check) => (
          <Badge key={check.label} tone={check.ready ? "success" : "warning"}>
            {check.label}: {check.ready ? "ready" : "missing"}
          </Badge>
        ))}
      </div>
    </Alert>
  );
}
