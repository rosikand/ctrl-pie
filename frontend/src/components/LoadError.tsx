import { AlertCircle, Database, Lock, RefreshCw, WifiOff } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { Link } from "react-router-dom";

import { Alert } from "./ui/Alert";
import { Button, buttonClass } from "./ui/Button";
import { EmptyState } from "./ui/EmptyState";

export type LoadFailure = { message: string; status: number | null };

export type LoadResource =
  | "datasets"
  | "dataset"
  | "episode"
  | "models"
  | "runs"
  | "run"
  | "deployments"
  | "inference"
  | "robots";

const LABELS: Record<LoadResource, string> = {
  datasets: "Datasets",
  dataset: "This dataset",
  episode: "This episode",
  models: "Models",
  runs: "Training runs",
  run: "This training run",
  deployments: "Deployments",
  inference: "Inference state",
  robots: "Robots",
};

type Copy = { title: string; description: string; icon: LucideIcon; settings: boolean };

/** Turns an HTTP status into operator-facing copy, identically on every screen. */
export function describeFailure(error: LoadFailure, resource: LoadResource): Copy {
  const label = LABELS[resource];
  const hub = resource === "models" || resource === "datasets" || resource === "dataset";
  if (error.status === 503) {
    return {
      title: hub ? "Hugging Face is not configured" : `${label} storage is unavailable`,
      description: hub
        ? "Complete the server-side Hugging Face configuration before browsing this namespace."
        : "Connect the configured database before browsing this data.",
      icon: Database,
      settings: true,
    };
  }
  if (error.status === 403) {
    return {
      title: "Hugging Face access was denied",
      description:
        "The backend could not access the configured namespace. Verify its credentials and namespace access.",
      icon: Lock,
      settings: true,
    };
  }
  if (error.status === 502) {
    return {
      title: "Hugging Face is unavailable",
      description:
        "The backend could not reach the Hub. Nothing already loaded has been changed.",
      icon: WifiOff,
      settings: false,
    };
  }
  if (error.status === 404) {
    return {
      title: `${label} was not found`,
      description: "It may have been removed since this view was loaded.",
      icon: AlertCircle,
      settings: false,
    };
  }
  if (error.status === 409) {
    return {
      title: `${label} changed while loading`,
      description: "Reload to pick up the authoritative version.",
      icon: AlertCircle,
      settings: false,
    };
  }
  if (error.status === 422) {
    return {
      title: `${label} could not be read`,
      description: "The backend rejected the request parameters. Reload to start again.",
      icon: AlertCircle,
      settings: false,
    };
  }
  return {
    title: `${label} could not be loaded`,
    description: "The app could not reach the backend service. Check the connection and try again.",
    icon: WifiOff,
    settings: false,
  };
}

/** Full-width failure state, used when there is nothing else to show. */
export function LoadErrorState({
  error,
  resource,
  onRetry,
  busy = false,
  bordered = true,
}: {
  error: LoadFailure;
  resource: LoadResource;
  onRetry: () => void;
  busy?: boolean;
  bordered?: boolean;
}) {
  const copy = describeFailure(error, resource);
  return (
    <EmptyState
      role="alert"
      bordered={bordered}
      icon={copy.icon}
      title={copy.title}
      description={copy.description}
      detail={error.message}
      action={
        <>
          <Button variant="primary" icon={RefreshCw} loading={busy} onClick={onRetry}>
            Try again
          </Button>
          {copy.settings && (
            <Link to="/settings" className={buttonClass("secondary", "md")}>
              Open settings
            </Link>
          )}
        </>
      }
    />
  );
}

/** Inline failure bar, used when stale-but-valid content stays on screen. */
export function LoadErrorBar({
  error,
  resource,
  onRetry,
  busy = false,
  retryLabel = "Retry",
  className = "",
}: {
  error: LoadFailure;
  resource: LoadResource;
  onRetry: () => void;
  busy?: boolean;
  retryLabel?: string;
  className?: string;
}) {
  const copy = describeFailure(error, resource);
  return (
    <Alert
      tone="danger"
      title={copy.title}
      className={className}
      action={
        <Button size="sm" variant="secondary" loading={busy} onClick={onRetry}>
          {retryLabel}
        </Button>
      }
    >
      {error.message}
    </Alert>
  );
}
