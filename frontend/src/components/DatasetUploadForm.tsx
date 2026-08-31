import { CheckCircle2, CloudUpload, ExternalLink, RefreshCw } from "lucide-react";
import { useState } from "react";
import type { FormEvent, ReactNode } from "react";
import { Link } from "react-router-dom";

import { hubRepoUrl, repoNameIssue } from "../lib/format";
import type { UploadRecordingResponse } from "../types/recordings";
import { Alert } from "./ui/Alert";
import { Button } from "./ui/Button";
import { Checkbox, Field } from "./ui/Form";
import { InlineCode } from "./ui/Code";

/**
 * The one dataset-upload form. Record and Inference both own their own request
 * lifecycle; the rules for namespace, naming, and public consent live here.
 */
export function DatasetUploadForm({
  namespace,
  namespaceError,
  onRetryNamespace,
  defaultRepoName,
  lockedRepoName,
  lockedNote,
  disabled = false,
  busy = false,
  error,
  result,
  blockingHint,
  primary = true,
  onSubmit,
}: {
  namespace: string | null | undefined;
  namespaceError?: string | null;
  onRetryNamespace?: () => void;
  defaultRepoName: string;
  lockedRepoName?: string | null;
  lockedNote?: ReactNode;
  disabled?: boolean;
  busy?: boolean;
  error?: string | null;
  result?: UploadRecordingResponse | null;
  blockingHint?: ReactNode;
  /** False when another step is the screen's single primary action. */
  primary?: boolean;
  onSubmit: (payload: { repo_name: string; private: boolean }) => void;
}) {
  const [repoName, setRepoName] = useState(lockedRepoName ?? defaultRepoName);
  const [isPrivate, setIsPrivate] = useState(true);
  const [publicConfirmed, setPublicConfirmed] = useState(false);

  const issue = repoNameIssue(repoName);
  const canSubmit =
    !disabled &&
    !busy &&
    !issue &&
    Boolean(namespace) &&
    (isPrivate || publicConfirmed) &&
    !blockingHint;

  if (result) {
    const url = hubRepoUrl(result.repo_id, "dataset");
    return (
      <Alert tone="success" icon={CheckCircle2} title="Dataset uploaded" role="status">
        <p className="font-mono text-2xs">{result.repo_id}</p>
        {result.revision && (
          <p className="mt-1 font-mono text-2xs">Revision {result.revision.slice(0, 12)}</p>
        )}
        {url && (
          <a
            href={url}
            target="_blank"
            rel="noreferrer"
            className="mt-2 inline-flex items-center gap-1 text-xs font-medium underline underline-offset-2"
          >
            Open dataset on Hugging Face
            <ExternalLink className="h-3 w-3" aria-hidden="true" />
          </a>
        )}
      </Alert>
    );
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!canSubmit) return;
    onSubmit({ repo_name: repoName.trim(), private: isPrivate });
  }

  return (
    <form onSubmit={submit} className="space-y-4">
      <Field
        label="Dataset repository"
        error={issue ?? undefined}
        hint={
          namespaceError
            ? "The backend settings endpoint could not be read."
            : namespace
              ? `Target: ${namespace}/${repoName.trim() || "…"}`
              : "HF_NAMESPACE is read from the backend environment."
        }
      >
        <div
          className={`flex min-w-0 items-center overflow-hidden rounded-lg border bg-surface focus-within:ring-4 ${
            issue
              ? "border-critical-500 focus-within:ring-critical-100"
              : "border-line focus-within:border-accent-500 focus-within:ring-accent-100"
          }`}
        >
          <span
            className="max-w-[45%] shrink-0 truncate border-r border-line bg-canvas px-2.5 py-2 font-mono text-2xs text-ink-muted"
            title={namespace ?? undefined}
          >
            {namespace === undefined
              ? "Loading namespace"
              : namespaceError
                ? "Settings unavailable"
                : namespace || "Not configured"}
            /
          </span>
          <input
            value={repoName}
            maxLength={96}
            disabled={disabled || busy || Boolean(lockedRepoName)}
            onChange={(event) => setRepoName(event.target.value)}
            aria-invalid={issue !== null}
            aria-label="Dataset repository name"
            className="min-w-0 flex-1 border-0 bg-transparent px-2.5 py-2 font-mono text-xs text-ink outline-none disabled:bg-canvas disabled:text-ink-muted"
          />
        </div>
      </Field>

      {lockedNote}

      <Checkbox
        label="Private dataset"
        description="Recommended. Access follows your Hugging Face account permissions."
        checked={isPrivate}
        disabled={disabled || busy}
        onChange={(event) => {
          setIsPrivate(event.target.checked);
          setPublicConfirmed(false);
        }}
      />

      {!isPrivate && (
        <Checkbox
          tone="warning"
          label="I understand this recording will be publicly accessible."
          description="Camera frames, robot state, actions, task text, and episode metadata will be visible to anyone."
          checked={publicConfirmed}
          disabled={busy}
          onChange={(event) => setPublicConfirmed(event.target.checked)}
        />
      )}

      {error && (
        <Alert tone="danger" title="Upload failed">
          {error}{" "}
          {lockedRepoName
            ? "You can retry with the same repository name."
            : "Choose an available repository name and retry."}
        </Alert>
      )}

      {namespaceError && (
        <Alert
          tone="danger"
          title="Settings unavailable"
          action={
            onRetryNamespace && (
              <Button size="sm" icon={RefreshCw} onClick={onRetryNamespace}>
                Retry
              </Button>
            )
          }
        >
          {namespaceError}
        </Alert>
      )}

      {!namespace && namespace !== undefined && !namespaceError && (
        <Alert tone="warning">
          Configure <InlineCode>HF_NAMESPACE</InlineCode> and <InlineCode>HF_TOKEN</InlineCode>, then
          recheck connections in{" "}
          <Link to="/settings" className="font-medium underline underline-offset-2">
            Settings
          </Link>
          .
        </Alert>
      )}

      {blockingHint && <p className="text-2xs leading-5 text-ink-muted">{blockingHint}</p>}

      <Button
        type="submit"
        variant={primary ? "primary" : "secondary"}
        icon={CloudUpload}
        loading={busy}
        disabled={!canSubmit}
        fullWidth
      >
        {busy ? "Uploading…" : error ? "Retry dataset upload" : "Upload dataset"}
      </Button>
    </form>
  );
}
