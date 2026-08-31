import { Activity, ExternalLink, Terminal } from "lucide-react";
import { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { Page, PageHeader, PageSection } from "../components/layout/Page";
import { LoadErrorState } from "../components/LoadError";
import { Alert } from "../components/ui/Alert";
import { Badge, Mono } from "../components/ui/Badge";
import { buttonClass } from "../components/ui/Button";
import { DescriptionList } from "../components/ui/DescriptionList";
import { EmptyState } from "../components/ui/EmptyState";
import { LineChart } from "../components/ui/LineChart";
import { Panel, PanelHeader, SectionHeading } from "../components/ui/Panel";
import { Skeleton } from "../components/ui/Skeleton";
import { Stat, StatGrid } from "../components/ui/Stat";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
} from "../components/ui/Table";
import { TabPanel, Tabs } from "../components/ui/Tabs";
import { useTrainingRun } from "../hooks/useTrainingRuns";
import {
  appendHubPath,
  formatCount,
  formatDateTime,
  formatRelative,
  formatScalar,
  formatTime,
  hubRepoUrl,
  shortRevision,
} from "../lib/format";
import { isBusyStatus, managedJobTone, trainingTone } from "../lib/status";
import type {
  ManagedTrainingJobSummary,
  TrainingConsoleLog,
  TrainingRun,
} from "../types/training";

type RunTab = "metrics" | "console" | "checkpoints" | "config";

function ArtifactLink({
  repoId,
  kind = "model",
}: {
  repoId: string | null;
  kind?: "model" | "dataset";
}) {
  const url = hubRepoUrl(repoId, kind);
  if (!repoId) return <span className="text-ink-faint">Not reported</span>;
  return url ? (
    <a
      href={url}
      target="_blank"
      rel="noreferrer"
      className="inline-flex min-w-0 items-center gap-1.5 font-mono text-accent-700 hover:text-accent-800"
    >
      <span className="truncate" title={repoId}>
        {repoId}
      </span>
      <ExternalLink className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
    </a>
  ) : (
    <span className="break-all font-mono">{repoId}</span>
  );
}

/** Managed compute lifecycle: provider state, artifact identity, and teardown. */
function ManagedJobPanel({ job }: { job: ManagedTrainingJobSummary }) {
  const outputUrl = hubRepoUrl(job.output_model_repo);
  const verifiedRevision = job.output_revision ?? job.output_marker_revision;
  const revisionUrl =
    outputUrl && verifiedRevision ? appendHubPath(outputUrl, ["tree", verifiedRevision]) : null;
  const terminal =
    job.status === "completed" || job.status === "failed" || job.status === "cancelled";

  return (
    <Panel>
      <PanelHeader
        title={job.target_kind === "modal" ? "Managed Modal training" : "Managed training simulation"}
        description={job.id}
        actions={
          <Badge tone={managedJobTone[job.status]} dot pulse={isBusyStatus(job.status)}>
            {job.status}
          </Badge>
        }
      />
      <div className="px-5 py-5">
        <DescriptionList
          columns={3}
          items={[
            {
              label: "Compute",
              value: job.target_kind === "modal" ? job.compute_size : "Stub · no GPU",
            },
            { label: "Provider state", value: job.provider_state },
            { label: "Outcome", value: job.outcome },
            { label: "Hard deadline", value: formatDateTime(job.deadline_at) },
            {
              label: "Output artifact",
              value:
                job.target_kind === "modal" && revisionUrl ? (
                  <a
                    href={revisionUrl}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex max-w-full items-center gap-1.5 font-mono text-accent-700 hover:text-accent-800"
                  >
                    <span className="truncate" title={job.output_model_repo}>
                      {job.output_model_repo}
                    </span>
                    <span className="shrink-0 text-2xs text-ink-faint">
                      @{shortRevision(verifiedRevision, 12)}
                    </span>
                    <ExternalLink className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                  </a>
                ) : job.target_kind === "modal" ? (
                  <span className="break-all font-mono">{job.output_model_repo}</span>
                ) : (
                  <span className="text-ink-muted">Simulated · no Hub artifact created</span>
                ),
              hint:
                job.target_kind === "modal" && !revisionUrl
                  ? "Requested repo · existence not yet verified"
                  : undefined,
            },
            {
              label: "Compute teardown",
              value: (
                <span
                  className={
                    job.teardown_verified
                      ? "text-positive-700"
                      : terminal
                        ? "text-critical-700"
                        : "text-ink-secondary"
                  }
                >
                  {job.teardown_verified
                    ? job.target_kind === "stub"
                      ? "Simulation teardown complete · no provider tasks"
                      : "Provider-verified stopped · zero tasks"
                    : terminal
                      ? "Cleanup is not verified"
                      : "Cleanup verification pending"}
                </span>
              ),
            },
          ]}
        />
        {job.event_gap && (
          <Alert tone="warning" role="status" className="mt-5">
            Part of the bounded provider event stream was unavailable after reconnect. Stored
            metrics, checkpoints, and output remain visible, but this history is incomplete.
          </Alert>
        )}
        {job.last_error && (
          <Alert tone="danger" className="mt-3">
            {job.last_error}
          </Alert>
        )}
      </div>
    </Panel>
  );
}

function ConsoleOutput({
  logs,
  loading,
  truncated,
  error,
  onRetry,
}: {
  logs: TrainingConsoleLog[];
  loading: boolean;
  truncated: boolean;
  error: { message: string } | null;
  onRetry: () => void;
}) {
  const visibleLogs = logs.slice().reverse();
  return (
    <Panel>
      <PanelHeader
        title={
          <span className="inline-flex items-center gap-2">
            <Terminal className="h-4 w-4 text-ink-faint" aria-hidden="true" />
            Trainer console
          </span>
        }
        description="Newest reported line first"
        actions={
          <span className="font-mono text-2xs text-ink-faint">
            {formatCount(logs.length)} retained
          </span>
        }
      />
      {truncated && (
        <p className="border-b border-caution-100 bg-caution-50 px-5 py-2 text-2xs text-caution-700">
          Older output is unavailable because the bounded server or browser tail was trimmed.
        </p>
      )}
      {error && (
        <div
          role="alert"
          className="flex items-center justify-between gap-3 border-b border-critical-100 bg-critical-50 px-5 py-2 text-2xs text-critical-700"
        >
          <span>{error.message}</span>
          <button type="button" onClick={onRetry} className="font-medium underline underline-offset-2">
            Retry
          </button>
        </div>
      )}
      {loading && logs.length === 0 ? (
        <p className="px-5 py-16 text-center text-xs text-ink-muted" role="status">
          Loading trainer output…
        </p>
      ) : visibleLogs.length > 0 ? (
        <ol
          className="scroll-quiet max-h-[28rem] divide-y divide-line-subtle overflow-y-auto font-mono text-xs"
          aria-live="polite"
          aria-relevant="additions text"
        >
          {visibleLogs.map((log) => (
            <li
              key={log.sequence}
              className="grid gap-x-4 px-5 py-1.5 sm:grid-cols-[6rem_5rem_minmax(0,1fr)]"
            >
              <span className="tabular-nums text-ink-faint" title={formatDateTime(log.timestamp)}>
                {formatTime(log.timestamp)}
              </span>
              <span
                className={
                  log.source === "stderr"
                    ? "text-caution-700"
                    : log.source === "system"
                      ? "text-accent-700"
                      : "text-ink-faint"
                }
              >
                {log.source}
                {log.step === null ? "" : ` · ${formatCount(log.step)}`}
              </span>
              <span className="whitespace-pre-wrap break-words text-ink-secondary">{log.line}</span>
            </li>
          ))}
        </ol>
      ) : (
        <p className="px-5 py-16 text-center text-xs text-ink-muted">
          No trainer output is available yet.
        </p>
      )}
    </Panel>
  );
}

function Checkpoints({ run }: { run: TrainingRun }) {
  const outputUrl = hubRepoUrl(run.output_model_repo);
  const managedArtifactRevision =
    run.managed_job?.output_revision ?? run.managed_job?.output_marker_revision;
  const managedArtifactUrl =
    outputUrl && managedArtifactRevision
      ? appendHubPath(outputUrl, ["tree", managedArtifactRevision])
      : null;
  const checkpointUrl =
    outputUrl && run.checkpoint_revision
      ? appendHubPath(outputUrl, ["tree", run.checkpoint_revision])
      : null;

  return (
    <div className="space-y-8">
      <DescriptionList
        items={[
          {
            label: "Output model",
            value: run.managed_job ? (
              managedArtifactUrl ? (
                <a
                  href={managedArtifactUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex max-w-full items-center gap-1.5 font-mono text-accent-700 hover:text-accent-800"
                >
                  <span className="truncate" title={run.output_model_repo ?? undefined}>
                    {run.output_model_repo}
                  </span>
                  <span className="shrink-0 text-2xs text-ink-faint">
                    @{shortRevision(managedArtifactRevision, 12)}
                  </span>
                  <ExternalLink className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                </a>
              ) : (
                <span className="break-all font-mono">{run.output_model_repo}</span>
              )
            ) : <ArtifactLink repoId={run.output_model_repo} />,
            hint:
              run.managed_job && !managedArtifactUrl
                ? "Requested repo · existence not yet verified"
                : undefined,
          },
          {
            label: "Current revision",
            value: run.checkpoint_revision ? (
              checkpointUrl ? (
                <a
                  href={checkpointUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1.5 font-mono text-accent-700 hover:text-accent-800"
                >
                  {shortRevision(run.checkpoint_revision, 12)}
                  <ExternalLink className="h-3.5 w-3.5" aria-hidden="true" />
                </a>
              ) : (
                <span className="font-mono">{run.checkpoint_revision}</span>
              )
            ) : (
              <span className="text-ink-faint">Not registered</span>
            ),
          },
        ]}
      />

      <div>
        <SectionHeading
          level={3}
          title="Registered checkpoints"
          description="Immutable revisions the trainer reported for this run."
          className="mb-4"
        />
        {run.checkpoints.length ? (
          <Table label="Registered checkpoints" minWidth="34rem">
            <TableHead>
              <TableHeaderCell>Repository</TableHeaderCell>
              <TableHeaderCell>Revision</TableHeaderCell>
              <TableHeaderCell align="right">Step</TableHeaderCell>
              <TableHeaderCell align="right" />
            </TableHead>
            <TableBody>
              {run.checkpoints.map((checkpoint, index) => {
                const base = hubRepoUrl(checkpoint.repo_id);
                const url = base ? appendHubPath(base, ["tree", checkpoint.revision]) : null;
                return (
                  <TableRow key={`${checkpoint.repo_id}-${checkpoint.revision}-${index}`}>
                    <TableCell mono>{checkpoint.repo_id}</TableCell>
                    <TableCell mono muted>
                      {shortRevision(checkpoint.revision, 12)}
                    </TableCell>
                    <TableCell align="right">{formatCount(checkpoint.step)}</TableCell>
                    <TableCell align="right">
                      {url && (
                        <a
                          href={url}
                          target="_blank"
                          rel="noreferrer"
                          aria-label={`Open checkpoint at step ${checkpoint.step}`}
                          className="inline-flex text-accent-700 hover:text-accent-800"
                        >
                          <ExternalLink className="h-3.5 w-3.5" aria-hidden="true" />
                        </a>
                      )}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        ) : (
          <p className="text-xs text-ink-muted">No checkpoint revisions registered.</p>
        )}
      </div>
    </div>
  );
}

export function TrainingRunPage() {
  const { runId = "" } = useParams<{ runId: string }>();
  const {
    detail,
    detailLoading,
    detailError,
    liveError,
    lastLiveUpdate,
    consoleLogs,
    consoleLoading,
    consoleTruncated,
    consoleError,
    retryDetail,
    retryConsole,
  } = useTrainingRun(runId);
  const [tab, setTab] = useState<RunTab>("metrics");

  const metrics = useMemo(
    () =>
      detail
        ? Object.entries(detail.metrics).sort(([left], [right]) => left.localeCompare(right))
        : [],
    [detail],
  );

  if (detailLoading && !detail) {
    return (
      <Page>
        <PageHeader back={{ to: "/training", label: "Training" }} title="Loading run…" />
        <PageSection>
          <Skeleton className="h-24 w-full" />
          <Skeleton className="mt-5 h-72 w-full" />
        </PageSection>
      </Page>
    );
  }

  if (detailError || !detail) {
    return (
      <Page>
        <PageHeader back={{ to: "/training", label: "Training" }} title="Training run" />
        <PageSection>
          {detailError ? (
            <LoadErrorState
              error={detailError}
              resource="run"
              onRetry={() => void retryDetail()}
              busy={detailLoading}
            />
          ) : (
            <EmptyState
              icon={Activity}
              title="Training run unavailable"
              description="This run is no longer reported by the backend."
              action={
                <Link to="/training" className={buttonClass("primary", "md")}>
                  Back to training
                </Link>
              }
            />
          )}
        </PageSection>
      </Page>
    );
  }

  const configText = JSON.stringify(detail.config, null, 2);

  return (
    <Page>
      <PageHeader
        back={{ to: "/training", label: "Training" }}
        title={detail.name}
        meta={
          <>
            <Badge tone={trainingTone[detail.status]} dot pulse={isBusyStatus(detail.status)}>
              {detail.status}
            </Badge>
            <Mono title={detail.id}>{detail.id}</Mono>
            <span className="text-xs text-ink-faint">
              Auto-refresh every 2s
              {lastLiveUpdate ? ` · checked ${formatRelative(lastLiveUpdate)}` : ""}
            </span>
          </>
        }
      />

      {liveError && (
        <PageSection className="mt-8">
          <Alert tone="warning" role="status">
            Live metric refresh is temporarily unavailable; the last successful values remain
            visible.
          </Alert>
        </PageSection>
      )}

      <PageSection className="mt-8">
        <StatGrid columns={4}>
          <Stat label="Current step" value={formatCount(detail.current_step)} />
          <Stat
            label="Runtime"
            value={<span className="text-base">{detail.runtime || "Not reported"}</span>}
            hint={detail.framework || "Framework not reported"}
          />
          <Stat
            label="Dataset"
            value={
              <span className="block truncate font-mono text-sm" title={detail.dataset_repo ?? undefined}>
                {detail.dataset_repo ?? "—"}
              </span>
            }
            hint={detail.base_model ? `Base ${detail.base_model}` : "No base model reported"}
          />
          <Stat
            label="Updated"
            value={<span className="text-base">{formatRelative(detail.updated_at)}</span>}
            hint={`Created ${formatDateTime(detail.created_at)}`}
          />
        </StatGrid>
      </PageSection>

      {detail.managed_job && (
        <PageSection>
          <ManagedJobPanel job={detail.managed_job} />
        </PageSection>
      )}

      <PageSection>
        <Tabs
          label="Training run detail"
          value={tab}
          onChange={setTab}
          items={[
            { id: "metrics", label: "Metrics", count: metrics.length },
            { id: "console", label: "Console", count: consoleLogs.length },
            { id: "checkpoints", label: "Checkpoints", count: detail.checkpoints.length },
            { id: "config", label: "Configuration" },
          ]}
        />

        <div className="mt-6">
          <TabPanel id="metrics" active={tab === "metrics"}>
            {metrics.length ? (
              <div className={`grid gap-5 ${metrics.length > 1 ? "xl:grid-cols-2" : ""}`}>
                {metrics.map(([name, points]) => {
                  const latest = points.at(-1);
                  return (
                    <figure key={name} className="rounded-xl border border-line bg-surface p-5">
                      <figcaption className="flex items-start justify-between gap-3">
                        <div className="min-w-0">
                          <h3 className="truncate font-mono text-xs font-medium text-ink" title={name}>
                            {name}
                          </h3>
                          <p className="mt-1 text-2xs text-ink-muted">
                            {formatCount(points.length)} points
                          </p>
                        </div>
                        {latest && (
                          <div className="text-right">
                            <p className="font-mono text-lg font-semibold tracking-tight text-ink">
                              {formatScalar(latest.value)}
                            </p>
                            <p className="mt-0.5 text-2xs tabular-nums text-ink-muted">
                              step {formatCount(latest.step)}
                            </p>
                          </div>
                        )}
                      </figcaption>
                      <LineChart
                        className="mt-4"
                        label={name}
                        points={points}
                        height={220}
                        formatValue={formatScalar}
                        formatStep={formatCount}
                      />
                    </figure>
                  );
                })}
              </div>
            ) : (
              <EmptyState
                icon={Activity}
                title="No metrics reported"
                description="Curves appear as soon as the trainer client logs scalar values for this run."
              />
            )}
          </TabPanel>

          <TabPanel id="console" active={tab === "console"}>
            <ConsoleOutput
              logs={consoleLogs}
              loading={consoleLoading}
              truncated={consoleTruncated}
              error={consoleError}
              onRetry={() => void retryConsole()}
            />
          </TabPanel>

          <TabPanel id="checkpoints" active={tab === "checkpoints"}>
            <Checkpoints run={detail} />
          </TabPanel>

          <TabPanel id="config" active={tab === "config"}>
            {Object.keys(detail.config).length ? (
              <Panel>
                <pre className="scroll-quiet max-h-[32rem] overflow-auto whitespace-pre-wrap break-words px-5 py-5 font-mono text-xs leading-6 text-ink-secondary">
                  {configText}
                </pre>
              </Panel>
            ) : (
              <EmptyState title="No configuration reported" description="This run did not report a training configuration." />
            )}
          </TabPanel>
        </div>
      </PageSection>
    </Page>
  );
}
