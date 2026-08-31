import { BrainCircuit, RefreshCw } from "lucide-react";

import { Page, PageHeader, PageSection } from "../components/layout/Page";
import { LoadErrorBar, LoadErrorState } from "../components/LoadError";
import { Badge } from "../components/ui/Badge";
import { Button } from "../components/ui/Button";
import { EmptyState } from "../components/ui/EmptyState";
import { TableSkeleton } from "../components/ui/Skeleton";
import {
  RowLink,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
} from "../components/ui/Table";
import { useTrainingRunList } from "../hooks/useTrainingRuns";
import { formatCount, formatDateTime, formatRelative } from "../lib/format";
import { isBusyStatus, trainingTone } from "../lib/status";
import type { TrainingRun } from "../types/training";

function runSource(run: TrainingRun): string {
  if (!run.managed_job) return "External client";
  return run.managed_job.target_kind === "modal" ? "Managed · Modal" : "Managed · simulation";
}

export function TrainingPage() {
  const { runs, initialLoading, refreshing, error, refresh } = useTrainingRunList();
  const active = runs.filter((run) => run.status === "created" || run.status === "running").length;

  return (
    <Page>
      <PageHeader
        title="Training"
        description="Managed Modal jobs and externally reported runs, with bounded metrics, checkpoints, and sanitized console output."
        meta={
          runs.length > 0 ? (
            <>
              <span className="text-xs text-ink-muted">{formatCount(runs.length)} runs</span>
              {active > 0 && (
                <span className="text-xs text-accent-700">{formatCount(active)} in progress</span>
              )}
            </>
          ) : undefined
        }
        actions={
          <Button
            variant="primary"
            icon={RefreshCw}
            loading={refreshing}
            disabled={refreshing || initialLoading}
            onClick={() => void refresh()}
          >
            {refreshing ? "Refreshing…" : "Refresh"}
          </Button>
        }
      />

      <PageSection>
        {initialLoading && <TableSkeleton rows={5} columns={5} label="Loading training runs" />}

        {!initialLoading && error && runs.length === 0 && (
          <LoadErrorState error={error} resource="runs" onRetry={() => void refresh()} busy={refreshing} />
        )}

        {!initialLoading && !error && runs.length === 0 && (
          <EmptyState
            icon={BrainCircuit}
            title="No training runs yet"
            description="Launch a managed job from the Python SDK or report an external run through the Trainer API. Mock mode simulates managed compute without Modal or Hugging Face."
          />
        )}

        {!initialLoading && runs.length > 0 && (
          <div className="space-y-5">
            {error && (
              <LoadErrorBar error={error} resource="runs" onRetry={() => void refresh()} busy={refreshing} />
            )}
            <Table label="Training runs" minWidth="58rem" busy={refreshing}>
              <TableHead>
                <TableHeaderCell>Run</TableHeaderCell>
                <TableHeaderCell>Status</TableHeaderCell>
                <TableHeaderCell>Source</TableHeaderCell>
                <TableHeaderCell align="right">Step</TableHeaderCell>
                <TableHeaderCell>Dataset</TableHeaderCell>
                <TableHeaderCell>Output model</TableHeaderCell>
                <TableHeaderCell align="right">Updated</TableHeaderCell>
              </TableHead>
              <TableBody>
                {runs.map((run) => (
                  <TableRow key={run.id} interactive>
                    <TableCell>
                      <RowLink to={`/training/${encodeURIComponent(run.id)}`}>{run.name}</RowLink>
                      <p className="mt-0.5 truncate font-mono text-2xs text-ink-faint" title={run.id}>
                        {run.id}
                      </p>
                    </TableCell>
                    <TableCell>
                      <Badge tone={trainingTone[run.status]} dot pulse={isBusyStatus(run.status)}>
                        {run.status}
                      </Badge>
                    </TableCell>
                    <TableCell>{runSource(run)}</TableCell>
                    <TableCell align="right">{formatCount(run.current_step)}</TableCell>
                    <TableCell mono muted className="max-w-[16rem] truncate">
                      {run.dataset_repo ?? "—"}
                    </TableCell>
                    <TableCell mono muted className="max-w-[16rem] truncate">
                      {run.output_model_repo ?? "—"}
                    </TableCell>
                    <TableCell align="right" muted>
                      <span title={formatDateTime(run.updated_at)}>{formatRelative(run.updated_at)}</span>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </PageSection>
    </Page>
  );
}
