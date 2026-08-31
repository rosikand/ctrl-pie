import { Plus, RadioTower, RefreshCw } from "lucide-react";
import { Link } from "react-router-dom";

import { InferenceReadinessNotice } from "../components/InferenceReadiness";
import { Page, PageHeader, PageSection } from "../components/layout/Page";
import { LoadErrorBar, LoadErrorState } from "../components/LoadError";
import { Badge } from "../components/ui/Badge";
import { buttonClass, IconButton } from "../components/ui/Button";
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
import { useDeploymentList } from "../hooks/useInference";
import { formatDateTime, formatRelative, shortRevision } from "../lib/format";
import { deploymentTone, isBusyStatus } from "../lib/status";

const RUNTIME_LABELS: Record<string, string> = {
  lerobot: "LeRobot",
  openpi: "OpenPI",
  stub: "Stub",
};

export function InferencePage() {
  const { deployments, initialLoading, refreshing, error, refresh } = useDeploymentList(10_000);
  const active = deployments.filter(
    (deployment) => !["stopped", "failed"].includes(deployment.status),
  ).length;

  return (
    <Page>
      <PageHeader
        title="Inference"
        description="Deploy one pinned policy revision, then explicitly start and stop robot execution against it."
        meta={
          deployments.length > 0 ? (
            <span className="text-xs text-ink-muted">
              {deployments.length} deployment{deployments.length === 1 ? "" : "s"}
              {active > 0 ? ` · ${active} active` : ""}
            </span>
          ) : undefined
        }
        actions={
          <>
            <IconButton
              icon={RefreshCw}
              label="Refresh deployments"
              spinning={refreshing}
              disabled={refreshing || initialLoading}
              onClick={() => void refresh()}
            />
            <Link to="/inference/new" className={buttonClass("primary", "md")}>
              <Plus className="h-4 w-4" aria-hidden="true" />
              New deployment
            </Link>
          </>
        }
      />

      <PageSection className="mt-8">
        <InferenceReadinessNotice />
      </PageSection>

      <PageSection>
        {initialLoading && <TableSkeleton rows={4} columns={5} label="Loading deployments" />}

        {!initialLoading && error && deployments.length === 0 && (
          <LoadErrorState
            error={error}
            resource="deployments"
            onRetry={() => void refresh()}
            busy={refreshing}
          />
        )}

        {!initialLoading && !error && deployments.length === 0 && (
          <EmptyState
            icon={RadioTower}
            title="No deployments yet"
            description="A deployment verifies the exact runtime, model, and revision identity of an endpoint. It never moves the robot on its own."
            action={
              <Link to="/inference/new" className={buttonClass("primary", "md")}>
                Deploy a policy
              </Link>
            }
          />
        )}

        {!initialLoading && deployments.length > 0 && (
          <div className="space-y-5">
            {error && (
              <LoadErrorBar
                error={error}
                resource="deployments"
                onRetry={() => void refresh()}
                busy={refreshing}
              />
            )}
            <Table label="Inference deployments" minWidth="56rem" busy={refreshing}>
              <TableHead>
                <TableHeaderCell>Deployment</TableHeaderCell>
                <TableHeaderCell>Status</TableHeaderCell>
                <TableHeaderCell>Policy</TableHeaderCell>
                <TableHeaderCell>Runtime</TableHeaderCell>
                <TableHeaderCell>Compute</TableHeaderCell>
                <TableHeaderCell>Robot</TableHeaderCell>
                <TableHeaderCell align="right">Updated</TableHeaderCell>
              </TableHead>
              <TableBody>
                {deployments.map((deployment) => (
                  <TableRow key={deployment.id} interactive>
                    <TableCell>
                      <RowLink to={`/inference/${encodeURIComponent(deployment.id)}`}>
                        {deployment.name}
                      </RowLink>
                    </TableCell>
                    <TableCell>
                      <Badge
                        tone={deploymentTone[deployment.status]}
                        dot
                        pulse={isBusyStatus(deployment.status)}
                      >
                        {deployment.status}
                      </Badge>
                    </TableCell>
                    <TableCell mono muted className="max-w-[18rem]">
                      <span className="block truncate" title={deployment.model_repo}>
                        {deployment.model_repo}
                      </span>
                      <span className="mt-0.5 block text-2xs text-ink-faint">
                        @{shortRevision(deployment.checkpoint_revision, 10) || "default branch"}
                      </span>
                    </TableCell>
                    <TableCell>
                      {RUNTIME_LABELS[deployment.runtime] ?? deployment.runtime}
                    </TableCell>
                    <TableCell>{deployment.compute_size}</TableCell>
                    <TableCell mono muted>
                      {deployment.arm_id ?? "—"}
                    </TableCell>
                    <TableCell align="right" muted>
                      <span title={formatDateTime(deployment.updated_at)}>
                        {formatRelative(deployment.updated_at)}
                      </span>
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
