import { Database, FileQuestion, Lock, RefreshCw, ShieldAlert, Unlock } from "lucide-react";
import { Link } from "react-router-dom";

import { Page, PageHeader, PageSection } from "../components/layout/Page";
import { LoadErrorBar, LoadErrorState } from "../components/LoadError";
import { Alert } from "../components/ui/Alert";
import { Badge, Mono } from "../components/ui/Badge";
import { Button, buttonClass } from "../components/ui/Button";
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
import { useDatasets } from "../hooks/useDatasets";
import { formatCount, formatDateTime, formatRelative } from "../lib/format";

export function DatasetsPage() {
  const {
    datasets,
    namespace,
    total,
    nextCursor,
    fetchedAt,
    initialLoading,
    refreshing,
    loadingMore,
    error,
    busy,
    refresh,
    loadMore,
    retry,
  } = useDatasets();

  const hasDatasets = datasets.length > 0;

  return (
    <Page>
      <PageHeader
        title="Datasets"
        description="LeRobot datasets in the server-configured Hugging Face namespace. Open one to browse its immutable episodes."
        meta={
          namespace || hasDatasets ? (
            <>
              {namespace && <Mono>{namespace}</Mono>}
              <span className="text-xs text-ink-muted" aria-live="polite">
                {formatCount(datasets.length)} of {formatCount(total)}
              </span>
              {fetchedAt && (
                <span className="text-xs text-ink-faint" title={fetchedAt}>
                  Synced {formatRelative(fetchedAt)}
                </span>
              )}
            </>
          ) : undefined
        }
        actions={
          <Button
            variant="primary"
            icon={RefreshCw}
            loading={refreshing}
            disabled={busy}
            onClick={() => void refresh()}
          >
            {refreshing ? "Refreshing…" : "Refresh"}
          </Button>
        }
      />

      <PageSection>
        {initialLoading && !hasDatasets && <TableSkeleton rows={6} columns={6} label="Loading datasets" />}

        {!initialLoading && error && !hasDatasets && (
          <LoadErrorState error={error} resource="datasets" onRetry={() => void retry()} busy={busy} />
        )}

        {!initialLoading && !error && !hasDatasets && (
          <EmptyState
            icon={FileQuestion}
            title="No LeRobot datasets yet"
            description="The configured namespace has no datasets to show. Record and upload an episode to create the first one."
            action={
              <Link to="/record" className={buttonClass("primary", "md")}>
                Record demonstrations
              </Link>
            }
          />
        )}

        {hasDatasets && (
          <div className="space-y-5">
            {refreshing && (
              <Alert tone="info" role="status">
                Refreshing from Hugging Face. The current catalog stays available.
              </Alert>
            )}
            {error && (
              <LoadErrorBar
                error={error}
                resource="datasets"
                onRetry={() => void retry()}
                busy={busy}
                retryLabel={error.status === 422 ? "Refresh" : "Retry"}
              />
            )}

            <Table label="LeRobot datasets" minWidth="56rem" busy={refreshing || loadingMore}>
              <TableHead>
                <TableHeaderCell>Dataset</TableHeaderCell>
                <TableHeaderCell align="right">Episodes</TableHeaderCell>
                <TableHeaderCell align="right">Frames</TableHeaderCell>
                <TableHeaderCell align="right">FPS</TableHeaderCell>
                <TableHeaderCell>Robot</TableHeaderCell>
                <TableHeaderCell>Access</TableHeaderCell>
                <TableHeaderCell align="right">Updated</TableHeaderCell>
              </TableHead>
              <TableBody>
                {datasets.map((dataset) => {
                  const lerobot = dataset.lerobot;
                  const title = dataset.card?.title?.trim() || dataset.name;
                  return (
                    <TableRow key={dataset.repo_id} interactive>
                      <TableCell>
                        <RowLink to={`/datasets/${encodeURIComponent(dataset.name)}`}>
                          {title}
                        </RowLink>
                        <p
                          className="mt-0.5 truncate font-mono text-2xs text-ink-faint"
                          title={dataset.repo_id}
                        >
                          {dataset.repo_id}
                        </p>
                      </TableCell>
                      <TableCell align="right">
                        {lerobot ? formatCount(lerobot.total_episodes) : "—"}
                      </TableCell>
                      <TableCell align="right">
                        {lerobot ? formatCount(lerobot.total_frames) : "—"}
                      </TableCell>
                      <TableCell align="right">
                        {lerobot?.fps ? formatCount(lerobot.fps) : "—"}
                      </TableCell>
                      <TableCell>
                        {lerobot?.robot_type || (
                          <span className="text-ink-faint">Metadata unavailable</span>
                        )}
                      </TableCell>
                      <TableCell>
                        <span className="flex flex-wrap items-center gap-1.5">
                          <Badge tone={dataset.private ? "neutral" : "success"}>
                            {dataset.private ? (
                              <Lock className="h-3 w-3" aria-hidden="true" />
                            ) : (
                              <Unlock className="h-3 w-3" aria-hidden="true" />
                            )}
                            {dataset.private ? "Private" : "Public"}
                          </Badge>
                          {dataset.gated && (
                            <Badge tone="warning">
                              <ShieldAlert className="h-3 w-3" aria-hidden="true" />
                              Gated
                            </Badge>
                          )}
                        </span>
                      </TableCell>
                      <TableCell align="right" muted>
                        <span title={formatDateTime(dataset.last_modified)}>
                          {formatRelative(dataset.last_modified)}
                        </span>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>

            <div className="flex justify-center">
              {nextCursor ? (
                <Button
                  icon={Database}
                  loading={loadingMore}
                  disabled={busy}
                  onClick={() => void loadMore()}
                >
                  {loadingMore ? "Loading more…" : "Load more"}
                </Button>
              ) : (
                <p className="text-xs text-ink-faint">
                  All {formatCount(datasets.length)} datasets loaded
                </p>
              )}
            </div>
          </div>
        )}
      </PageSection>
    </Page>
  );
}
