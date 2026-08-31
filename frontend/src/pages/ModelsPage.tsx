import { ExternalLink, GitBranch, Lock, Package, RefreshCw, ShieldAlert, Unlock } from "lucide-react";
import { useState } from "react";

import { Page, PageHeader, PageSection } from "../components/layout/Page";
import { LoadErrorBar, LoadErrorState } from "../components/LoadError";
import { Alert } from "../components/ui/Alert";
import { Badge, Mono } from "../components/ui/Badge";
import { Button } from "../components/ui/Button";
import { DescriptionList } from "../components/ui/DescriptionList";
import { Drawer } from "../components/ui/Drawer";
import { EmptyState } from "../components/ui/EmptyState";
import { TableSkeleton } from "../components/ui/Skeleton";
import {
  RowButton,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
} from "../components/ui/Table";
import { useTrainerModels } from "../hooks/useTrainerModels";
import { appendHubPath, formatCount, formatDateTime, formatRelative, safeHubUrl, shortRevision } from "../lib/format";
import type { TrainerModelSummary } from "../types/training";

/** Read-only detail for one Hugging Face model repository. */
function ModelDetail({ model }: { model: TrainerModelSummary }) {
  const hubUrl = safeHubUrl(model.hub_url);
  const revisionUrl = hubUrl && model.revision ? appendHubPath(hubUrl, ["tree", model.revision]) : null;

  return (
    <div className="space-y-7">
      {model.card ? (
        <p className="text-[13px] leading-6 text-ink-secondary">
          {model.card.description?.trim() || "No description was provided in the model card."}
        </p>
      ) : (
        <Alert tone="warning">Model card metadata is unavailable for this repository.</Alert>
      )}

      <DescriptionList
        items={[
          { label: "Repository", value: model.repo_id, mono: true, span: true },
          { label: "Pipeline", value: model.pipeline_tag || "—" },
          { label: "Library", value: model.library_name || "—" },
          {
            label: "Revision",
            value: model.revision ? (
              revisionUrl ? (
                <a
                  href={revisionUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1 text-accent-700 hover:text-accent-800"
                >
                  {shortRevision(model.revision, 12)}
                  <ExternalLink className="h-3.5 w-3.5" aria-hidden="true" />
                </a>
              ) : (
                shortRevision(model.revision, 12)
              )
            ) : (
              <span className="text-caution-700">Revision unavailable</span>
            ),
            mono: true,
          },
          { label: "Updated", value: formatDateTime(model.last_modified) },
        ]}
      />

      {model.card && (
        <div className="grid gap-6 sm:grid-cols-2">
          <div>
            <p className="text-2xs font-medium uppercase tracking-[0.08em] text-ink-faint">
              Base model
            </p>
            <div className="mt-2 flex flex-wrap gap-1.5">
              {model.card.base_model.length ? (
                model.card.base_model.map((repo, index) => (
                  <Mono key={`${repo}-${index}`} title={repo}>
                    {repo}
                  </Mono>
                ))
              ) : (
                <span className="text-xs text-ink-faint">Not specified</span>
              )}
            </div>
          </div>
          <div>
            <p className="text-2xs font-medium uppercase tracking-[0.08em] text-ink-faint">
              Training datasets
            </p>
            <div className="mt-2 flex flex-wrap gap-1.5">
              {model.card.datasets.length ? (
                model.card.datasets.map((repo, index) => (
                  <Mono key={`${repo}-${index}`} title={repo}>
                    {repo}
                  </Mono>
                ))
              ) : (
                <span className="text-xs text-ink-faint">Not specified</span>
              )}
            </div>
          </div>
        </div>
      )}

      <div>
        <p className="text-2xs font-medium uppercase tracking-[0.08em] text-ink-faint">
          Checkpoint files
          <span className="ml-2 font-normal normal-case tracking-normal text-ink-muted">
            {formatCount(model.checkpoints.length)}
          </span>
        </p>
        {model.checkpoints.length ? (
          <ul className="mt-2 space-y-1.5">
            {model.checkpoints.map((checkpoint) => {
              const url =
                hubUrl && model.revision
                  ? appendHubPath(hubUrl, ["blob", model.revision, checkpoint])
                  : null;
              return (
                <li key={checkpoint} className="flex min-w-0 items-center gap-2 font-mono text-2xs">
                  <GitBranch className="h-3 w-3 shrink-0 text-ink-faint" aria-hidden="true" />
                  {url ? (
                    <a
                      href={url}
                      target="_blank"
                      rel="noreferrer"
                      className="truncate text-ink-secondary hover:text-accent-700"
                      title={checkpoint}
                    >
                      {checkpoint}
                    </a>
                  ) : (
                    <span className="truncate text-ink-secondary" title={checkpoint}>
                      {checkpoint}
                    </span>
                  )}
                </li>
              );
            })}
          </ul>
        ) : (
          <p className="mt-2 text-xs text-ink-faint">No checkpoint files discovered.</p>
        )}
        <p className="mt-3 text-2xs leading-5 text-ink-muted">
          Checkpoint paths are read-only metadata. Deployments always pin one immutable Git
          revision, never an artifact path.
        </p>
      </div>

      {model.tags.length > 0 && (
        <div>
          <p className="text-2xs font-medium uppercase tracking-[0.08em] text-ink-faint">Tags</p>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {model.tags.map((tag, index) => (
              <span
                key={`${tag}-${index}`}
                className="max-w-[12rem] truncate rounded-md px-1.5 py-0.5 text-2xs text-ink-muted ring-1 ring-line"
                title={tag}
              >
                {tag}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export function ModelsPage() {
  const { data, initialLoading, refreshing, error, refresh, retry } = useTrainerModels();
  const [selectedRepoId, setSelectedRepoId] = useState<string | null>(null);
  const models = data?.models ?? [];
  const selected = models.find((model) => model.repo_id === selectedRepoId) ?? null;

  return (
    <Page>
      <PageHeader
        title="Models"
        description="Model repositories discovered in the configured Hugging Face namespace, with their immutable revisions and checkpoint metadata."
        meta={
          data ? (
            <>
              <Mono>{data.namespace}</Mono>
              <span className="text-xs text-ink-muted">{formatCount(data.total)} models</span>
              <span className="text-xs text-ink-faint" title={data.fetched_at}>
                Synced {formatRelative(data.fetched_at)}
              </span>
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
        {initialLoading && <TableSkeleton rows={5} columns={6} label="Loading models" />}

        {!initialLoading && error && !data && (
          <LoadErrorState
            error={error}
            resource="models"
            onRetry={() => void retry()}
            busy={refreshing}
          />
        )}

        {!initialLoading && data && (
          <div className="space-y-5">
            {error && (
              <LoadErrorBar error={error} resource="models" onRetry={() => void retry()} busy={refreshing} />
            )}

            {models.length === 0 ? (
              <EmptyState
                icon={Package}
                title="No model repositories found"
                description="The configured namespace has no models yet. Managed or external training runs publish them here."
              />
            ) : (
              <Table label="Namespace models" minWidth="56rem" busy={refreshing}>
                <TableHead>
                  <TableHeaderCell>Model</TableHeaderCell>
                  <TableHeaderCell>Pipeline</TableHeaderCell>
                  <TableHeaderCell>Library</TableHeaderCell>
                  <TableHeaderCell align="right">Checkpoints</TableHeaderCell>
                  <TableHeaderCell>Access</TableHeaderCell>
                  <TableHeaderCell>Revision</TableHeaderCell>
                  <TableHeaderCell align="right">Updated</TableHeaderCell>
                </TableHead>
                <TableBody>
                  {models.map((model) => (
                    <TableRow
                      key={model.repo_id}
                      interactive
                      selected={model.repo_id === selectedRepoId}
                    >
                      <TableCell>
                        <RowButton onClick={() => setSelectedRepoId(model.repo_id)}>
                          {model.name}
                        </RowButton>
                        <p
                          className="mt-0.5 truncate font-mono text-2xs text-ink-faint"
                          title={model.repo_id}
                        >
                          {model.repo_id}
                        </p>
                      </TableCell>
                      <TableCell>{model.pipeline_tag || "—"}</TableCell>
                      <TableCell>{model.library_name || "—"}</TableCell>
                      <TableCell align="right">{formatCount(model.checkpoints.length)}</TableCell>
                      <TableCell>
                        <span className="flex flex-wrap items-center gap-1.5">
                          <Badge tone={model.private ? "neutral" : "success"}>
                            {model.private ? (
                              <Lock className="h-3 w-3" aria-hidden="true" />
                            ) : (
                              <Unlock className="h-3 w-3" aria-hidden="true" />
                            )}
                            {model.private ? "Private" : "Public"}
                          </Badge>
                          {model.gated && (
                            <Badge tone="warning">
                              <ShieldAlert className="h-3 w-3" aria-hidden="true" />
                              Gated
                            </Badge>
                          )}
                        </span>
                      </TableCell>
                      <TableCell mono muted>
                        {model.revision ? (
                          shortRevision(model.revision)
                        ) : (
                          <span className="text-caution-700">unavailable</span>
                        )}
                      </TableCell>
                      <TableCell align="right" muted>
                        <span title={formatDateTime(model.last_modified)}>
                          {formatRelative(model.last_modified)}
                        </span>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </div>
        )}
      </PageSection>

      <Drawer
        open={selected !== null}
        onClose={() => setSelectedRepoId(null)}
        title={selected?.name ?? "Model"}
        description={selected?.repo_id}
        width="max-w-xl"
        footer={
          selected && safeHubUrl(selected.hub_url) ? (
            <a
              href={safeHubUrl(selected.hub_url) ?? "#"}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1.5 text-[13px] font-medium text-accent-700 hover:text-accent-800"
            >
              Open on Hugging Face
              <ExternalLink className="h-3.5 w-3.5" aria-hidden="true" />
            </a>
          ) : (
            <span className="text-xs text-ink-faint">Hub link unavailable</span>
          )
        }
      >
        {selected && <ModelDetail model={selected} />}
      </Drawer>
    </Page>
  );
}
