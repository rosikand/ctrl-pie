import {
  AlertCircle,
  ExternalLink,
  GitBranch,
  Lock,
  Package,
  ShieldAlert,
  Tag,
  Unlock,
} from "lucide-react";

import {
  appendHubPath,
  formatTrainerTimestamp as formatTimestamp,
  trainerCountFormatter as countFormatter,
  TrainerDetailValue as DetailValue,
  TrainerErrorPanel as ErrorPanel,
  TrainerInlineError as InlineError,
  TrainerViewHeader as ViewHeader,
} from "../components/TrainerView";
import { useTrainerModels } from "../hooks/useTrainerModels";
import type { TrainerModelSummary } from "../types/training";

function safeHubUrl(value: string): string | null {
  try {
    const url = new URL(value);
    return url.protocol === "https:" && url.hostname === "huggingface.co"
      ? url.toString()
      : null;
  } catch {
    return null;
  }
}

function ModelsLoading() {
  return (
    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3" aria-busy="true">
      <p className="sr-only" role="status">Loading models</p>
      {Array.from({ length: 6 }).map((_, index) => (
        <div key={index} className="h-[29rem] animate-pulse rounded-xl border border-slate-200 bg-white p-5 shadow-panel">
          <div className="h-3 w-40 rounded bg-slate-100" />
          <div className="mt-3 h-5 w-2/3 rounded bg-slate-100" />
          <div className="mt-5 h-12 rounded bg-slate-50" />
          <div className="mt-6 h-20 rounded bg-slate-50" />
          <div className="mt-6 h-24 rounded bg-slate-50" />
        </div>
      ))}
    </div>
  );
}

function ModelCard({ model }: { model: TrainerModelSummary }) {
  const hubUrl = safeHubUrl(model.hub_url);
  const revisionUrl = hubUrl && model.revision
    ? appendHubPath(hubUrl, ["tree", model.revision])
    : null;
  const visibleTags = model.tags.slice(0, 4);

  return (
    <article className="flex min-h-full flex-col rounded-xl border border-slate-200 bg-white shadow-panel">
      <div className="border-b border-slate-100 px-5 py-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="truncate font-mono text-[11px] text-slate-400" title={model.repo_id}>{model.repo_id}</p>
            <h3 className="mt-1.5 text-base font-semibold text-slate-950">{model.name}</h3>
          </div>
          <div className="flex shrink-0 flex-wrap justify-end gap-1.5">
            <span className={`inline-flex items-center gap-1 rounded-full px-2 py-1 text-[10px] font-semibold ${model.private ? "bg-slate-100 text-slate-700" : "bg-emerald-50 text-emerald-700"}`}>
              {model.private ? <Lock className="h-3 w-3" aria-hidden="true" /> : <Unlock className="h-3 w-3" aria-hidden="true" />}
              {model.private ? "Private" : "Public"}
            </span>
            {model.gated && <span className="inline-flex items-center gap-1 rounded-full bg-amber-50 px-2 py-1 text-[10px] font-semibold text-amber-700"><ShieldAlert className="h-3 w-3" aria-hidden="true" />Gated</span>}
          </div>
        </div>
        {model.card
          ? <p className="mt-3 min-h-10 line-clamp-3 text-sm leading-5 text-slate-500">{model.card.description?.trim() || "No description was provided in the model card."}</p>
          : <div className="mt-3 flex items-start gap-2 rounded-lg border border-amber-100 bg-amber-50 px-3 py-2 text-xs text-amber-800"><AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />Model card metadata is unavailable.</div>}
      </div>
      <div className="flex flex-1 flex-col px-5 py-4">
        <dl className="grid grid-cols-2 gap-2">
          <DetailValue label="Pipeline">{model.pipeline_tag || "—"}</DetailValue>
          <DetailValue label="Library">{model.library_name || "—"}</DetailValue>
        </dl>
        {model.card && (
          <div className="mt-4 space-y-3 text-xs">
            <div>
              <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-400">Base model</p>
              <div className="mt-1.5 flex flex-wrap gap-1.5">
                {model.card.base_model.length
                  ? model.card.base_model.slice(0, 3).map((repo, index) => <span key={`${repo}-${index}`} className="max-w-full truncate rounded bg-blue-50 px-2 py-1 font-mono text-[10px] text-blue-700" title={repo}>{repo}</span>)
                  : <span className="text-slate-400">Not specified</span>}
              </div>
            </div>
            <div>
              <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-400">Datasets</p>
              <div className="mt-1.5 flex flex-wrap gap-1.5">
                {model.card.datasets.length
                  ? model.card.datasets.slice(0, 3).map((repo, index) => <span key={`${repo}-${index}`} className="max-w-full truncate rounded bg-violet-50 px-2 py-1 font-mono text-[10px] text-violet-700" title={repo}>{repo}</span>)
                  : <span className="text-slate-400">Not specified</span>}
              </div>
            </div>
          </div>
        )}
        <div className="mt-4 rounded-lg bg-slate-50 px-3 py-3">
          <div className="flex items-center justify-between gap-2">
            <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-400">Checkpoint files</p>
            <span className="text-[10px] text-slate-400">{countFormatter.format(model.checkpoints.length)}</span>
          </div>
          {model.checkpoints.length ? (
            <ul className="mt-2 space-y-1.5">
              {model.checkpoints.slice(0, 5).map((checkpoint) => {
                const url = hubUrl && model.revision
                  ? appendHubPath(hubUrl, ["blob", model.revision, checkpoint])
                  : null;
                return (
                  <li key={checkpoint} className="flex min-w-0 items-center gap-1.5 font-mono text-[10px] text-slate-600">
                    <GitBranch className="h-3 w-3 shrink-0 text-slate-300" aria-hidden="true" />
                    {url
                      ? <a href={url} target="_blank" rel="noreferrer" className="truncate hover:text-brand-600" title={checkpoint}>{checkpoint}</a>
                      : <span className="truncate" title={checkpoint}>{checkpoint}</span>}
                  </li>
                );
              })}
              {model.checkpoints.length > 5 && <li className="text-[10px] text-slate-400">+{model.checkpoints.length - 5} more files</li>}
            </ul>
          ) : <p className="mt-2 text-[11px] text-slate-400">No checkpoint files discovered.</p>}
        </div>
        {visibleTags.length > 0 && (
          <div className="mt-3 flex items-start gap-2">
            <Tag className="mt-1 h-3.5 w-3.5 shrink-0 text-slate-300" aria-hidden="true" />
            <div className="flex flex-wrap gap-1">
              {visibleTags.map((tag, index) => <span key={`${tag}-${index}`} className="max-w-[11rem] truncate rounded px-1.5 py-1 text-[10px] text-slate-500 ring-1 ring-slate-100" title={tag}>{tag}</span>)}
              {model.tags.length > visibleTags.length && <span className="px-1.5 py-1 text-[10px] text-slate-400">+{model.tags.length - visibleTags.length}</span>}
            </div>
          </div>
        )}
        <div className="mt-auto pt-4">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-slate-100 pt-3 text-[10px] text-slate-400">
            <span>Updated {formatTimestamp(model.last_modified)}</span>
            {model.revision
              ? revisionUrl
                ? <a href={revisionUrl} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 font-mono text-brand-600 hover:text-brand-700" title={model.revision}>@{model.revision.slice(0, 8)}<ExternalLink className="h-3 w-3" aria-hidden="true" /></a>
                : <span className="font-mono">@{model.revision.slice(0, 8)}</span>
              : <span className="text-amber-600">Revision unavailable</span>}
          </div>
          {hubUrl && <a href={hubUrl} target="_blank" rel="noreferrer" className="mt-3 inline-flex items-center gap-1.5 text-xs font-semibold text-brand-600 hover:text-brand-700">Open model on Hugging Face<ExternalLink className="h-3.5 w-3.5" aria-hidden="true" /></a>}
        </div>
      </div>
    </article>
  );
}

function ModelsCatalog() {
  const { data, initialLoading, refreshing, error, refresh, retry } = useTrainerModels();
  const models = data?.models ?? [];

  return (
    <div>
      <ViewHeader title="Namespace models" description={data ? `Hugging Face models discovered under ${data.namespace}.` : "Model artifacts discovered through the backend's configured Hugging Face namespace."} count={data?.total ?? (initialLoading ? null : models.length)} refreshing={refreshing || initialLoading} onRefresh={() => void refresh()} />
      {data && <div className="mb-5 flex flex-wrap items-center gap-2 text-xs text-slate-400"><span className="rounded-md bg-white px-2 py-1 font-mono ring-1 ring-slate-200">{data.namespace}</span><span title={data.fetched_at}>Synced {formatTimestamp(data.fetched_at)}</span></div>}
      {initialLoading && <ModelsLoading />}
      {!initialLoading && error && !data && <ErrorPanel error={error} context="models" onRetry={() => void retry()} loading={refreshing || initialLoading} />}
      {!initialLoading && data && error && <InlineError error={error} onRetry={() => void retry()} loading={refreshing} />}
      {!initialLoading && data && !error && models.length === 0 && <section className="grid min-h-[20rem] place-items-center rounded-xl border border-slate-200 bg-white px-6 text-center shadow-panel"><div className="max-w-md"><Package className="mx-auto h-7 w-7 text-slate-300" aria-hidden="true" /><h2 className="mt-3 text-sm font-semibold text-slate-900">No model repositories found</h2><p className="mt-2 text-sm leading-6 text-slate-500">The configured namespace has no models to show yet.</p></div></section>}
      {!initialLoading && data && models.length > 0 && <div className="grid items-stretch gap-4 md:grid-cols-2 xl:grid-cols-3" aria-busy={refreshing}>{models.map((model) => <ModelCard key={model.repo_id} model={model} />)}</div>}
    </div>
  );
}

export function ModelsPage() {
  return (
    <div className="mx-auto max-w-7xl px-5 py-8 sm:px-8 lg:px-10 lg:py-10">
      <header>
        <p className="text-xs font-semibold uppercase tracking-[0.16em] text-brand-600">Hugging Face artifacts</p>
        <h1 className="mt-2 text-2xl font-semibold tracking-tight text-slate-950 sm:text-3xl">Models</h1>
        <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">Browse model repositories, immutable revisions, checkpoint files, and card metadata in the configured namespace.</p>
      </header>
      <section className="mt-7">
        <ModelsCatalog />
      </section>
    </div>
  );
}
