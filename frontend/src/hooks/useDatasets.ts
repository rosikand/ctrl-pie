import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, fetchDatasets } from "../lib/api";
import type { DatasetSummary } from "../types/datasets";

export type DatasetLoadMode = "initial" | "refresh" | "more";

export type DatasetLoadError = {
  mode: DatasetLoadMode;
  message: string;
  status: number | null;
};

function mergeDatasets(
  current: DatasetSummary[],
  incoming: DatasetSummary[],
): DatasetSummary[] {
  const result = [...current];
  const indices = new Map(result.map((dataset, index) => [dataset.repo_id, index]));
  for (const dataset of incoming) {
    const index = indices.get(dataset.repo_id);
    if (index === undefined) {
      indices.set(dataset.repo_id, result.length);
      result.push(dataset);
    } else {
      result[index] = dataset;
    }
  }
  return result;
}

export function useDatasets() {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [namespace, setNamespace] = useState<string | null>(null);
  const [total, setTotal] = useState(0);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [fetchedAt, setFetchedAt] = useState<string | null>(null);
  const [initialLoading, setInitialLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<DatasetLoadError | null>(null);
  const requestSequence = useRef(0);
  const abortController = useRef<AbortController | null>(null);

  const load = useCallback(async (mode: DatasetLoadMode, cursor?: string | null) => {
    const sequence = requestSequence.current + 1;
    requestSequence.current = sequence;
    abortController.current?.abort();
    const controller = new AbortController();
    abortController.current = controller;
    setError(null);
    setInitialLoading(mode === "initial");
    setRefreshing(mode === "refresh");
    setLoadingMore(mode === "more");

    try {
      const response = await fetchDatasets({
        cursor: mode === "more" ? cursor : null,
        refresh: mode === "refresh",
        signal: controller.signal,
      });
      if (requestSequence.current !== sequence) return;
      setDatasets((current) =>
        mode === "more" ? mergeDatasets(current, response.datasets) : response.datasets,
      );
      setNamespace(response.namespace);
      setTotal(response.total);
      setNextCursor(response.next_cursor);
      setFetchedAt(response.fetched_at);
    } catch (reason) {
      if (controller.signal.aborted || requestSequence.current !== sequence) return;
      setError({
        mode,
        message: reason instanceof Error ? reason.message : "Could not load datasets.",
        status: reason instanceof ApiError ? reason.status : null,
      });
    } finally {
      if (requestSequence.current === sequence) {
        if (mode === "initial") setInitialLoading(false);
        if (mode === "refresh") setRefreshing(false);
        if (mode === "more") setLoadingMore(false);
      }
    }
  }, []);

  useEffect(() => {
    void load("initial");
    return () => abortController.current?.abort();
  }, [load]);

  const refresh = useCallback(() => load("refresh"), [load]);
  const loadMore = useCallback(() => {
    if (!nextCursor) return Promise.resolve();
    return load("more", nextCursor);
  }, [load, nextCursor]);
  const retry = useCallback(() => {
    if (error?.status === 422) return load("refresh");
    if (error?.mode === "more") return loadMore();
    return load(error?.mode === "refresh" ? "refresh" : "initial");
  }, [error?.mode, error?.status, load, loadMore]);

  return {
    datasets,
    namespace,
    total,
    nextCursor,
    fetchedAt,
    initialLoading,
    refreshing,
    loadingMore,
    error,
    busy: initialLoading || refreshing || loadingMore,
    refresh,
    loadMore,
    retry,
  };
}
