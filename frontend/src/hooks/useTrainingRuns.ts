import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, fetchTrainingRun, fetchTrainingRuns } from "../lib/api";
import type { TrainingRun } from "../types/training";

export type TrainingLoadError = {
  message: string;
  status: number | null;
};

function loadError(reason: unknown, fallback: string): TrainingLoadError {
  return {
    message: reason instanceof Error ? reason.message : fallback,
    status: reason instanceof ApiError ? reason.status : null,
  };
}

export function useTrainingRuns() {
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [detail, setDetail] = useState<TrainingRun | null>(null);
  const [initialLoading, setInitialLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [listError, setListError] = useState<TrainingLoadError | null>(null);
  const [detailError, setDetailError] = useState<TrainingLoadError | null>(null);
  const listSequence = useRef(0);
  const detailSequence = useRef(0);
  const listController = useRef<AbortController | null>(null);
  const detailController = useRef<AbortController | null>(null);

  const loadRuns = useCallback(async (preserve = false) => {
    const sequence = listSequence.current + 1;
    listSequence.current = sequence;
    listController.current?.abort();
    const controller = new AbortController();
    listController.current = controller;
    setListError(null);
    setInitialLoading(!preserve);
    setRefreshing(preserve);
    if (!preserve) {
      detailSequence.current += 1;
      detailController.current?.abort();
      setRuns([]);
      setSelectedRunId(null);
      setDetail(null);
      setDetailError(null);
    }

    try {
      const response = await fetchTrainingRuns(controller.signal);
      if (controller.signal.aborted || listSequence.current !== sequence) return;
      setRuns(response.runs);
      setSelectedRunId((current) =>
        response.runs.some((run) => run.id === current)
          ? current
          : response.runs[0]?.id ?? null,
      );
      if (response.runs.length === 0) {
        detailSequence.current += 1;
        detailController.current?.abort();
        setDetail(null);
        setDetailError(null);
      }
    } catch (reason) {
      if (controller.signal.aborted || listSequence.current !== sequence) return;
      setListError(loadError(reason, "Could not load training runs."));
    } finally {
      if (listSequence.current === sequence) {
        setInitialLoading(false);
        setRefreshing(false);
      }
    }
  }, []);

  const loadDetail = useCallback(async (runId: string) => {
    const sequence = detailSequence.current + 1;
    detailSequence.current = sequence;
    detailController.current?.abort();
    const controller = new AbortController();
    detailController.current = controller;
    setDetailLoading(true);
    setDetailError(null);
    setDetail(null);

    try {
      const response = await fetchTrainingRun(runId, controller.signal);
      if (controller.signal.aborted || detailSequence.current !== sequence) return;
      if (response.id !== runId) {
        setDetailError({ status: 409, message: "The selected run changed while loading." });
        return;
      }
      setDetail(response);
    } catch (reason) {
      if (controller.signal.aborted || detailSequence.current !== sequence) return;
      setDetailError(loadError(reason, "Could not load this training run."));
    } finally {
      if (detailSequence.current === sequence) setDetailLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadRuns();
    return () => {
      listSequence.current += 1;
      detailSequence.current += 1;
      listController.current?.abort();
      detailController.current?.abort();
    };
  }, [loadRuns]);

  useEffect(() => {
    if (!selectedRunId || !runs.some((run) => run.id === selectedRunId)) {
      setDetail(null);
      setDetailLoading(false);
      return;
    }
    void loadDetail(selectedRunId);
    return () => {
      detailSequence.current += 1;
      detailController.current?.abort();
    };
  }, [loadDetail, runs, selectedRunId]);

  const selectRun = useCallback((runId: string) => {
    if (runId === selectedRunId || !runs.some((run) => run.id === runId)) return;
    detailSequence.current += 1;
    detailController.current?.abort();
    setDetail(null);
    setDetailError(null);
    setDetailLoading(true);
    setSelectedRunId(runId);
  }, [runs, selectedRunId]);

  const refresh = useCallback(() => loadRuns(true), [loadRuns]);
  const retryDetail = useCallback(() => {
    if (!selectedRunId) return Promise.resolve();
    return loadDetail(selectedRunId);
  }, [loadDetail, selectedRunId]);

  return {
    runs,
    selectedRunId,
    detail,
    initialLoading,
    refreshing,
    detailLoading,
    listError,
    detailError,
    selectRun,
    refresh,
    retryList: loadRuns,
    retryDetail,
  };
}
