import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, fetchTrainerModels } from "../lib/api";
import type { TrainerModelsResponse } from "../types/training";
import type { TrainingLoadError } from "./useTrainingRuns";

export function useTrainerModels() {
  const [data, setData] = useState<TrainerModelsResponse | null>(null);
  const [initialLoading, setInitialLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<TrainingLoadError | null>(null);
  const requestSequence = useRef(0);
  const controllerRef = useRef<AbortController | null>(null);
  const dataRef = useRef<TrainerModelsResponse | null>(null);

  const load = useCallback(async (refresh = false) => {
    const sequence = requestSequence.current + 1;
    requestSequence.current = sequence;
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    const hasData = dataRef.current !== null;
    setError(null);
    setInitialLoading(!refresh && !hasData);
    setRefreshing(refresh || hasData);

    try {
      const response = await fetchTrainerModels(refresh, controller.signal);
      if (controller.signal.aborted || requestSequence.current !== sequence) return;
      dataRef.current = response;
      setData(response);
    } catch (reason) {
      if (controller.signal.aborted || requestSequence.current !== sequence) return;
      setError({
        message: reason instanceof Error ? reason.message : "Could not load models.",
        status: reason instanceof ApiError ? reason.status : null,
      });
    } finally {
      if (requestSequence.current === sequence) {
        setInitialLoading(false);
        setRefreshing(false);
      }
    }
  }, []);

  useEffect(() => {
    void load();
    return () => {
      requestSequence.current += 1;
      controllerRef.current?.abort();
    };
  }, [load]);

  const refresh = useCallback(() => load(true), [load]);
  const retry = useCallback(() => load(Boolean(data)), [data, load]);

  return {
    data,
    initialLoading,
    refreshing,
    error,
    refresh,
    retry,
  };
}
