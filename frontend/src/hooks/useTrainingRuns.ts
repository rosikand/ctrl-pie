import { useCallback, useEffect, useRef, useState } from "react";

import {
  ApiError,
  fetchTrainingConsoleLogs,
  fetchTrainingRun,
  fetchTrainingRuns,
} from "../lib/api";
import type {
  TrainingConsoleLog,
  TrainingLoadError,
  TrainingRun,
} from "../types/training";

const LIVE_POLL_INTERVAL_MS = 2_000;

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
  const [consoleLogs, setConsoleLogs] = useState<TrainingConsoleLog[]>([]);
  const [consoleLoading, setConsoleLoading] = useState(false);
  const [consoleTruncated, setConsoleTruncated] = useState(false);
  const [lastLiveUpdate, setLastLiveUpdate] = useState<string | null>(null);
  const [listError, setListError] = useState<TrainingLoadError | null>(null);
  const [detailError, setDetailError] = useState<TrainingLoadError | null>(null);
  const [consoleError, setConsoleError] = useState<TrainingLoadError | null>(null);
  const [liveError, setLiveError] = useState<TrainingLoadError | null>(null);
  const listSequence = useRef(0);
  const detailSequence = useRef(0);
  const consoleSequence = useRef(0);
  const consoleCursor = useRef<number | undefined>(undefined);
  const listController = useRef<AbortController | null>(null);
  const detailController = useRef<AbortController | null>(null);
  const consoleController = useRef<AbortController | null>(null);

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
      consoleSequence.current += 1;
      detailController.current?.abort();
      consoleController.current?.abort();
      setRuns([]);
      setSelectedRunId(null);
      setDetail(null);
      setConsoleLogs([]);
      setConsoleTruncated(false);
      setLastLiveUpdate(null);
      setDetailError(null);
      setConsoleError(null);
      setLiveError(null);
      consoleCursor.current = undefined;
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
        consoleSequence.current += 1;
        detailController.current?.abort();
        consoleController.current?.abort();
        setDetail(null);
        setConsoleLogs([]);
        setDetailError(null);
        setConsoleError(null);
        setLiveError(null);
        setConsoleTruncated(false);
        consoleCursor.current = undefined;
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

  const loadDetail = useCallback(async (runId: string, background = false) => {
    const sequence = detailSequence.current + 1;
    detailSequence.current = sequence;
    detailController.current?.abort();
    const controller = new AbortController();
    detailController.current = controller;
    if (!background) {
      setDetailLoading(true);
      setDetailError(null);
      setLiveError(null);
      setDetail(null);
    }

    try {
      const response = await fetchTrainingRun(runId, controller.signal);
      if (controller.signal.aborted || detailSequence.current !== sequence) return;
      if (response.id !== runId) {
        const error = { status: 409, message: "The selected run changed while loading." };
        if (background) setLiveError(error);
        else setDetailError(error);
        return;
      }
      setDetail(response);
      setDetailError(null);
      setLiveError(null);
      setRuns((current) => current.map((run) => (
        run.id === response.id ? response : run
      )));
      setLastLiveUpdate(new Date().toISOString());
    } catch (reason) {
      if (controller.signal.aborted || detailSequence.current !== sequence) return;
      const error = loadError(reason, "Could not refresh this training run.");
      if (background) setLiveError(error);
      else setDetailError(error);
    } finally {
      if (!background && detailSequence.current === sequence) {
        setDetailLoading(false);
      }
    }
  }, []);

  const loadConsole = useCallback(async (runId: string, background = false) => {
    const sequence = consoleSequence.current + 1;
    consoleSequence.current = sequence;
    consoleController.current?.abort();
    const controller = new AbortController();
    consoleController.current = controller;
    if (!background) {
      setConsoleLoading(true);
      setConsoleError(null);
      setConsoleTruncated(false);
      setConsoleLogs([]);
      consoleCursor.current = undefined;
    }

    try {
      const response = await fetchTrainingConsoleLogs({
        runId,
        afterSequence: background ? consoleCursor.current : undefined,
        signal: controller.signal,
      });
      if (controller.signal.aborted || consoleSequence.current !== sequence) return;
      setConsoleLogs((current) => {
        const combined = background ? [...current, ...response.logs] : response.logs;
        const unique = new Map(combined.map((item) => [item.sequence, item]));
        return [...unique.values()]
          .sort((left, right) => left.sequence - right.sequence)
          .slice(-1_000);
      });
      setConsoleTruncated((current) => current || response.truncated);
      consoleCursor.current = response.next_sequence;
      setConsoleError(null);
      setLastLiveUpdate(new Date().toISOString());
    } catch (reason) {
      if (controller.signal.aborted || consoleSequence.current !== sequence) return;
      setConsoleError(loadError(reason, "Could not refresh trainer console output."));
    } finally {
      if (!background && consoleSequence.current === sequence) {
        setConsoleLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    void loadRuns();
    return () => {
      listSequence.current += 1;
      detailSequence.current += 1;
      consoleSequence.current += 1;
      listController.current?.abort();
      detailController.current?.abort();
      consoleController.current?.abort();
    };
  }, [loadRuns]);

  const selectedRunExists = Boolean(
    selectedRunId && runs.some((run) => run.id === selectedRunId),
  );

  useEffect(() => {
    if (!selectedRunId || !selectedRunExists) {
      setDetail(null);
      setDetailLoading(false);
      setConsoleLogs([]);
      setConsoleLoading(false);
      setConsoleTruncated(false);
      setConsoleError(null);
      setLiveError(null);
      setLastLiveUpdate(null);
      consoleCursor.current = undefined;
      return;
    }
    void loadDetail(selectedRunId);
    void loadConsole(selectedRunId);
    return () => {
      detailSequence.current += 1;
      consoleSequence.current += 1;
      detailController.current?.abort();
      consoleController.current?.abort();
    };
  }, [loadConsole, loadDetail, selectedRunExists, selectedRunId]);

  useEffect(() => {
    if (!selectedRunId || !selectedRunExists) return;
    let cancelled = false;
    let inFlight = false;
    let timer: number | undefined;

    const schedule = () => {
      if (!cancelled) timer = window.setTimeout(poll, LIVE_POLL_INTERVAL_MS);
    };
    const poll = async () => {
      if (cancelled || inFlight) return;
      if (document.visibilityState === "hidden") {
        schedule();
        return;
      }
      inFlight = true;
      try {
        await Promise.all([
          loadDetail(selectedRunId, true),
          loadConsole(selectedRunId, true),
        ]);
      } finally {
        inFlight = false;
        schedule();
      }
    };
    const handleVisibility = () => {
      if (document.visibilityState !== "visible" || inFlight) return;
      if (timer !== undefined) window.clearTimeout(timer);
      void poll();
    };
    document.addEventListener("visibilitychange", handleVisibility);
    schedule();
    return () => {
      cancelled = true;
      if (timer !== undefined) window.clearTimeout(timer);
      document.removeEventListener("visibilitychange", handleVisibility);
    };
  }, [loadConsole, loadDetail, selectedRunExists, selectedRunId]);

  const selectRun = useCallback((runId: string) => {
    if (runId === selectedRunId || !runs.some((run) => run.id === runId)) return;
    detailSequence.current += 1;
    consoleSequence.current += 1;
    detailController.current?.abort();
    consoleController.current?.abort();
    setDetail(null);
    setConsoleLogs([]);
    setConsoleTruncated(false);
    setLastLiveUpdate(null);
    setDetailError(null);
    setConsoleError(null);
    setLiveError(null);
    setDetailLoading(true);
    setConsoleLoading(true);
    consoleCursor.current = undefined;
    setSelectedRunId(runId);
  }, [runs, selectedRunId]);

  const refresh = useCallback(async () => {
    const requests: Promise<void>[] = [loadRuns(true)];
    if (selectedRunId && selectedRunExists) {
      requests.push(
        loadDetail(selectedRunId, true),
        loadConsole(selectedRunId, true),
      );
    }
    await Promise.all(requests);
  }, [loadConsole, loadDetail, loadRuns, selectedRunExists, selectedRunId]);
  const retryDetail = useCallback(() => {
    if (!selectedRunId) return Promise.resolve();
    return loadDetail(selectedRunId);
  }, [loadDetail, selectedRunId]);
  const retryConsole = useCallback(() => {
    if (!selectedRunId) return Promise.resolve();
    return loadConsole(selectedRunId, consoleLogs.length > 0);
  }, [consoleLogs.length, loadConsole, selectedRunId]);

  return {
    runs,
    selectedRunId,
    detail,
    consoleLogs,
    consoleLoading,
    consoleTruncated,
    consoleError,
    liveError,
    lastLiveUpdate,
    initialLoading,
    refreshing,
    detailLoading,
    listError,
    detailError,
    selectRun,
    refresh,
    retryList: loadRuns,
    retryDetail,
    retryConsole,
  };
}
