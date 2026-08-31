import { useCallback, useEffect, useRef, useState } from "react";

import {
  ApiError,
  fetchTrainingConsoleLogs,
  fetchTrainingRun,
  fetchTrainingRuns,
} from "../lib/api";
import type { TrainingConsoleLog, TrainingLoadError, TrainingRun } from "../types/training";

const LIVE_POLL_INTERVAL_MS = 2_000;
const LIST_POLL_INTERVAL_MS = 5_000;

function loadError(reason: unknown, fallback: string): TrainingLoadError {
  return {
    message: reason instanceof Error ? reason.message : fallback,
    status: reason instanceof ApiError ? reason.status : null,
  };
}

/** Bounded polling that pauses while the tab is hidden. */
function usePolling(enabled: boolean, intervalMs: number, poll: () => Promise<void>) {
  const pollRef = useRef(poll);
  pollRef.current = poll;

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    let inFlight = false;
    let timer: number | undefined;

    const schedule = () => {
      if (!cancelled) timer = window.setTimeout(() => void run(), intervalMs);
    };
    const run = async () => {
      if (cancelled || inFlight) return;
      if (document.visibilityState === "hidden") {
        schedule();
        return;
      }
      inFlight = true;
      try {
        await pollRef.current();
      } finally {
        inFlight = false;
        schedule();
      }
    };
    const handleVisibility = () => {
      if (document.visibilityState !== "visible" || inFlight) return;
      if (timer !== undefined) window.clearTimeout(timer);
      void run();
    };

    document.addEventListener("visibilitychange", handleVisibility);
    schedule();
    return () => {
      cancelled = true;
      if (timer !== undefined) window.clearTimeout(timer);
      document.removeEventListener("visibilitychange", handleVisibility);
    };
  }, [enabled, intervalMs]);
}

/** The run index behind the Training table. */
export function useTrainingRunList() {
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [initialLoading, setInitialLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<TrainingLoadError | null>(null);
  const sequence = useRef(0);
  const controller = useRef<AbortController | null>(null);
  const loaded = useRef(false);

  const load = useCallback(async (background = false) => {
    const current = sequence.current + 1;
    sequence.current = current;
    controller.current?.abort();
    const abort = new AbortController();
    controller.current = abort;
    if (!background) {
      setError(null);
      setInitialLoading(!loaded.current);
      setRefreshing(loaded.current);
    }
    try {
      const response = await fetchTrainingRuns(abort.signal);
      if (abort.signal.aborted || sequence.current !== current) return;
      setRuns(response.runs);
      setError(null);
      loaded.current = true;
    } catch (reason) {
      if (abort.signal.aborted || sequence.current !== current) return;
      setError(loadError(reason, "Could not load training runs."));
    } finally {
      if (sequence.current === current) {
        setInitialLoading(false);
        setRefreshing(false);
      }
    }
  }, []);

  useEffect(() => {
    void load();
    return () => {
      sequence.current += 1;
      controller.current?.abort();
    };
  }, [load]);

  usePolling(
    true,
    LIST_POLL_INTERVAL_MS,
    useCallback(() => load(true), [load]),
  );

  return {
    runs,
    initialLoading,
    refreshing,
    error,
    refresh: useCallback(() => load(), [load]),
  };
}

/** One run with its bounded metrics and sanitized console tail. */
export function useTrainingRun(runId: string) {
  const [detail, setDetail] = useState<TrainingRun | null>(null);
  const [detailLoading, setDetailLoading] = useState(true);
  const [detailError, setDetailError] = useState<TrainingLoadError | null>(null);
  const [liveError, setLiveError] = useState<TrainingLoadError | null>(null);
  const [lastLiveUpdate, setLastLiveUpdate] = useState<string | null>(null);
  const [consoleLogs, setConsoleLogs] = useState<TrainingConsoleLog[]>([]);
  const [consoleLoading, setConsoleLoading] = useState(true);
  const [consoleTruncated, setConsoleTruncated] = useState(false);
  const [consoleError, setConsoleError] = useState<TrainingLoadError | null>(null);

  const detailSequence = useRef(0);
  const consoleSequence = useRef(0);
  const detailController = useRef<AbortController | null>(null);
  const consoleController = useRef<AbortController | null>(null);
  const consoleCursor = useRef<number | undefined>(undefined);

  const loadDetail = useCallback(
    async (background = false) => {
      if (!runId) return;
      const current = detailSequence.current + 1;
      detailSequence.current = current;
      detailController.current?.abort();
      const abort = new AbortController();
      detailController.current = abort;
      if (!background) {
        setDetailLoading(true);
        setDetailError(null);
        setLiveError(null);
      }
      try {
        const response = await fetchTrainingRun(runId, abort.signal);
        if (abort.signal.aborted || detailSequence.current !== current) return;
        if (response.id !== runId) {
          const mismatch = { status: 409, message: "The selected run changed while loading." };
          if (background) setLiveError(mismatch);
          else setDetailError(mismatch);
          return;
        }
        setDetail(response);
        setDetailError(null);
        setLiveError(null);
        setLastLiveUpdate(new Date().toISOString());
      } catch (reason) {
        if (abort.signal.aborted || detailSequence.current !== current) return;
        const failure = loadError(reason, "Could not load this training run.");
        if (background) setLiveError(failure);
        else setDetailError(failure);
      } finally {
        if (!background && detailSequence.current === current) setDetailLoading(false);
      }
    },
    [runId],
  );

  const loadConsole = useCallback(
    async (background = false) => {
      if (!runId) return;
      const current = consoleSequence.current + 1;
      consoleSequence.current = current;
      consoleController.current?.abort();
      const abort = new AbortController();
      consoleController.current = abort;
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
          signal: abort.signal,
        });
        if (abort.signal.aborted || consoleSequence.current !== current) return;
        setConsoleLogs((existing) => {
          const combined = background ? [...existing, ...response.logs] : response.logs;
          const unique = new Map(combined.map((item) => [item.sequence, item]));
          return [...unique.values()]
            .sort((left, right) => left.sequence - right.sequence)
            .slice(-1_000);
        });
        setConsoleTruncated((current2) => current2 || response.truncated);
        consoleCursor.current = response.next_sequence;
        setConsoleError(null);
      } catch (reason) {
        if (abort.signal.aborted || consoleSequence.current !== current) return;
        setConsoleError(loadError(reason, "Could not refresh trainer console output."));
      } finally {
        if (!background && consoleSequence.current === current) setConsoleLoading(false);
      }
    },
    [runId],
  );

  useEffect(() => {
    setDetail(null);
    setConsoleLogs([]);
    setConsoleTruncated(false);
    setLastLiveUpdate(null);
    consoleCursor.current = undefined;
    void loadDetail();
    void loadConsole();
    return () => {
      detailSequence.current += 1;
      consoleSequence.current += 1;
      detailController.current?.abort();
      consoleController.current?.abort();
    };
  }, [loadConsole, loadDetail]);

  usePolling(
    Boolean(runId),
    LIVE_POLL_INTERVAL_MS,
    useCallback(async () => {
      await Promise.all([loadDetail(true), loadConsole(true)]);
    }, [loadConsole, loadDetail]),
  );

  return {
    detail,
    detailLoading,
    detailError,
    liveError,
    lastLiveUpdate,
    consoleLogs,
    consoleLoading,
    consoleTruncated,
    consoleError,
    retryDetail: useCallback(() => loadDetail(), [loadDetail]),
    retryConsole: useCallback(
      () => loadConsole(consoleLogs.length > 0),
      [consoleLogs.length, loadConsole],
    ),
  };
}
