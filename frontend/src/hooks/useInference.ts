import { useCallback, useEffect, useRef, useState } from "react";

import {
  ApiError,
  createInferenceDeployment,
  fetchInferenceDeployments,
  fetchInferenceState,
  startInferenceSession,
  stopInferenceDeployment,
} from "../lib/api";
import type {
  CreateInferenceDeploymentRequest,
  DeploymentRead,
  InferenceStateRead,
  InferenceStreamConnection,
  InferenceStreamMessage,
  StartInferenceSessionRequest,
  StopInferenceDeploymentRequest,
} from "../types/inference";

const RECONNECT_DELAYS_MS = [500, 1_000, 2_000, 4_000, 5_000];

export type InferenceRequestError = {
  message: string;
  status: number | null;
};

export type InferenceOperation = "deploy" | "start" | "stop";

function requestError(reason: unknown, fallback: string): InferenceRequestError {
  return {
    message: reason instanceof Error ? reason.message : fallback,
    status: reason instanceof ApiError ? reason.status : null,
  };
}

function websocketUrl(path: string): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}${path}`;
}

function isTerminal(state: Pick<InferenceStateRead, "status">): boolean {
  return state.status === "stopped" || state.status === "failed";
}

function isInferenceState(value: unknown): value is InferenceStateRead {
  if (!value || typeof value !== "object") return false;
  const state = value as Partial<InferenceStateRead>;
  return (
    typeof state.id === "string" &&
    typeof state.endpoint_id === "string" &&
    typeof state.status === "string" &&
    typeof state.session_status === "string" &&
    typeof state.endpoint_healthy === "boolean" &&
    typeof state.teardown_verified === "boolean" &&
    typeof state.steps_executed === "number" &&
    typeof state.requests_completed === "number" &&
    typeof state.recording === "object" &&
    state.recording !== null
  );
}

function isInferenceMessage(value: unknown): value is InferenceStreamMessage {
  if (!value || typeof value !== "object") return false;
  const message = value as Partial<InferenceStreamMessage>;
  return (
    message.type === "inference_state" &&
    typeof message.timestamp === "string" &&
    isInferenceState(message.state)
  );
}

/** Durable deployment history behind the Inference table. */
export function useDeploymentList(pollMs = 0) {
  const [deployments, setDeployments] = useState<DeploymentRead[]>([]);
  const [initialLoading, setInitialLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<InferenceRequestError | null>(null);
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
      const response = await fetchInferenceDeployments(abort.signal);
      if (abort.signal.aborted || sequence.current !== current) return;
      setDeployments(response.deployments);
      setError(null);
      loaded.current = true;
    } catch (reason) {
      if (abort.signal.aborted || sequence.current !== current) return;
      setError(requestError(reason, "Could not load inference deployments."));
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

  useEffect(() => {
    if (!pollMs) return;
    const timer = window.setInterval(() => {
      if (document.visibilityState === "visible") void load(true);
    }, pollMs);
    return () => window.clearInterval(timer);
  }, [load, pollMs]);

  return {
    deployments,
    initialLoading,
    refreshing,
    error,
    refresh: useCallback(() => load(), [load]),
  };
}

/** Creates one deployment. Deploy never moves the robot. */
export function useDeployPolicy() {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<InferenceRequestError | null>(null);
  const controller = useRef<AbortController | null>(null);

  useEffect(() => () => controller.current?.abort(), []);

  const deploy = useCallback(async (payload: CreateInferenceDeploymentRequest) => {
    controller.current?.abort();
    const abort = new AbortController();
    controller.current = abort;
    setBusy(true);
    setError(null);
    try {
      return await createInferenceDeployment(payload, abort.signal);
    } catch (reason) {
      if (abort.signal.aborted) return null;
      setError(requestError(reason, "Could not deploy this policy."));
      return null;
    } finally {
      if (!abort.signal.aborted) setBusy(false);
    }
  }, []);

  return { deploy, busy, error, clearError: useCallback(() => setError(null), []) };
}

/**
 * One deployment's authoritative state: an initial snapshot, then the bounded
 * live stream until the deployment reaches a terminal status.
 */
export function useDeployment(deploymentId: string) {
  const [state, setState] = useState<InferenceStateRead | null>(null);
  const [stateLoading, setStateLoading] = useState(true);
  const [stateError, setStateError] = useState<InferenceRequestError | null>(null);
  const [operationError, setOperationError] = useState<InferenceRequestError | null>(null);
  const [busy, setBusy] = useState<InferenceOperation | null>(null);
  const [connection, setConnection] = useState<InferenceStreamConnection>("idle");
  const [reload, setReload] = useState(0);

  const terminalRef = useRef(false);
  const operationSequence = useRef(0);
  const operationController = useRef<AbortController | null>(null);

  const publishState = useCallback((next: InferenceStateRead) => {
    terminalRef.current = isTerminal(next);
    setState(next);
    setStateError(null);
  }, []);

  useEffect(() => {
    if (!deploymentId) {
      setState(null);
      setStateLoading(false);
      setConnection("idle");
      return;
    }

    let active = true;
    let socket: WebSocket | null = null;
    let reconnectTimer: number | undefined;
    let reconnectAttempt = 0;
    let terminal = false;
    const controller = new AbortController();
    terminalRef.current = false;

    setState(null);
    setStateError(null);
    setStateLoading(true);
    setConnection("connecting");

    const connect = () => {
      if (!active || terminal || terminalRef.current) return;
      setConnection(reconnectAttempt === 0 ? "connecting" : "reconnecting");
      socket = new WebSocket(
        websocketUrl(`/api/inference/deployments/${encodeURIComponent(deploymentId)}/stream`),
      );

      socket.addEventListener("open", () => {
        if (!active) return;
        if (terminalRef.current) {
          setConnection("closed");
          socket?.close(1000);
          return;
        }
        setConnection("live");
      });

      socket.addEventListener("message", (event) => {
        if (!active) return;
        try {
          const payload: unknown = JSON.parse(String(event.data));
          if (isInferenceMessage(payload) && payload.state.id === deploymentId) {
            reconnectAttempt = 0;
            publishState(payload.state);
            terminal = isTerminal(payload.state);
            if (terminal) {
              setConnection("closed");
              socket?.close(1000);
            }
          } else if (
            payload &&
            typeof payload === "object" &&
            (payload as { type?: unknown }).type === "error"
          ) {
            const detail = (payload as { detail?: unknown }).detail;
            terminal = true;
            setConnection("closed");
            setStateError({
              message: typeof detail === "string" ? detail : "The live stream closed safely.",
              status: null,
            });
          }
        } catch {
          // Preserve the last valid authoritative snapshot when a frame is malformed.
        }
      });

      socket.addEventListener("close", () => {
        if (!active || terminal || terminalRef.current) return;
        setConnection("reconnecting");
        const delay =
          RECONNECT_DELAYS_MS[Math.min(reconnectAttempt, RECONNECT_DELAYS_MS.length - 1)];
        reconnectAttempt += 1;
        reconnectTimer = window.setTimeout(connect, delay);
      });

      socket.addEventListener("error", () => socket?.close());
    };

    void fetchInferenceState(deploymentId, controller.signal)
      .then((snapshot) => {
        if (!active || controller.signal.aborted || snapshot.id !== deploymentId) return;
        publishState(snapshot);
        terminal = isTerminal(snapshot);
        if (terminal) setConnection("closed");
        else connect();
      })
      .catch((reason: unknown) => {
        if (!active || controller.signal.aborted) return;
        setStateError(requestError(reason, "Could not load inference state."));
        setConnection("closed");
      })
      .finally(() => {
        if (active) setStateLoading(false);
      });

    return () => {
      active = false;
      controller.abort();
      window.clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, [deploymentId, publishState, reload]);

  useEffect(
    () => () => {
      operationSequence.current += 1;
      operationController.current?.abort();
    },
    [],
  );

  const runOperation = useCallback(
    async (
      operation: InferenceOperation,
      task: (signal: AbortSignal) => Promise<InferenceStateRead>,
    ) => {
      const sequence = operationSequence.current + 1;
      operationSequence.current = sequence;
      operationController.current?.abort();
      const controller = new AbortController();
      operationController.current = controller;
      setBusy(operation);
      setOperationError(null);
      try {
        const result = await task(controller.signal);
        if (controller.signal.aborted || operationSequence.current !== sequence) return null;
        publishState(result);
        if (isTerminal(result)) setConnection("closed");
        return result;
      } catch (reason) {
        if (controller.signal.aborted || operationSequence.current !== sequence) return null;
        setOperationError(requestError(reason, `Could not ${operation} inference.`));
        return null;
      } finally {
        if (operationSequence.current === sequence) setBusy(null);
      }
    },
    [publishState],
  );

  const start = useCallback(
    (payload: StartInferenceSessionRequest) =>
      runOperation("start", (signal) => startInferenceSession(deploymentId, payload, signal)),
    [deploymentId, runOperation],
  );

  const stop = useCallback(
    (payload: StopInferenceDeploymentRequest = {}) =>
      runOperation("stop", (signal) => stopInferenceDeployment(deploymentId, payload, signal)),
    [deploymentId, runOperation],
  );

  return {
    state,
    stateLoading,
    stateError,
    operationError,
    clearOperationError: useCallback(() => setOperationError(null), []),
    busy,
    connection,
    retryState: useCallback(() => setReload((value) => value + 1), []),
    start,
    stop,
  };
}
