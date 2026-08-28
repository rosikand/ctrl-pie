import { useCallback, useEffect, useMemo, useRef, useState } from "react";

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

function chooseDeployment(deployments: DeploymentRead[]): string | null {
  return (
    deployments.find((deployment) =>
      ["deploying", "running", "stopping"].includes(deployment.status),
    )?.id ?? deployments[0]?.id ?? null
  );
}

export function useInference() {
  const [deployments, setDeployments] = useState<DeploymentRead[]>([]);
  const [selectedDeploymentId, setSelectedDeploymentId] = useState<string | null>(null);
  const [state, setState] = useState<InferenceStateRead | null>(null);
  const [initialLoading, setInitialLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [stateLoading, setStateLoading] = useState(false);
  const [listError, setListError] = useState<InferenceRequestError | null>(null);
  const [stateError, setStateError] = useState<InferenceRequestError | null>(null);
  const [operationError, setOperationError] = useState<InferenceRequestError | null>(null);
  const [busy, setBusy] = useState<InferenceOperation | null>(null);
  const [connection, setConnection] = useState<InferenceStreamConnection>("idle");
  const [stateReload, setStateReload] = useState(0);

  const listSequence = useRef(0);
  const listController = useRef<AbortController | null>(null);
  const operationSequence = useRef(0);
  const operationController = useRef<AbortController | null>(null);
  const selectedIdRef = useRef<string | null>(null);
  const selectedTerminalRef = useRef(false);
  const deploymentsRef = useRef<DeploymentRead[]>([]);

  const storeDeployments = useCallback((next: DeploymentRead[]) => {
    deploymentsRef.current = next;
    setDeployments(next);
  }, []);

  const upsertDeployment = useCallback((deployment: DeploymentRead) => {
    const next = [
      deployment,
      ...deploymentsRef.current.filter((item) => item.id !== deployment.id),
    ];
    storeDeployments(next);
  }, [storeDeployments]);

  const publishState = useCallback((next: InferenceStateRead) => {
    upsertDeployment(next);
    if (selectedIdRef.current === next.id) {
      selectedTerminalRef.current = isTerminal(next);
      setState(next);
      setStateError(null);
    }
  }, [upsertDeployment]);

  const loadDeployments = useCallback(async (refresh = false) => {
    const sequence = listSequence.current + 1;
    listSequence.current = sequence;
    listController.current?.abort();
    const controller = new AbortController();
    listController.current = controller;
    setListError(null);
    setInitialLoading(!refresh && deploymentsRef.current.length === 0);
    setRefreshing(refresh || deploymentsRef.current.length > 0);
    try {
      const response = await fetchInferenceDeployments(controller.signal);
      if (controller.signal.aborted || listSequence.current !== sequence) return;
      storeDeployments(response.deployments);
      setSelectedDeploymentId((current) => {
        const next = current && response.deployments.some((item) => item.id === current)
          ? current
          : chooseDeployment(response.deployments);
        selectedIdRef.current = next;
        return next;
      });
    } catch (reason) {
      if (controller.signal.aborted || listSequence.current !== sequence) return;
      setListError(requestError(reason, "Could not load inference deployments."));
    } finally {
      if (listSequence.current === sequence) {
        setInitialLoading(false);
        setRefreshing(false);
      }
    }
  }, [storeDeployments]);

  useEffect(() => {
    void loadDeployments();
    return () => {
      listSequence.current += 1;
      listController.current?.abort();
    };
  }, [loadDeployments]);

  useEffect(() => {
    if (!selectedDeploymentId) {
      setState(null);
      setStateError(null);
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
    const deploymentId = selectedDeploymentId;
    selectedTerminalRef.current = false;

    setState(null);
    setStateError(null);
    setStateLoading(true);
    setConnection("connecting");

    const connect = () => {
      if (!active || terminal || selectedTerminalRef.current) return;
      setConnection(reconnectAttempt === 0 ? "connecting" : "reconnecting");
      socket = new WebSocket(
        websocketUrl(`/api/inference/deployments/${encodeURIComponent(deploymentId)}/stream`),
      );

      socket.addEventListener("open", () => {
        if (!active) return;
        if (selectedTerminalRef.current) {
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
        if (!active || terminal || selectedTerminalRef.current) return;
        setConnection("reconnecting");
        const delay = RECONNECT_DELAYS_MS[
          Math.min(reconnectAttempt, RECONNECT_DELAYS_MS.length - 1)
        ];
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
        if (terminal) {
          setConnection("closed");
        } else {
          connect();
        }
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
  }, [publishState, selectedDeploymentId, stateReload]);

  useEffect(() => () => {
    operationSequence.current += 1;
    operationController.current?.abort();
  }, []);

  const selectDeployment = useCallback((deploymentId: string) => {
    if (deploymentId === selectedIdRef.current) return;
    selectedIdRef.current = deploymentId;
    selectedTerminalRef.current = false;
    setSelectedDeploymentId(deploymentId);
  }, []);

  const runOperation = useCallback(async <T,>(
    operation: InferenceOperation,
    task: (signal: AbortSignal) => Promise<T>,
  ): Promise<T | null> => {
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
      return result;
    } catch (reason) {
      if (controller.signal.aborted || operationSequence.current !== sequence) return null;
      setOperationError(requestError(reason, `Could not ${operation} inference.`));
      return null;
    } finally {
      if (operationSequence.current === sequence) setBusy(null);
    }
  }, []);

  const deploy = useCallback(async (payload: CreateInferenceDeploymentRequest) => {
    const created = await runOperation("deploy", (signal) =>
      createInferenceDeployment(payload, signal),
    );
    if (!created) return null;
    upsertDeployment(created);
    selectedIdRef.current = created.id;
    setSelectedDeploymentId(created.id);
    return created;
  }, [runOperation, upsertDeployment]);

  const start = useCallback(async (
    deploymentId: string,
    payload: StartInferenceSessionRequest,
  ) => {
    const started = await runOperation("start", (signal) =>
      startInferenceSession(deploymentId, payload, signal),
    );
    if (started) publishState(started);
    return started;
  }, [publishState, runOperation]);

  const stop = useCallback(async (
    deploymentId: string,
    payload: StopInferenceDeploymentRequest = {},
  ) => {
    const stopped = await runOperation("stop", (signal) =>
      stopInferenceDeployment(deploymentId, payload, signal),
    );
    if (stopped) {
      publishState(stopped);
      if (isTerminal(stopped)) setConnection("closed");
    }
    return stopped;
  }, [publishState, runOperation]);

  const selectedDeployment = useMemo(
    () => deployments.find((deployment) => deployment.id === selectedDeploymentId) ?? null,
    [deployments, selectedDeploymentId],
  );

  return {
    deployments,
    selectedDeployment,
    selectedDeploymentId,
    selectDeployment,
    state,
    initialLoading,
    refreshing,
    stateLoading,
    listError,
    stateError,
    operationError,
    clearOperationError: () => setOperationError(null),
    busy,
    connection,
    refresh: () => loadDeployments(true),
    retryState: () => setStateReload((value) => value + 1),
    deploy,
    start,
    stop,
  };
}
