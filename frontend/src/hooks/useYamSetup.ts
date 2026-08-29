import { useCallback, useEffect, useRef, useState } from "react";

import {
  connectYamSetup,
  deleteYamSetup,
  discoverYamSetup,
  fetchYamSetup,
  preflightYamSetup,
  saveYamSetup,
} from "../lib/api";
import type {
  YamSetupConfig,
  YamSetupDiscovery,
  YamSetupPreflight,
  YamSetupStatus,
} from "../types/yamSetup";

export type YamSetupOperation =
  | "refresh"
  | "discover"
  | "preflight"
  | "save"
  | "connect"
  | "forget"
  | null;

function errorMessage(reason: unknown): string {
  return reason instanceof Error ? reason.message : "The YAM setup request failed.";
}

export function useYamSetup() {
  const [setup, setSetup] = useState<YamSetupStatus | null>(null);
  const [discovery, setDiscovery] = useState<YamSetupDiscovery | null>(null);
  const [preflight, setPreflight] = useState<YamSetupPreflight | null>(null);
  const [loading, setLoading] = useState(true);
  const [operation, setOperation] = useState<YamSetupOperation>(null);
  const [error, setError] = useState<string | null>(null);
  const [stale, setStale] = useState(false);
  const mounted = useRef(true);
  const hasSetup = useRef(false);
  const controllers = useRef(new Set<AbortController>());
  const currentController = useRef<AbortController | null>(null);
  const operationRef = useRef<YamSetupOperation>(null);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      controllers.current.forEach((controller) => controller.abort());
      controllers.current.clear();
      currentController.current = null;
      operationRef.current = null;
    };
  }, []);

  const startRequest = useCallback((nextOperation: Exclude<YamSetupOperation, null>) => {
    if (operationRef.current !== null) {
      if (operationRef.current !== "refresh" || nextOperation === "refresh") return null;
    }
    controllers.current.forEach((active) => active.abort());
    controllers.current.clear();
    const controller = new AbortController();
    controllers.current.add(controller);
    currentController.current = controller;
    operationRef.current = nextOperation;
    setOperation(nextOperation);
    return controller;
  }, []);

  const finishRequest = useCallback((controller: AbortController) => {
    controllers.current.delete(controller);
    if (currentController.current === controller) {
      currentController.current = null;
      operationRef.current = null;
    }
  }, []);

  const isCurrent = useCallback(
    (controller: AbortController) => currentController.current === controller,
    [],
  );

  const captureError = useCallback((reason: unknown) => {
    if (!mounted.current || (reason instanceof DOMException && reason.name === "AbortError")) return;
    setError(errorMessage(reason));
  }, []);

  const refresh = useCallback(async () => {
    const controller = startRequest("refresh");
    if (!controller) return;
    setError(null);
    try {
      const next = await fetchYamSetup(controller.signal);
      if (!mounted.current || controller.signal.aborted || !isCurrent(controller)) return;
      setSetup(next);
      hasSetup.current = true;
      setStale(false);
    } catch (reason) {
      if (isCurrent(controller)) captureError(reason);
      if (mounted.current && !controller.signal.aborted && isCurrent(controller)) setStale(hasSetup.current);
    } finally {
      const latest = isCurrent(controller);
      finishRequest(controller);
      if (mounted.current && !controller.signal.aborted && latest) {
        setLoading(false);
        setOperation(null);
      }
    }
  }, [captureError, finishRequest, isCurrent, startRequest]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (
      !setup
      || setup.mode !== "hardware"
      || !setup.saved
      || !setup.auto_restore
      || setup.connected
      || setup.state === "error"
      || operation !== null
    ) return;

    let timer: number | undefined;
    const schedule = (delay = 2_000) => {
      window.clearTimeout(timer);
      if (document.visibilityState !== "visible") return;
      timer = window.setTimeout(() => void refresh(), delay);
    };
    const visibilityChanged = () => {
      if (document.visibilityState === "visible") schedule(0);
      else window.clearTimeout(timer);
    };
    document.addEventListener("visibilitychange", visibilityChanged);
    schedule();
    return () => {
      window.clearTimeout(timer);
      document.removeEventListener("visibilitychange", visibilityChanged);
    };
  }, [operation, refresh, setup]);

  const discover = useCallback(async () => {
    const controller = startRequest("discover");
    if (!controller) return null;
    setError(null);
    setPreflight(null);
    try {
      const result = await discoverYamSetup(controller.signal);
      if (!mounted.current || !isCurrent(controller)) return null;
      setDiscovery(result);
      setStale(false);
      return result;
    } catch (reason) {
      if (isCurrent(controller)) captureError(reason);
      return null;
    } finally {
      const latest = isCurrent(controller);
      finishRequest(controller);
      if (mounted.current && latest) setOperation(null);
    }
  }, [captureError, finishRequest, isCurrent, startRequest]);

  const check = useCallback(async (config: YamSetupConfig) => {
    const controller = startRequest("preflight");
    if (!controller) return null;
    setError(null);
    try {
      const result = await preflightYamSetup(config, controller.signal);
      if (!mounted.current || !isCurrent(controller)) return null;
      setPreflight(result);
      return result;
    } catch (reason) {
      if (isCurrent(controller)) captureError(reason);
      return null;
    } finally {
      const latest = isCurrent(controller);
      finishRequest(controller);
      if (mounted.current && latest) setOperation(null);
    }
  }, [captureError, finishRequest, isCurrent, startRequest]);

  const save = useCallback(async (
    config: YamSetupConfig,
    autoRestore: boolean,
    acknowledgeAutomaticMotionRisk: boolean,
  ) => {
    const controller = startRequest("save");
    if (!controller) return null;
    setError(null);
    try {
      const result = await saveYamSetup({
        config,
        auto_restore: autoRestore,
        acknowledge_automatic_motion_risk: acknowledgeAutomaticMotionRisk,
      }, controller.signal);
      if (!mounted.current || !isCurrent(controller)) return null;
      setSetup(result);
      hasSetup.current = true;
      setPreflight(null);
      setStale(false);
      return result;
    } catch (reason) {
      if (isCurrent(controller)) captureError(reason);
      return null;
    } finally {
      const latest = isCurrent(controller);
      finishRequest(controller);
      if (mounted.current && latest) setOperation(null);
    }
  }, [captureError, finishRequest, isCurrent, startRequest]);

  const connect = useCallback(async (acknowledgeHardwareMotionRisk: boolean) => {
    const controller = startRequest("connect");
    if (!controller) return null;
    setError(null);
    try {
      const result = await connectYamSetup({
        acknowledge_hardware_motion_risk: acknowledgeHardwareMotionRisk,
      }, controller.signal);
      if (!mounted.current || !isCurrent(controller)) return null;
      setSetup(result);
      hasSetup.current = true;
      setStale(false);
      return result;
    } catch (reason) {
      if (isCurrent(controller)) captureError(reason);
      return null;
    } finally {
      const latest = isCurrent(controller);
      finishRequest(controller);
      if (mounted.current && latest) setOperation(null);
    }
  }, [captureError, finishRequest, isCurrent, startRequest]);

  const forget = useCallback(async () => {
    const controller = startRequest("forget");
    if (!controller) return null;
    setError(null);
    try {
      const result = await deleteYamSetup(controller.signal);
      if (!mounted.current || !isCurrent(controller)) return null;
      setSetup(result);
      hasSetup.current = true;
      setDiscovery(null);
      setPreflight(null);
      setStale(false);
      return result;
    } catch (reason) {
      if (isCurrent(controller)) captureError(reason);
      return null;
    } finally {
      const latest = isCurrent(controller);
      finishRequest(controller);
      if (mounted.current && latest) setOperation(null);
    }
  }, [captureError, finishRequest, isCurrent, startRequest]);

  const clearPreflight = useCallback(() => setPreflight(null), []);

  return {
    setup,
    discovery,
    preflight,
    loading,
    operation,
    error,
    stale,
    refresh,
    discover,
    check,
    save,
    connect,
    forget,
    clearPreflight,
  };
}
