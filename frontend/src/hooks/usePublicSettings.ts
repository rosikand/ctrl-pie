import { useCallback, useEffect, useRef, useState } from "react";

import { fetchPublicSettings, type PublicSettings } from "../lib/api";

/** Non-secret backend defaults: namespace, recording FPS, runtime, compute. */
export function usePublicSettings() {
  const [settings, setSettings] = useState<PublicSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const controller = useRef<AbortController | null>(null);
  const sequence = useRef(0);

  const load = useCallback(async () => {
    const current = sequence.current + 1;
    sequence.current = current;
    controller.current?.abort();
    const abort = new AbortController();
    controller.current = abort;
    setLoading(true);
    setError(null);
    try {
      const response = await fetchPublicSettings(abort.signal);
      if (abort.signal.aborted || sequence.current !== current) return;
      setSettings(response);
    } catch (reason) {
      if (abort.signal.aborted || sequence.current !== current) return;
      setError(reason instanceof Error ? reason.message : "Could not load settings.");
    } finally {
      if (sequence.current === current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
    return () => {
      sequence.current += 1;
      controller.current?.abort();
    };
  }, [load]);

  return { settings, loading, error, refresh: load, setSettings };
}
