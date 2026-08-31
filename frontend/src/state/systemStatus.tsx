import { createContext, useCallback, useContext, useEffect, useState } from "react";
import type { ReactNode } from "react";

import { fetchSettingsStatus, type SettingsStatus } from "../lib/api";

type SystemStatusValue = {
  status: SettingsStatus | null;
  loading: boolean;
  error: string | null;
  refresh: () => void;
};

const SystemStatusContext = createContext<SystemStatusValue>({
  status: null,
  loading: true,
  error: null,
  refresh: () => {},
});

/**
 * One authoritative copy of `/api/settings/status` for the whole shell: the
 * sidebar mode indicator, the setup banner, Overview, and Settings all read it.
 */
export function SystemStatusProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<SettingsStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(() => {
    setLoading(true);
    setError(null);
    void fetchSettingsStatus()
      .then((next) => {
        setStatus(next);
        setError(null);
      })
      .catch((reason: unknown) => {
        setError(reason instanceof Error ? reason.message : "Could not reach the backend.");
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return (
    <SystemStatusContext.Provider value={{ status, loading, error, refresh }}>
      {children}
    </SystemStatusContext.Provider>
  );
}

export function useSystemStatus(): SystemStatusValue {
  return useContext(SystemStatusContext);
}
