import { useCallback, useEffect, useRef, useState } from "react";

import { fetchArms, jogArm } from "../lib/api";
import type {
  ArmTelemetry,
  ArmsTelemetryMessage,
  JogCommand,
  TelemetryConnectionState,
} from "../types/arms";

const RECONNECT_DELAYS_MS = [500, 1_000, 2_000, 4_000, 5_000];

function websocketUrl(path: string): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}${path}`;
}

function isTelemetryMessage(value: unknown): value is ArmsTelemetryMessage {
  if (!value || typeof value !== "object") return false;
  const message = value as Partial<ArmsTelemetryMessage>;
  return message.type === "telemetry" && Array.isArray(message.arms);
}

export function useArms() {
  const [arms, setArms] = useState<ArmTelemetry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [connectionState, setConnectionState] =
    useState<TelemetryConnectionState>("connecting");
  const [commandPending, setCommandPending] = useState(false);
  const [commandError, setCommandError] = useState<string | null>(null);
  const [lastCommandAt, setLastCommandAt] = useState<string | null>(null);
  const reconnectAttempt = useRef(0);
  const reconnectTimer = useRef<number | undefined>(undefined);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetchArms();
      setArms(response.arms);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load arms.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    let active = true;
    let socket: WebSocket | undefined;

    const connect = () => {
      if (!active) return;
      setConnectionState(reconnectAttempt.current === 0 ? "connecting" : "reconnecting");
      socket = new WebSocket(websocketUrl("/ws/arms"));

      socket.addEventListener("open", () => {
        if (!active) return;
        reconnectAttempt.current = 0;
        setConnectionState("live");
      });

      socket.addEventListener("message", (event) => {
        if (!active) return;
        try {
          const message: unknown = JSON.parse(String(event.data));
          if (isTelemetryMessage(message)) {
            setArms(message.arms);
            setError(null);
          }
        } catch {
          // Ignore malformed frames and keep the last valid snapshot visible.
        }
      });

      socket.addEventListener("close", () => {
        if (!active) return;
        setConnectionState("reconnecting");
        const delay =
          RECONNECT_DELAYS_MS[
            Math.min(reconnectAttempt.current, RECONNECT_DELAYS_MS.length - 1)
          ];
        reconnectAttempt.current += 1;
        reconnectTimer.current = window.setTimeout(connect, delay);
      });

      socket.addEventListener("error", () => {
        socket?.close();
      });
    };

    connect();
    return () => {
      active = false;
      window.clearTimeout(reconnectTimer.current);
      socket?.close();
      setConnectionState("offline");
    };
  }, []);

  const sendJog = useCallback(async (armId: string, command: JogCommand) => {
    setCommandPending(true);
    setCommandError(null);
    try {
      const updated = await jogArm(armId, command);
      setArms((current) => {
        const exists = current.some((arm) => arm.id === updated.id);
        return exists
          ? current.map((arm) => (arm.id === updated.id ? updated : arm))
          : [...current, updated];
      });
      setLastCommandAt(new Date().toISOString());
    } catch (reason) {
      setCommandError(reason instanceof Error ? reason.message : "Jog command failed.");
    } finally {
      setCommandPending(false);
    }
  }, []);

  return {
    arms,
    loading,
    error,
    refresh,
    connectionState,
    commandPending,
    commandError,
    lastCommandAt,
    sendJog,
  };
}
