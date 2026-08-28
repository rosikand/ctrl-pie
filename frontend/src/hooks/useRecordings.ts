import { useCallback, useEffect, useRef, useState } from "react";

import {
  createRecording as createRecordingRequest,
  fetchRecordings,
  fetchRecordingState,
  startEpisode as startEpisodeRequest,
  startTeleop as startTeleopRequest,
  stopEpisode as stopEpisodeRequest,
  stopTeleop as stopTeleopRequest,
} from "../lib/api";
import type {
  CreateRecordingRequest,
  Recording,
  RecordingState,
  StartEpisodeRequest,
  StopEpisodeRequest,
} from "../types/recordings";

type RecordingAction = "create" | "teleop" | "episode" | null;

export function useRecordings() {
  const [recordings, setRecordings] = useState<Recording[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [state, setState] = useState<RecordingState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [stateError, setStateError] = useState<string | null>(null);
  const [activeAction, setActiveAction] = useState<RecordingAction>(null);
  const selectedIdRef = useRef("");

  const selectRecording = useCallback((recordingId: string) => {
    selectedIdRef.current = recordingId;
    setSelectedId(recordingId);
  }, []);

  const refreshRecordings = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetchRecordings();
      setRecordings(response.recordings);
      setSelectedId((current) => {
        const next = response.recordings.some((recording) => recording.id === current)
          ? current
          : (response.recordings[0]?.id ?? "");
        selectedIdRef.current = next;
        return next;
      });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load recording sessions.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshRecordings();
  }, [refreshRecordings]);

  const refreshState = useCallback(async (recordingId: string) => {
    try {
      const updated = await fetchRecordingState(recordingId);
      if (selectedIdRef.current === recordingId) setState(updated);
      setStateError(null);
      return updated;
    } catch (reason) {
      setStateError(
        reason instanceof Error ? reason.message : "Could not load session state.",
      );
      return null;
    }
  }, []);

  useEffect(() => {
    if (!selectedId) {
      setState(null);
      setStateError(null);
      return;
    }
    selectedIdRef.current = selectedId;
    setStateError(null);
    let cancelled = false;
    let timer: number | undefined;
    setState(null);

    const poll = async () => {
      await refreshState(selectedId);
      if (!cancelled) timer = window.setTimeout(poll, 500);
    };
    void poll();

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [refreshState, selectedId]);

  const syncRecording = useCallback((recordingState: RecordingState) => {
    setRecordings((current) =>
      current.map((recording) =>
        recording.id === recordingState.recording_id
          ? {
              ...recording,
              status: recordingState.status,
              episode_count: recordingState.episode_count,
            }
          : recording,
      ),
    );
  }, []);

  const runStateAction = useCallback(
    async (
      action: Exclude<RecordingAction, "create" | null>,
      request: () => Promise<RecordingState>,
    ) => {
      setActiveAction(action);
      setError(null);
      try {
        const updated = await request();
        setState(updated);
        syncRecording(updated);
        await refreshRecordings();
        return updated;
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : "Session action failed.");
        return null;
      } finally {
        setActiveAction(null);
      }
    },
    [refreshRecordings, syncRecording],
  );

  const createRecording = useCallback(
    async (payload: CreateRecordingRequest) => {
      setActiveAction("create");
      setError(null);
      try {
        const created = await createRecordingRequest(payload);
        setRecordings((current) => [created, ...current.filter((item) => item.id !== created.id)]);
        selectRecording(created.id);
        return created;
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : "Could not create the session.");
        return null;
      } finally {
        setActiveAction(null);
      }
    },
    [selectRecording],
  );

  const selectedRecording =
    recordings.find((recording) => recording.id === selectedId) ?? null;

  return {
    recordings,
    selectedRecording,
    selectedId,
    setSelectedId: selectRecording,
    state,
    loading,
    error: error ?? stateError,
    activeAction,
    refreshRecordings,
    createRecording,
    startTeleop: () =>
      selectedId
        ? runStateAction("teleop", () => startTeleopRequest(selectedId))
        : Promise.resolve(null),
    stopTeleop: () =>
      selectedId
        ? runStateAction("teleop", () => stopTeleopRequest(selectedId))
        : Promise.resolve(null),
    startEpisode: (payload: StartEpisodeRequest) =>
      selectedId
        ? runStateAction("episode", () => startEpisodeRequest(selectedId, payload))
        : Promise.resolve(null),
    stopEpisode: (payload: StopEpisodeRequest) =>
      selectedId
        ? runStateAction("episode", () => stopEpisodeRequest(selectedId, payload))
        : Promise.resolve(null),
  };
}
