import { useCallback, useEffect, useRef, useState } from "react";

import {
  ApiError,
  fetchDatasetEpisode,
  fetchDatasetEpisodes,
} from "../lib/api";
import type {
  DatasetEpisodeDetail,
  DatasetEpisodesResponse,
} from "../types/datasetEpisodes";

export type EpisodeLoadError = {
  message: string;
  status: number | null;
};

function loadError(reason: unknown, fallback: string): EpisodeLoadError {
  return {
    message: reason instanceof Error ? reason.message : fallback,
    status: reason instanceof ApiError ? reason.status : null,
  };
}

export function useDatasetEpisodes(repoName: string) {
  const [dataset, setDataset] = useState<DatasetEpisodesResponse | null>(null);
  const [selectedEpisodeIndex, setSelectedEpisodeIndex] = useState<number | null>(null);
  const [detail, setDetail] = useState<DatasetEpisodeDetail | null>(null);
  const [datasetLoading, setDatasetLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [datasetError, setDatasetError] = useState<EpisodeLoadError | null>(null);
  const [detailError, setDetailError] = useState<EpisodeLoadError | null>(null);
  const datasetSequence = useRef(0);
  const detailSequence = useRef(0);
  const datasetController = useRef<AbortController | null>(null);
  const detailController = useRef<AbortController | null>(null);

  const loadDataset = useCallback(async () => {
    const sequence = datasetSequence.current + 1;
    datasetSequence.current = sequence;
    detailSequence.current += 1;
    datasetController.current?.abort();
    detailController.current?.abort();
    const controller = new AbortController();
    datasetController.current = controller;

    setDatasetLoading(true);
    setDatasetError(null);
    setDetailError(null);
    setDataset(null);
    setDetail(null);
    setSelectedEpisodeIndex(null);

    try {
      const response = await fetchDatasetEpisodes(repoName, controller.signal);
      if (controller.signal.aborted || datasetSequence.current !== sequence) return;
      setDataset(response);
      setSelectedEpisodeIndex(response.episodes[0]?.episode_index ?? null);
    } catch (reason) {
      if (controller.signal.aborted || datasetSequence.current !== sequence) return;
      setDatasetError(loadError(reason, "Could not load this dataset."));
    } finally {
      if (datasetSequence.current === sequence) setDatasetLoading(false);
    }
  }, [repoName]);

  const loadDetail = useCallback(async (
    episodeIndex: number,
    revision: string,
    repoId: string,
  ) => {
    const sequence = detailSequence.current + 1;
    detailSequence.current = sequence;
    detailController.current?.abort();
    const controller = new AbortController();
    detailController.current = controller;

    setDetailLoading(true);
    setDetailError(null);
    setDetail(null);

    try {
      const response = await fetchDatasetEpisode(
        repoName,
        episodeIndex,
        revision,
        controller.signal,
      );
      if (controller.signal.aborted || detailSequence.current !== sequence) return;
      if (
        response.repo_id !== repoId ||
        response.revision !== revision ||
        response.episode.episode_index !== episodeIndex
      ) {
        setDetailError({
          status: 409,
          message: "The dataset revision changed while this episode was loading. Reload the dataset.",
        });
        return;
      }
      setDetail(response);
    } catch (reason) {
      if (controller.signal.aborted || detailSequence.current !== sequence) return;
      setDetailError(loadError(reason, "Could not load this episode."));
    } finally {
      if (detailSequence.current === sequence) setDetailLoading(false);
    }
  }, [repoName]);

  useEffect(() => {
    void loadDataset();
    return () => {
      datasetSequence.current += 1;
      detailSequence.current += 1;
      datasetController.current?.abort();
      detailController.current?.abort();
    };
  }, [loadDataset]);

  useEffect(() => {
    if (!dataset || selectedEpisodeIndex === null) {
      setDetail(null);
      setDetailLoading(false);
      return;
    }
    void loadDetail(selectedEpisodeIndex, dataset.revision, dataset.repo_id);
    return () => {
      detailSequence.current += 1;
      detailController.current?.abort();
    };
  }, [dataset, loadDetail, selectedEpisodeIndex]);

  const selectEpisode = useCallback((episodeIndex: number) => {
    if (!dataset?.episodes.some((episode) => episode.episode_index === episodeIndex)) return;
    if (episodeIndex === selectedEpisodeIndex) return;
    detailSequence.current += 1;
    detailController.current?.abort();
    setDetail(null);
    setDetailError(null);
    setDetailLoading(true);
    setSelectedEpisodeIndex(episodeIndex);
  }, [dataset, selectedEpisodeIndex]);

  const retryDetail = useCallback(() => {
    if (!dataset || selectedEpisodeIndex === null) return Promise.resolve();
    return loadDetail(selectedEpisodeIndex, dataset.revision, dataset.repo_id);
  }, [dataset, loadDetail, selectedEpisodeIndex]);

  return {
    dataset,
    selectedEpisodeIndex,
    detail,
    datasetLoading,
    detailLoading,
    datasetError,
    detailError,
    loadDataset,
    selectEpisode,
    retryDetail,
  };
}
