"""Shared access to precomputed ORF feature files."""

from __future__ import annotations

import logging
from pathlib import Path
import threading
from typing import List, Optional, Union

import h5py
import numpy as np


logger = logging.getLogger(__name__)


class SharedFeaturesManager:
    """Load each unique set of HDF5 feature files once per process."""

    _instances = {}
    _lock = threading.Lock()

    def __new__(cls, features_files: Union[str, List[str]]):
        paths = [features_files] if isinstance(features_files, str) else features_files
        key = tuple(sorted(str(Path(path).expanduser().resolve()) for path in paths))
        with cls._lock:
            if key not in cls._instances:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instances[key] = instance
            return cls._instances[key]

    def __init__(self, features_files: Union[str, List[str]]):
        if self._initialized:
            return
        paths = [features_files] if isinstance(features_files, str) else features_files
        self.features_files = [str(Path(path).expanduser().resolve()) for path in paths]
        self.features, self.feature_ids = self._load_features()
        self.embedding_dim = int(self.features.shape[1])
        self.feature_map = self._build_feature_map(self.feature_ids)
        self._initialized = True

    def _load_features(self) -> tuple[np.ndarray, np.ndarray]:
        feature_arrays = []
        id_arrays = []
        embedding_dim = None

        for path_text in self.features_files:
            path = Path(path_text)
            if not path.is_file():
                raise FileNotFoundError(f"Feature file not found: {path}")
            with h5py.File(path, "r") as handle:
                if "features" not in handle or "ids" not in handle:
                    raise ValueError(f"Feature file must contain 'features' and 'ids': {path}")
                features = handle["features"][:]
                feature_ids = handle["ids"][:]

            if features.ndim != 2 or features.shape[0] != feature_ids.shape[0]:
                raise ValueError(f"Invalid feature shapes in {path}")
            if embedding_dim is None:
                embedding_dim = int(features.shape[1])
            elif int(features.shape[1]) != embedding_dim:
                raise ValueError(f"Embedding dimension mismatch in {path}")
            feature_arrays.append(features)
            id_arrays.append(feature_ids)

        if not feature_arrays:
            raise ValueError("At least one feature file is required")
        features = feature_arrays[0] if len(feature_arrays) == 1 else np.concatenate(feature_arrays)
        feature_ids = id_arrays[0] if len(id_arrays) == 1 else np.concatenate(id_arrays)
        return features, feature_ids

    @staticmethod
    def _build_feature_map(feature_ids: np.ndarray) -> dict[str, int]:
        mapping = {}
        for index, raw_id in enumerate(feature_ids):
            feature_id = raw_id.decode("utf-8") if isinstance(raw_id, bytes) else str(raw_id)
            if feature_id in mapping:
                logger.warning("Duplicate feature ID; keeping first: %s", feature_id)
                continue
            mapping[feature_id] = index
        return mapping

    def get_feature(
        self,
        orf_id: str,
        genome_id: Optional[str] = None,
        copy: bool = False,
    ) -> Optional[np.ndarray]:
        if not genome_id:
            raise ValueError("genome_id is required for composite feature lookup")
        index = self.feature_map.get(f"{genome_id}_{orf_id}")
        if index is None:
            return None
        feature = self.features[index]
        return feature.copy() if copy else feature


def create_shared_features_manager(
    features_files: Union[str, List[str]],
) -> SharedFeaturesManager:
    return SharedFeaturesManager(features_files)
