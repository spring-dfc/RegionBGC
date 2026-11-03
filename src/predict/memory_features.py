from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np

class InMemoryFeaturesManager:
    """A lightweight drop-in feature manager backed by in-memory arrays."""

    def __init__(self, features_dict: Mapping[str, np.ndarray]):
        if not features_dict:
            raise ValueError("features_dict is empty")

        feature_ids = list(features_dict.keys())
        features = [np.asarray(features_dict[feature_id], dtype=np.float32) for feature_id in feature_ids]

        embedding_dim = int(features[0].shape[0])
        for feature in features:
            if feature.ndim != 1:
                raise ValueError("Each feature must be a vector with shape [embedding_dim]")
            if int(feature.shape[0]) != embedding_dim:
                raise ValueError("All feature vectors must have the same dimension")

        self.feature_ids = np.asarray(feature_ids, dtype=object)
        self.features = np.stack(features, axis=0).astype(np.float32, copy=False)
        self.feature_map: Dict[str, int] = {feature_id: idx for idx, feature_id in enumerate(feature_ids)}
        self.embedding_dim = embedding_dim

    def get_feature(self, orf_id: str, genome_id: Optional[str] = None, copy: bool = False) -> Optional[np.ndarray]:
        if not genome_id:
            raise ValueError("genome_id is required for strict composite-id lookup")

        composite_id = f"{genome_id}_{orf_id}"
        feature_idx = self.feature_map.get(composite_id)
        if feature_idx is None:
            return None

        feature = self.features[feature_idx]
        return feature.copy() if copy else feature

    def get_stats(self) -> Dict[str, float]:
        return {
            "total_features": len(self.feature_ids),
            "embedding_dim": self.embedding_dim,
            "memory_usage_mb": float(self.features.nbytes / 1024 / 1024),
        }
