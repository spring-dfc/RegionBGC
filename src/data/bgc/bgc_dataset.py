import json
from typing import List, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from src.core import register
from utils.bgc_labels import parse_binary_labels
from utils.bgc_data import sample_orf_ids

# Optional dependency from the project-level utils
try:
    from utils.shared_features import create_shared_features_manager, SharedFeaturesManager
except Exception:  # pragma: no cover
    create_shared_features_manager = None
    SharedFeaturesManager = None


def _segments_from_binary_labels(labels: List[Union[int, float]]) -> List[Tuple[int, int]]:
    """Convert per-position binary labels to contiguous positive segments [start, end).
    Example: [0,1,1,0,1] -> [(1,3), (4,5)]
    """
    segs = []
    in_seg = False
    start = 0
    for i, v in enumerate(labels):
        is_pos = (float(v) >= 0.5)
        if is_pos and not in_seg:
            start = i
            in_seg = True
        elif (not is_pos) and in_seg:
            segs.append((start, i))
            in_seg = False
    if in_seg:
        segs.append((start, len(labels)))
    return segs


# BGC type vocabulary for multi-label classification head.
# We keep it local to the dataset so that model heads can use an independent
# "num_bgc_types" parameter while sharing the same semantics.
BGC_TYPE_VOCAB = ("PKS", "NRPS", "RiPP", "saccharide", "terpene", "other")
BGC_TYPE_TO_INDEX = {name: i for i, name in enumerate(BGC_TYPE_VOCAB)}


def _bgc_multi_hot_from_sample(sample: Dict) -> torch.Tensor:
    """Convert ``sample['bgc_class']`` into a multi-hot vector over ``BGC_TYPE_VOCAB``.

    We intentionally mirror the parsing rules used in ``tools/count_bgc_types.py``:
      - if value is a string, e.g. "PKS, NRPS/saccharide", split on , ; / | and strip;
      - if value is a list of strings ["PKS", "NRPS"], expand each string with the
        same splitting rule;
      - if value is a list of lists [["PKS"], ["PKS","NRPS"]], flatten then split.

    Background samples typically have "bgc_class": [] which yields an all-zero vector.
    """
    import re

    value = sample["bgc_class"]
    tokens: List[str] = []

    def _from_str(s: str) -> None:
        for t in re.split(r"[;,/|]", s):
            t = t.strip()
            if t:
                tokens.append(t)

    # Normalize according to the JSON patterns observed in the dataset
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("["):
            try:
                parsed = json.loads(s)
            except (TypeError, ValueError) as exc:
                raise ValueError("Invalid bgc_class JSON") from exc
            value = parsed
        else:
            _from_str(s)
    if isinstance(value, list):
        for v in value:
            if isinstance(v, str):
                _from_str(v)
            elif isinstance(v, list):
                for vv in v:
                    if isinstance(vv, str):
                        _from_str(vv)
                    else:
                        raise ValueError("bgc_class entries must be strings")
            else:
                raise ValueError("bgc_class entries must be strings or lists")
    elif not isinstance(value, str):
        raise ValueError("bgc_class must be a string or list")

    vec = torch.zeros(len(BGC_TYPE_VOCAB), dtype=torch.float32)
    for name in tokens:
        idx = BGC_TYPE_TO_INDEX.get(name)
        if idx is None:
            raise ValueError(f"Unknown BGC product class: {name!r}")
        vec[idx] = 1.0
    return vec



@register
class RegionBGCDataset(Dataset):
    """
    BGC dataset adapted for RegionBGC interval training.

    This dataset returns an ORF feature tensor and a detection target dict per sample:
      - sequence: Tensor [C,L], C=embedding dimension and L=number of ORFs
      - target: dict with keys: intervals (center,length normalized), labels, sample_id, orig_length
        Intervals are derived from contiguous per-ORF binary labels, on the ORF sequence axis.

    Args:
        data_file: path to JSON list of samples
        preloaded_data: already-loaded list of samples
        max_orfs: truncate each sample to at most this many ORFs
        features_files: str or list of HDF5 feature files with datasets 'features' and 'ids'
        use_features: if False, raises for now (sequence-encoding not implemented here)
        class_id: integer class id to use for BGC (default 0)
        dataset_name: for logging/debug
    """

    __share__ = []

    def __init__(
        self,
        data_file: Optional[str] = None,
        preloaded_data: Optional[List[Dict]] = None,
        max_orfs: int = 128,
        features_files: Optional[Union[str, List[str]]] = None,
        use_features: bool = True,
        class_id: int = 0,
        dataset_name: str = "bgc",
        strict_feature_match: bool = True,
    ) -> None:
        super().__init__()
        assert (data_file is not None) or (preloaded_data is not None), "need data_file or preloaded_data"
        self.max_orfs = max_orfs
        self.class_id = int(class_id)
        self.dataset_name = dataset_name
        # When True, raise immediately if any ORF feature cannot be found in the H5 feature map.
        # This prevents silent all-zero feature columns which can mask data/ID alignment bugs.
        if not strict_feature_match:
            raise ValueError("strict_feature_match=False is no longer supported")
        # load data
        if preloaded_data is not None:
            self.data = preloaded_data
        else:
            with open(data_file, "r") as f:
                self.data = json.load(f)
        if not isinstance(self.data, list):
            raise ValueError("RegionBGCDataset expects a list of samples")

        # features
        self.use_features = use_features
        self.embedding_dim: Optional[int] = None
        self.features_manager: Optional[SharedFeaturesManager] = None
        if self.use_features:
            if create_shared_features_manager is None:
                raise ImportError("utils.shared_features is required for use_features=True")
            if features_files is None:
                raise ValueError("features_files must be provided when use_features=True")
            self.features_manager = create_shared_features_manager(features_files)
            self.embedding_dim = int(self.features_manager.embedding_dim)
        else:
            raise NotImplementedError("use_features=False is not implemented in RegionBGCDataset")

    def __len__(self) -> int:
        return len(self.data)

    def _build_sequence_from_features(self, orf_ids: List[str], genome_id: str) -> torch.Tensor:
        """Stack per-ORF embedding vectors into a [C,L] tensor.
        Optimized to minimize per-ORF Python overhead by batching numpy copy.
        """
        orf_count = len(orf_ids)
        C = self.embedding_dim
        fm = self.features_manager
        feat_map = getattr(fm, 'feature_map', None)
        feats_np = getattr(fm, 'features', None)
        arr = np.empty((C, orf_count), dtype=np.float32)
        missing: List[str] = []
        if feat_map is not None and feats_np is not None:
            for i, oid in enumerate(orf_ids):
                if isinstance(oid, str):
                    oid_s = oid.replace('\\"', '"').replace('\\\\', '\\')
                else:
                    oid_s = str(oid)

                cid = f"{genome_id}_{oid_s}"
                idx = feat_map.get(cid)
                if idx is None:
                    missing.append(cid)
                else:
                    arr[:, i] = feats_np[idx].astype(np.float32, copy=False)
        else:
            for i, oid in enumerate(orf_ids):
                if isinstance(oid, str):
                    oid = oid.replace('\"', '"').replace('\\', '\\')
                vec = fm.get_feature(oid, genome_id, copy=False)
                if vec is None:
                    missing.append(f"{genome_id}_{str(oid)}")
                else:
                    arr[:, i] = vec.astype(np.float32, copy=False)

        if missing:
            # Show only a few examples to keep logs readable.
            examples = ", ".join(missing[:10])
            raise KeyError(
                "Missing ORF features in H5 feature_map. "
                f"dataset={self.dataset_name}, genome_id={genome_id}, "
                f"missing={len(missing)}/{orf_count}. Examples: {examples}. "
                "Please check JSON 'original_genome_id'/'orf_ids' naming and H5 'ids' naming convention "
                "(expected composite id: {original_genome_id}_{orf_id})."
            )
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite ORF features for genome {genome_id}")
        sequence_features = torch.from_numpy(arr)
        return sequence_features

    def _target_from_labels(self, labels: List[Union[int, float]], orf_count: int) -> Dict[str, torch.Tensor]:
        """
        Build normalized (center, length) intervals expected by RegionBGCCriterion.
        ORF endpoints use the half-open convention [start, end).
        """
        segs = _segments_from_binary_labels(labels)
        intervals = []
        classes = []
        for s, e in segs:
            s = max(0, min(s, orf_count))
            e = max(0, min(e, orf_count))
            if e <= s:
                continue
            x1 = s / max(orf_count, 1)
            x2 = e / max(orf_count, 1)
            cx = (x1 + x2) / 2.0
            w = max(x2 - x1, 1e-6)
            intervals.append([cx, w])
            classes.append(self.class_id)
        if len(intervals) == 0:
            intervals = torch.zeros((0, 2), dtype=torch.float32)
            classes = torch.zeros((0,), dtype=torch.int64)
        else:
            intervals = torch.tensor(intervals, dtype=torch.float32)
            classes = torch.tensor(classes, dtype=torch.int64)
        return {"intervals": intervals, "labels": classes}

    def __getitem__(self, idx: int):
        sample = self.data[idx]
        genome_id = sample.get("genome_id", f"sample_{idx}")


        original_genome_id = sample.get("original_genome_id", None)
        if original_genome_id is None or original_genome_id == "":
            original_genome_id = genome_id
            sample["original_genome_id"] = original_genome_id

        # orf ids and labels
        orf_sequences_data = sample.get("orf_sequences", [])
        orf_ids = sample_orf_ids(sample)
        # truncate
        if self.max_orfs > 0:
            orf_ids = orf_ids[: self.max_orfs]
        orf_count = len(orf_ids)

        # Features are channels arranged along the ORF sequence axis.
        orf_features = self._build_sequence_from_features(orf_ids, original_genome_id)

        # labels -> intervals/labels
        raw_labels = parse_binary_labels(
            sample['bgc_labels'], len(orf_sequences_data), context=f"Sample {genome_id}"
        )[:orf_count]

        tgt_det = self._target_from_labels(raw_labels, orf_count)

        # === BGC type multi-label targets (per positive segment) ===
        # Each sample has a sample-level "bgc_class" field, e.g. ["PKS", "NRPS"] or [].
        # We convert it to a multi-hot vector over BGC_TYPE_VOCAB and assign the same
        # vector to every positive BGC segment derived from bgc_labels.
        num_segs = tgt_det["intervals"].shape[0]
        sample_type_vec = _bgc_multi_hot_from_sample(sample)
        if num_segs > 0:
            bgc_type_vec = sample_type_vec.unsqueeze(0).repeat(num_segs, 1)
        else:
            bgc_type_vec = sample_type_vec.new_zeros((0, sample_type_vec.numel()))
        tgt_det["bgc_type_vec"] = bgc_type_vec

        target = {
            **tgt_det,
            "sample_id": torch.as_tensor(idx),
            "orig_length": torch.as_tensor(int(orf_count)),
            "length": torch.as_tensor(int(orf_count)),
        }
        return orf_features, target


@register
def bgc_collate_fn(items):
    """Collate function that matches RegionBGC expectations.
    Returns:
      orf_features: Tensor [B,C,L]
      targets: List[Dict]
    """
    orf_features = torch.cat([it[0][None] for it in items], dim=0)
    targets = [it[1] for it in items]
    return orf_features, targets
