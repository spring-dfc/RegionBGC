"""Modular prediction pipeline for BGC inference."""

from .feature_extractor import extract_features_in_memory
from .memory_features import InMemoryFeaturesManager
from .predictor import run_prediction
from .prepare_dataset import convert_fasta_dir_to_dataset

__all__ = [
    "InMemoryFeaturesManager",
    "convert_fasta_dir_to_dataset",
    "extract_features_in_memory",
    "run_prediction",
]
