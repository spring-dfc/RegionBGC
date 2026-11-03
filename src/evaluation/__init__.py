"""Standalone evaluation helpers; never imported by the training pipeline."""

from .test_evaluator import (
    evaluate_deployed_localization,
    evaluate_product_types,
    evaluate_region_output,
)

__all__ = ["evaluate_deployed_localization", "evaluate_product_types", "evaluate_region_output"]
