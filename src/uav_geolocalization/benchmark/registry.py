"""Pipeline and dataset registry exports."""

from uav_geolocalization.cli.benchmark_runner_impl import (
    DATASET_CONFIGS,
    PIPELINE_CONFIGS,
    resolve_pipeline_metadata,
)

__all__ = ["DATASET_CONFIGS", "PIPELINE_CONFIGS", "resolve_pipeline_metadata"]
