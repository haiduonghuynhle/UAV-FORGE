"""Native AnyVisLoc-style baseline adapter support code.

The package root stays lightweight so structure tests and metadata imports do
not require optional native dependencies such as SciPy or GPU matchers.
"""

from .types import Pose, MatchResult, CameraIntrinsics, GeoReference

__all__ = [
    "Pose",
    "MatchResult",
    "CameraIntrinsics",
    "GeoReference",
    "HomographyYawHeightSolver",
    "BenchmarkPipeline",
]


def __getattr__(name: str):
    if name == "HomographyYawHeightSolver":
        from .geometry_solver import HomographyYawHeightSolver

        return HomographyYawHeightSolver
    if name == "BenchmarkPipeline":
        from .pipeline import BenchmarkPipeline

        return BenchmarkPipeline
    raise AttributeError(name)
