"""
Evaluation utilities for benchmark runner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@dataclass
class SinglePrediction:
    image_name: str
    pred_lat: float
    pred_lon: float
    pred_alt: float
    pred_heading: float
    gt_lat: float
    gt_lon: float
    gt_alt: float
    gt_heading: float
    num_inliers: int
    total_time_ms: float
    retrieval_time_ms: float
    matching_time_ms: float
    geometry_time_ms: float
    success: bool
    failure_reason: str = ""

    @property
    def position_error_3d_m(self) -> float:
        # Local tangent approximation for short distances.
        lat_scale = 111320.0
        lon_scale = 111320.0 * np.cos(np.radians(self.gt_lat))
        d_n = (self.pred_lat - self.gt_lat) * lat_scale
        d_e = (self.pred_lon - self.gt_lon) * lon_scale
        d_u = self.pred_alt - self.gt_alt
        return float(np.sqrt(d_n * d_n + d_e * d_e + d_u * d_u))

    @property
    def position_error_2d_m(self) -> float:
        lat_scale = 111320.0
        lon_scale = 111320.0 * np.cos(np.radians(self.gt_lat))
        d_n = (self.pred_lat - self.gt_lat) * lat_scale
        d_e = (self.pred_lon - self.gt_lon) * lon_scale
        return float(np.sqrt(d_n * d_n + d_e * d_e))

    @property
    def heading_error_deg(self) -> float:
        err = (self.pred_heading - self.gt_heading + 180.0) % 360.0 - 180.0
        return float(abs(err))


@dataclass
class LocalizationMetrics:
    accuracy_at_5m: float = 0.0
    accuracy_at_10m: float = 0.0
    mean_error_m: float = float("inf")
    median_error_m: float = float("inf")
    rmse_m: float = float("inf")
    mean_error_3d_m: float = float("inf")
    median_error_3d_m: float = float("inf")
    heading_mae_deg: float = float("inf")
    success_rate: float = 0.0
    fps: float = 0.0
    position_errors: Optional[List[float]] = None

    def to_dict(self):
        return asdict(self)


class MetricsCalculator:
    def __init__(self, predictions: List[SinglePrediction]):
        self._predictions = predictions

    def compute(self) -> LocalizationMetrics:
        total = len(self._predictions)
        if total == 0:
            return LocalizationMetrics(position_errors=[])

        success = [p for p in self._predictions if p.success]
        pos_errors = np.array([p.position_error_2d_m for p in success], dtype=np.float64) if success else np.array([])
        pos_errors_3d = np.array([p.position_error_3d_m for p in success], dtype=np.float64) if success else np.array([])
        heading_errors = np.array([p.heading_error_deg for p in success], dtype=np.float64) if success else np.array([])
        total_times = np.array([max(p.total_time_ms, 1e-6) for p in self._predictions], dtype=np.float64)

        if len(pos_errors) > 0:
            accuracy_at_5m = float(np.mean(pos_errors <= 5.0))
            accuracy_at_10m = float(np.mean(pos_errors <= 10.0))
            mean_error = float(np.mean(pos_errors))
            median_error = float(np.median(pos_errors))
            rmse = float(np.sqrt(np.mean(pos_errors ** 2)))
            mean_error_3d = float(np.mean(pos_errors_3d))
            median_error_3d = float(np.median(pos_errors_3d))
            heading_mae = float(np.mean(heading_errors)) if len(heading_errors) > 0 else float("inf")
        else:
            accuracy_at_5m = 0.0
            accuracy_at_10m = 0.0
            mean_error = float("inf")
            median_error = float("inf")
            rmse = float("inf")
            mean_error_3d = float("inf")
            median_error_3d = float("inf")
            heading_mae = float("inf")

        fps = float(1000.0 / np.mean(total_times)) if len(total_times) > 0 else 0.0
        return LocalizationMetrics(
            accuracy_at_5m=accuracy_at_5m,
            accuracy_at_10m=accuracy_at_10m,
            mean_error_m=mean_error,
            median_error_m=median_error,
            rmse_m=rmse,
            mean_error_3d_m=mean_error_3d,
            median_error_3d_m=median_error_3d,
            heading_mae_deg=heading_mae,
            success_rate=float(len(success) / total),
            fps=fps,
            position_errors=pos_errors.tolist(),
        )

    def save_results(self, output_path: Path) -> None:
        metrics = self.compute()
        payload = {
            "metrics": metrics.to_dict(),
            "predictions": [asdict(p) | {"position_error_2d_m": p.position_error_2d_m, "position_error_3d_m": p.position_error_3d_m, "heading_error_deg": p.heading_error_deg} for p in self._predictions],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2, cls=NpEncoder)


@dataclass
class ProposalMetrics:
    m_me: float
    hr: float
    e_me: float

    def to_dict(self):
        return asdict(self)


class ProposalMetricsCalculator:
    def __init__(self):
        self._raw = []
        self._fgo = []
        self._valid = []

    def add_frame(self, raw_position_error: float, fgo_position_error: float, homography_valid: bool, pitch_angle_deg: float):
        _ = pitch_angle_deg
        self._raw.append(raw_position_error)
        self._fgo.append(fgo_position_error)
        self._valid.append(bool(homography_valid))

    def compute(self) -> ProposalMetrics:
        raw = np.array([v for v in self._raw if np.isfinite(v)], dtype=np.float64)
        fgo = np.array([v for v in self._fgo if np.isfinite(v)], dtype=np.float64)
        m_me = float(np.mean(raw)) if len(raw) > 0 else float("inf")
        e_me = float(np.mean(fgo)) if len(fgo) > 0 else float("inf")
        hr = float(np.mean(self._valid)) if len(self._valid) > 0 else 0.0
        return ProposalMetrics(m_me=m_me, hr=hr, e_me=e_me)


def generate_cdf_plot(errors: List[np.ndarray], output_path: Path, labels: List[str], title: str, thresholds: List[float]):
    _ = thresholds
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    for arr, label in zip(errors, labels):
        arr = np.sort(np.asarray(arr, dtype=np.float64))
        if len(arr) == 0:
            continue
        y = np.arange(1, len(arr) + 1) / len(arr)
        plt.plot(arr, y, label=label)
    plt.xlabel("Position error (m)")
    plt.ylabel("CDF")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if labels:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_latex_table(metrics_dict: Dict[str, LocalizationMetrics], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Method & A@5m & A@10m & Mean(m) & Success(\\%) \\\\",
        "\\midrule",
    ]
    for name, m in metrics_dict.items():
        lines.append(
            f"{name} & {m.accuracy_at_5m*100:.1f} & {m.accuracy_at_10m*100:.1f} & {m.mean_error_m:.2f} & {m.success_rate*100:.1f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    return "\n".join(lines)
