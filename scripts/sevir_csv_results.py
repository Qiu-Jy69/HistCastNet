# Derived from Earthformer (Apache License 2.0).
# Modified for HistCastNet.
import csv
import os
from typing import Dict, Sequence

import numpy as np
import torch
from torchmetrics import Metric


CSV_COLUMNS = (
    "mode",
    "epoch",
    "group",
    "metric",
    "threshold",
    "lead_time_min",
    "frame_index",
    "value",
)


class ScalarMeanMetric(Metric):
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("value_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("value_count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, value):
        value = value.detach().float()
        self.value_sum += value
        self.value_count += torch.ones_like(value)

    def compute(self):
        return self.value_sum / torch.clamp(self.value_count, min=1.0)


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def _to_float(value):
    array = _to_numpy(value)
    if array.size == 0:
        return ""
    return float(np.mean(array.astype(np.float64)))


def _to_float_list(value):
    array = _to_numpy(value)
    if array.size == 0:
        return []
    return [float(v) for v in array.astype(np.float64).reshape(-1)]


def _threshold_sort_key(threshold):
    try:
        return float(threshold)
    except (TypeError, ValueError):
        return str(threshold)


def _threshold_label(threshold):
    try:
        value = float(threshold)
        if value.is_integer():
            return str(int(value))
    except (TypeError, ValueError):
        pass
    return str(threshold)


def _get_threshold_scores(score_dict: Dict, threshold):
    if threshold in score_dict:
        return score_dict[threshold]
    try:
        int_threshold = int(threshold)
        if int_threshold in score_dict:
            return score_dict[int_threshold]
    except (TypeError, ValueError):
        pass
    str_threshold = str(threshold)
    if str_threshold in score_dict:
        return score_dict[str_threshold]
    return {}


def _mean_threshold_metric(score_dict: Dict, metric: str, thresholds: Sequence[int]):
    values = []
    for threshold in thresholds:
        threshold_scores = _get_threshold_scores(score_dict, threshold)
        if metric not in threshold_scores:
            return None
        values.append(_to_float(threshold_scores[metric]))
    return float(np.mean(values)) if values else None


def save_sevir_test_results_csv(
    scores_dir: str,
    epoch: int,
    score_dict: Dict,
    frame_score_dict: Dict,
    extra_score_dict: Dict,
    mse,
    mae,
    metrics_list: Sequence[str],
    threshold_list: Sequence[int],
    interval_real_time: int = 10,
) -> str:
    os.makedirs(scores_dir, exist_ok=True)
    save_path = os.path.join(scores_dir, f"test_results_epoch_{epoch}.csv")
    thresholds = sorted(list(threshold_list), key=_threshold_sort_key)

    rows = [
        {
            "mode": "test",
            "epoch": epoch,
            "group": "scalar",
            "metric": "mse",
            "threshold": "",
            "lead_time_min": "all",
            "frame_index": "",
            "value": _to_float(mse),
        },
        {
            "mode": "test",
            "epoch": epoch,
            "group": "scalar",
            "metric": "mae",
            "threshold": "",
            "lead_time_min": "all",
            "frame_index": "",
            "value": _to_float(mae),
        },
    ]

    for metric in metrics_list:
        for threshold in thresholds:
            threshold_scores = _get_threshold_scores(score_dict, threshold)
            if metric not in threshold_scores:
                continue
            rows.append(
                {
                    "mode": "test",
                    "epoch": epoch,
                    "group": "threshold_summary",
                    "metric": metric,
                    "threshold": _threshold_label(threshold),
                    "lead_time_min": "all",
                    "frame_index": "",
                    "value": _to_float(threshold_scores[metric]),
                }
            )

        avg_scores = score_dict.get("avg", {})
        if metric in avg_scores:
            rows.append(
                {
                    "mode": "test",
                    "epoch": epoch,
                    "group": "threshold_average",
                    "metric": metric,
                    "threshold": "avg",
                    "lead_time_min": "all",
                    "frame_index": "",
                    "value": _to_float(avg_scores[metric]),
                }
            )

    high_thresholds = (160, 181, 219)
    for metric in ("csi", "pod", "ets", "bias"):
        high_mean = _mean_threshold_metric(score_dict, metric, high_thresholds)
        if high_mean is not None:
            rows.append(
                {
                    "mode": "test",
                    "epoch": epoch,
                    "group": "high_threshold_average",
                    "metric": metric,
                    "threshold": "160|181|219",
                    "lead_time_min": "all",
                    "frame_index": "",
                    "value": high_mean,
                }
            )

    if extra_score_dict:
        extra_metrics = []
        for threshold in thresholds:
            for metric in _get_threshold_scores(extra_score_dict, threshold).keys():
                if metric not in extra_metrics:
                    extra_metrics.append(metric)
        for metric in extra_metrics:
            for threshold in thresholds:
                threshold_scores = _get_threshold_scores(extra_score_dict, threshold)
                if metric not in threshold_scores:
                    continue
                rows.append(
                    {
                        "mode": "test",
                        "epoch": epoch,
                        "group": "threshold_summary",
                        "metric": metric,
                        "threshold": _threshold_label(threshold),
                        "lead_time_min": "all",
                        "frame_index": "",
                        "value": _to_float(threshold_scores[metric]),
                    }
                )

    if frame_score_dict:
        for metric in metrics_list:
            for threshold in thresholds:
                frame_scores = _get_threshold_scores(frame_score_dict, threshold)
                if metric not in frame_scores:
                    continue
                values = _to_float_list(frame_scores[metric])
                for frame_index, value in enumerate(values):
                    rows.append(
                        {
                            "mode": "test",
                            "epoch": epoch,
                            "group": "time_evolution",
                            "metric": metric,
                            "threshold": _threshold_label(threshold),
                            "lead_time_min": interval_real_time * (frame_index + 1),
                            "frame_index": frame_index,
                            "value": value,
                        }
                    )

    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return save_path
