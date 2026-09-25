from __future__ import annotations

from typing import Iterable

import numpy as np


def classification_metrics(
    truth: Iterable[int], prediction: Iterable[int], num_classes: int = 3
) -> dict[str, float]:
    truth_array = np.asarray(list(truth), dtype=np.int64)
    prediction_array = np.asarray(list(prediction), dtype=np.int64)
    accuracy = float((truth_array == prediction_array).mean())
    f1_scores = []
    for label in range(num_classes):
        true_positive = np.sum((truth_array == label) & (prediction_array == label))
        false_positive = np.sum((truth_array != label) & (prediction_array == label))
        false_negative = np.sum((truth_array == label) & (prediction_array != label))
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1_scores.append(2 * precision * recall / max(1e-12, precision + recall))
    return {"accuracy": accuracy, "macro_f1": float(np.mean(f1_scores))}


def regression_metrics(
    truth: Iterable[float], prediction: Iterable[float]
) -> dict[str, float]:
    truth_array = np.asarray(list(truth), dtype=np.float64)
    prediction_array = np.asarray(list(prediction), dtype=np.float64)
    mae = float(np.mean(np.abs(truth_array - prediction_array)))
    if truth_array.size < 2 or np.std(truth_array) < 1e-12 or np.std(prediction_array) < 1e-12:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(truth_array, prediction_array)[0, 1])
    return {"mae": mae, "pearson": pearson}


def all_metrics(
    class_truth: Iterable[int],
    class_prediction: Iterable[int],
    regression_truth: Iterable[float],
    regression_prediction: Iterable[float],
) -> dict[str, float]:
    return {
        **classification_metrics(class_truth, class_prediction),
        **regression_metrics(regression_truth, regression_prediction),
    }
