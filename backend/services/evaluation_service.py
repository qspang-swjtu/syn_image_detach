import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from schemas import EvalMetrics, EvaluationSummary, PredictionPage, PredictionRecord
from services.dataset_service import OUTPUT_ROOT, _display_path


DATASETS = ["val", "test_unseen", "test_all", "hard", "replay"]


def _iteration_output_dir(iteration_id: str) -> Path:
    return OUTPUT_ROOT / "iterations" / iteration_id


def _metrics_dirs(iteration_id: str) -> List[Path]:
    root = _iteration_output_dir(iteration_id)
    return [root / "eval" / "metrics", root / "release" / "metrics"]


def _predictions_path(iteration_id: str, dataset: str) -> Path:
    return _iteration_output_dir(iteration_id) / "eval" / "predictions" / f"{dataset}.csv"


def _read_json(path: Path) -> Dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _metric_float(payload: Dict[str, object], *keys: str) -> Optional[float]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return None


def _metric_int(payload: Dict[str, object], *keys: str) -> Optional[int]:
    value = _metric_float(payload, *keys)
    return int(value) if value is not None else None


def _find_metrics_file(iteration_id: str, dataset: str) -> Optional[Path]:
    for directory in _metrics_dirs(iteration_id):
        candidate = directory / f"{dataset}.json"
        if candidate.exists():
            return candidate
    return None


def get_eval_metrics(iteration_id: str, dataset: str) -> EvalMetrics:
    path = _find_metrics_file(iteration_id, dataset)
    if not path:
        warning = None
        if dataset == "test_unseen":
            warning = "当前没有 test_unseen 评估结果，可能是输入数据中没有 split_hint=unseen 的样本。"
        return EvalMetrics(dataset=dataset, exists=False, warning=warning)
    payload = _read_json(path)
    return EvalMetrics(
        dataset=dataset,
        exists=True,
        numSamples=_metric_int(payload, "numSamples", "num_samples", "n", "count"),
        acc=_metric_float(payload, "acc", "accuracy"),
        real_acc=_metric_float(payload, "real_acc"),
        fake_acc=_metric_float(payload, "fake_acc"),
        ap=_metric_float(payload, "ap", "average_precision"),
        auroc=_metric_float(payload, "auroc", "roc_auc"),
        recall_p95=_metric_float(payload, "recall_p95", "recall@p95"),
        recall_p98=_metric_float(payload, "recall_p98", "recall@p98"),
        recall_p99=_metric_float(payload, "recall_p99", "recall@p99"),
        threshold_p95=_metric_float(payload, "threshold_p95"),
        threshold_p98=_metric_float(payload, "threshold_p98"),
        threshold_p99=_metric_float(payload, "threshold_p99"),
        real_fpr=_metric_float(payload, "real_fpr"),
        fake_fnr=_metric_float(payload, "fake_fnr"),
        metricsPath=_display_path(path),
    )


def get_evaluation_summary(iteration_id: str) -> EvaluationSummary:
    output_dir = _iteration_output_dir(iteration_id)
    metrics = [get_eval_metrics(iteration_id, dataset) for dataset in DATASETS]
    warnings = [metric.warning for metric in metrics if metric.warning]
    has_metrics = any(metric.exists for metric in metrics)
    status = "success" if has_metrics else "missing"
    return EvaluationSummary(
        iterationId=iteration_id,
        status=status,
        outputDir=_display_path(output_dir),
        datasets=metrics,
        warnings=warnings,
    )


def _int_value(row: Dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, str(default)) or default))
    except ValueError:
        return default


def _float_value(row: Dict[str, str], *keys: str) -> float:
    for key in keys:
        raw = row.get(key)
        if raw not in [None, ""]:
            try:
                return float(raw)
            except ValueError:
                continue
    return 0.0


def _error_type(label: int, prediction: int) -> str:
    if label == prediction:
        return "correct"
    if label == 0 and prediction == 1:
        return "false_positive"
    return "false_negative"


def get_predictions(
    iteration_id: str,
    dataset: str,
    page: int,
    page_size: int,
    error_type: str = "all",
    source: Optional[str] = None,
    generator: Optional[str] = None,
    split_hint: Optional[str] = None,
) -> PredictionPage:
    path = _predictions_path(iteration_id, dataset)
    if not path.exists():
        return PredictionPage(iterationId=iteration_id, dataset=dataset, total=0, page=page, pageSize=page_size, records=[])

    start = max(0, (page - 1) * page_size)
    end = start + page_size
    total = 0
    records: List[PredictionRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            label = _int_value(row, "label")
            prediction = _int_value(row, "prediction")
            row_error_type = _error_type(label, prediction)
            if error_type and error_type != "all" and row_error_type != error_type:
                continue
            if source and row.get("source") != source:
                continue
            if generator and row.get("generator") != generator:
                continue
            if split_hint and row.get("split_hint") != split_hint:
                continue
            if start <= total < end:
                records.append(
                    PredictionRecord(
                        id=str(row.get("id") or row_index),
                        path=str(row.get("path") or ""),
                        label=label,
                        probability=_float_value(row, "probability", "score"),
                        prediction=prediction,
                        source=row.get("source") or None,
                        generator=row.get("generator") or None,
                        split_hint=row.get("split_hint") or None,
                        hard_type=row.get("hard_type") or None,
                        is_error=label != prediction,
                        error_type=row_error_type,
                    )
                )
            total += 1
    return PredictionPage(iterationId=iteration_id, dataset=dataset, total=total, page=page, pageSize=page_size, records=records)
