from typing import Dict, Optional

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(np.int64)
    acc = float((y_pred == y_true).mean())
    real_mask = y_true == 0
    fake_mask = y_true == 1
    real_acc = float((y_pred[real_mask] == y_true[real_mask]).mean()) if np.any(real_mask) else float('nan')
    fake_acc = float((y_pred[fake_mask] == y_true[fake_mask]).mean()) if np.any(fake_mask) else float('nan')
    ap = float(average_precision_score(y_true, y_prob))
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auroc = float('nan')
    return {
        'acc': acc,
        'real_acc': real_acc,
        'fake_acc': fake_acc,
        'ap': ap,
        'auroc': auroc,
    }


def recall_at_precision(y_true: np.ndarray, y_prob: np.ndarray, target_precision: float) -> Optional[float]:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    mask = precision >= target_precision
    if not np.any(mask):
        return None
    return float(np.max(recall[mask]))


def threshold_for_precision(y_true: np.ndarray, y_prob: np.ndarray, target_precision: float) -> Optional[float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return None
    precision = precision[:-1]
    mask = precision >= target_precision
    if not np.any(mask):
        return None
    idx = np.argmax(recall[:-1][mask])
    chosen = np.where(mask)[0][idx]
    return float(thresholds[chosen])
