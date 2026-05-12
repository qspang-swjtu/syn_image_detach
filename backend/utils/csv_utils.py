import csv
from datetime import datetime
from pathlib import Path
from typing import Dict

from schemas import DatasetInfo


def _norm(text: object, default: str = "") -> str:
    value = "" if text is None else str(text).strip()
    return value or default


def _count_map_increment(counter: Dict[str, int], key: str) -> None:
    counter[key] = int(counter.get(key, 0)) + 1


def summarize_csv(path: Path) -> Dict[str, object]:
    rows = 0
    by_label: Dict[str, int] = {}
    by_split_hint: Dict[str, int] = {}

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            label = _norm(row.get("label"), "unknown")
            split_hint = _norm(row.get("split_hint"), "seen").lower()
            _count_map_increment(by_label, label)
            _count_map_increment(by_split_hint, split_hint)

    return {
        "rows": rows,
        "by_label": by_label,
        "by_split_hint": by_split_hint,
    }


def dataset_info(path: Path) -> DatasetInfo:
    summary = summarize_csv(path)
    by_label = summary["by_label"]
    by_split_hint = summary["by_split_hint"]
    if not isinstance(by_label, dict) or not isinstance(by_split_hint, dict):
        raise RuntimeError("Invalid CSV summary")

    mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return DatasetInfo(
        csvPath=str(path),
        totalRows=int(summary["rows"]),
        realCount=int(by_label.get("0", 0)),
        fakeCount=int(by_label.get("1", 0)),
        seenCount=int(by_split_hint.get("seen", 0)),
        hardCount=sum(int(v) for k, v in by_split_hint.items() if "hard" in str(k).lower()),
        unseenCount=sum(int(v) for k, v in by_split_hint.items() if "unseen" in str(k).lower()),
        reviewedPoolCount=sum(int(v) for k, v in by_split_hint.items() if "reviewed" in str(k).lower()),
        lastModified=mtime,
    )
