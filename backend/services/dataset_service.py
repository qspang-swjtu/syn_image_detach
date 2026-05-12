import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from schemas import (
    DatasetInfo,
    DatasetSplitRequest,
    DatasetSplitSummary,
    ManifestPreview,
    ManifestSource,
    MergeIndexRequest,
    MergeIndexResponse,
    SaveManifestRequest,
    SaveManifestResponse,
    ScanIncrementResponse,
    SplitFiles,
)
from utils.csv_utils import dataset_info, summarize_csv
from utils.shell import run_command


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_backend_env() -> None:
    env_path = REPO_ROOT / "backend" / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_backend_env()
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", str(REPO_ROOT / "safepp_pytorch"))).resolve()
DEFAULT_PYTORCH_PYTHON = Path(r"C:\Users\LENOVO\.conda\envs\pytorch\python.exe")
PYTHON_BIN = os.getenv(
    "PYTHON_BIN",
    str(DEFAULT_PYTORCH_PYTHON) if DEFAULT_PYTORCH_PYTHON.exists() else sys.executable,
)
DATA_ROOT = Path(os.getenv("DATA_ROOT", str(PROJECT_ROOT / "data"))).resolve()
OUTPUT_ROOT = Path(os.getenv("OUTPUT_ROOT", str(PROJECT_ROOT / "outputs"))).resolve()


def _resolve_existing_file(path_text: str) -> Path:
    candidates = []
    raw = Path(path_text)
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([REPO_ROOT / raw, PROJECT_ROOT / raw])
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists() and resolved.is_file():
            return resolved
    raise FileNotFoundError(f"File does not exist: {path_text}")


def _iteration_data_dir(iteration_id: str) -> Path:
    safe_id = iteration_id.replace("/", "_").replace("\\", "_")
    path = DATA_ROOT / "iterations" / safe_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _read_manifest(path: Path) -> Dict[str, object]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".json", ".jsonl"}:
        if path.suffix.lower() == ".jsonl":
            sources = [json.loads(line) for line in text.splitlines() if line.strip()]
            return {"sources": sources}
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {"sources": payload}
    if yaml is None:
        raise RuntimeError("PyYAML is required to read YAML manifests")
    payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError("Manifest must be a mapping with sources")
    return payload


def get_base_dataset_info(csv_path: str) -> DatasetInfo:
    return dataset_info(_resolve_existing_file(csv_path))


def get_manifest_preview(manifest_path: str) -> ManifestPreview:
    path = _resolve_existing_file(manifest_path)
    manifest = _read_manifest(path)
    raw_sources = manifest.get("sources", [])
    if not isinstance(raw_sources, list):
        raise ValueError("Manifest sources must be a list")

    sources: List[ManifestSource] = []
    warnings: List[str] = []
    for index, raw_source in enumerate(raw_sources):
        if not isinstance(raw_source, dict):
            warnings.append(f"source #{index} is not an object and was skipped")
            continue
        if "path" not in raw_source or "label" not in raw_source:
            warnings.append(f"source #{index} missing path or label")
            continue
        name = str(raw_source.get("name") or Path(str(raw_source["path"])).name or f"source_{index}")
        sources.append(
            ManifestSource(
                name=name,
                path=str(raw_source["path"]),
                label=int(raw_source["label"]),
                source=raw_source.get("source"),
                dataset=raw_source.get("dataset"),
                domain=raw_source.get("domain"),
                generator=raw_source.get("generator"),
                split_hint=raw_source.get("split_hint"),
                sample_weight=raw_source.get("sample_weight"),
                is_hard_negative=raw_source.get("is_hard_negative"),
                recursive=raw_source.get("recursive"),
            )
        )
    return ManifestPreview(manifestPath=str(path), sources=sources, estimatedRows=None, warnings=warnings)


def save_increment_manifest(req: SaveManifestRequest) -> SaveManifestResponse:
    if not req.iterationId.strip():
        raise ValueError("iterationId is required")
    if not req.sources:
        raise ValueError("sources must not be empty")

    warnings: List[str] = []
    serialized_sources: List[Dict[str, object]] = []
    for index, source in enumerate(req.sources):
        if not source.name.strip():
            raise ValueError(f"sources[{index}].name is required")
        if not source.path.strip():
            raise ValueError(f"sources[{index}].path is required")
        if source.label not in [0, 1]:
            raise ValueError(f"sources[{index}].label must be 0 or 1")
        if not str(source.split_hint or "").strip():
            raise ValueError(f"sources[{index}].split_hint is required")

        raw_path = Path(source.path)
        candidates = [raw_path] if raw_path.is_absolute() else [REPO_ROOT / raw_path, PROJECT_ROOT / raw_path]
        if not any(candidate.exists() for candidate in candidates):
            warnings.append(f"path does not exist: {source.path}")

        serialized_sources.append(
            {
                "name": source.name,
                "path": source.path,
                "label": int(source.label),
                "source": source.source or source.name,
                "dataset": source.dataset or "unknown",
                "domain": source.domain or ("real" if source.label == 0 else "fake"),
                "generator": source.generator or ("real" if source.label == 0 else "unknown"),
                "split_hint": source.split_hint,
                "sample_weight": source.sample_weight if source.sample_weight is not None else 1.0,
                "is_hard_negative": source.is_hard_negative if source.is_hard_negative is not None else 0,
                "recursive": True if source.recursive is None else bool(source.recursive),
            }
        )

    if yaml is None:
        raise RuntimeError("PyYAML is required to write YAML manifests")

    out_dir = _iteration_data_dir(req.iterationId)
    manifest_path = out_dir / "increment_manifest.yaml"
    payload = {"sources": serialized_sources}
    yaml_text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    manifest_path.write_text(yaml_text, encoding="utf-8")
    return SaveManifestResponse(
        iterationId=req.iterationId,
        manifestPath=_display_path(manifest_path),
        sourceCount=len(serialized_sources),
        yamlText=yaml_text,
        warnings=warnings,
    )


def scan_increment_manifest(iteration_id: str, increment_manifest: str) -> ScanIncrementResponse:
    manifest = _resolve_existing_file(increment_manifest)
    out_dir = _iteration_data_dir(iteration_id)
    increment_csv = out_dir / "increment_from_manifest.csv"
    summary_yaml = out_dir / "increment_from_manifest_summary.yaml"
    script = PROJECT_ROOT / "src" / "tools" / "scan_manifest_to_csv.py"
    if not script.exists():
        raise FileNotFoundError(f"scan_manifest_to_csv.py not found: {script}")

    run_command(
        [
            PYTHON_BIN,
            str(script),
            "--manifest",
            str(manifest),
            "--output_csv",
            str(increment_csv),
            "--summary_yaml",
            str(summary_yaml),
        ],
        cwd=PROJECT_ROOT,
    )
    summary = summarize_csv(increment_csv)
    return ScanIncrementResponse(
        iterationId=iteration_id,
        incrementCsv=_display_path(increment_csv),
        summaryYaml=_display_path(summary_yaml),
        rows=int(summary["rows"]),
        byLabel={str(k): int(v) for k, v in dict(summary["by_label"]).items()},
        bySplitHint={str(k): int(v) for k, v in dict(summary["by_split_hint"]).items()},
    )


def merge_dataset_index(req: MergeIndexRequest) -> MergeIndexResponse:
    base_csv = _resolve_existing_file(req.baseCsv)
    out_dir = _iteration_data_dir(req.iterationId)
    all_samples_csv = out_dir / "all_samples.csv"
    summary_yaml = out_dir / "all_samples_summary.yaml"

    if not req.incrementManifest:
        shutil.copyfile(base_csv, all_samples_csv)
        summary = summarize_csv(all_samples_csv)
        return MergeIndexResponse(
            iterationId=req.iterationId,
            allSamplesCsv=_display_path(all_samples_csv),
            summaryYaml=_display_path(summary_yaml),
            totalRows=int(summary["rows"]),
            duplicateRemoved=0,
            byLabel={str(k): int(v) for k, v in dict(summary["by_label"]).items()},
            bySplitHint={str(k): int(v) for k, v in dict(summary["by_split_hint"]).items()},
        )

    scan_result = scan_increment_manifest(req.iterationId, req.incrementManifest)
    increment_csv = _resolve_existing_file(scan_result.incrementCsv)
    script = PROJECT_ROOT / "scripts" / "merge_dataset_index.py"
    if not script.exists():
        raise FileNotFoundError(f"merge_dataset_index.py not found: {script}")

    run_command(
        [
            PYTHON_BIN,
            str(script),
            "--base_csv",
            str(base_csv),
            "--append_csvs",
            str(increment_csv),
            "--output_csv",
            str(all_samples_csv),
            "--summary_yaml",
            str(summary_yaml),
            "--added_iter",
            req.iterationId,
        ],
        cwd=PROJECT_ROOT,
    )

    summary = summarize_csv(all_samples_csv)
    duplicate_removed = 0
    if yaml is not None and summary_yaml.exists():
        raw_summary = yaml.safe_load(summary_yaml.read_text(encoding="utf-8")) or {}
        duplicate_removed = int(raw_summary.get("duplicates_removed", 0))

    return MergeIndexResponse(
        iterationId=req.iterationId,
        allSamplesCsv=_display_path(all_samples_csv),
        summaryYaml=_display_path(summary_yaml),
        totalRows=int(summary["rows"]),
        duplicateRemoved=duplicate_removed,
        byLabel={str(k): int(v) for k, v in dict(summary["by_label"]).items()},
        bySplitHint={str(k): int(v) for k, v in dict(summary["by_split_hint"]).items()},
    )


def _num_rows_from_split(summary: Dict[str, object], key: str) -> int:
    splits = summary.get("splits", {})
    if not isinstance(splits, dict):
        return 0
    item = splits.get(key, {})
    if not isinstance(item, dict):
        return 0
    return int(item.get("num_rows", 0))


def preview_dataset_split(req: DatasetSplitRequest) -> DatasetSplitSummary:
    input_csv = _resolve_existing_file(req.inputCsv)
    out_dir = _iteration_data_dir(req.iterationId)
    script = PROJECT_ROOT / "src" / "tools" / "build_full_seen_random_val.py"
    if not script.exists():
        raise FileNotFoundError(f"build_full_seen_random_val.py not found: {script}")

    run_command(
        [
            PYTHON_BIN,
            str(script),
            "--source_csv",
            str(input_csv),
            "--output_dir",
            str(out_dir),
            "--train_plan",
            req.trainPlan,
            "--val_real_total",
            str(req.valRealTotal),
            "--val_fake_total",
            str(req.valFakeTotal),
            "--seed",
            str(req.seed),
        ],
        cwd=PROJECT_ROOT,
    )
    summary_path = out_dir / "split_summary.yaml"
    if yaml is None:
        raise RuntimeError("PyYAML is required to read split_summary.yaml")
    summary = yaml.safe_load(summary_path.read_text(encoding="utf-8")) or {}
    test_unseen = _num_rows_from_split(summary, "test_unseen")
    warnings: List[str] = []
    if test_unseen == 0:
        warnings.append("当前没有 split_hint=unseen 的数据，本轮不会生成泛化测试集。")

    files = SplitFiles(
        trainBaseCsv=_display_path(out_dir / "train_base.csv"),
        trainHardCsv=_display_path(out_dir / "train_hard.csv"),
        trainStage1Csv=_display_path(out_dir / "train_stage1.csv"),
        trainStage2Csv=_display_path(out_dir / "train_stage2.csv"),
        trainStage3Csv=_display_path(out_dir / "train_stage3.csv"),
        valCsv=_display_path(out_dir / "val.csv"),
        testUnseenCsv=_display_path(out_dir / "test_unseen.csv"),
        testAllCsv=_display_path(out_dir / "test_all.csv"),
        reviewedPoolCsv=_display_path(out_dir / "reviewed_pool.csv"),
    )
    return DatasetSplitSummary(
        allInput=_num_rows_from_split(summary, "all_input"),
        seenForSplit=_num_rows_from_split(summary, "seen_for_split"),
        trainBase=_num_rows_from_split(summary, "train_base"),
        trainHard=_num_rows_from_split(summary, "train_hard"),
        trainStage1=_num_rows_from_split(summary, "train_stage1"),
        trainStage2=_num_rows_from_split(summary, "train_stage2"),
        trainStage3Initial=_num_rows_from_split(summary, "train_stage3_initial"),
        val=_num_rows_from_split(summary, "val"),
        testUnseen=test_unseen,
        testAll=_num_rows_from_split(summary, "test_all"),
        reviewedPool=_num_rows_from_split(summary, "reviewed_pool"),
        outputDir=_display_path(out_dir),
        files=files,
        warnings=warnings,
    )
