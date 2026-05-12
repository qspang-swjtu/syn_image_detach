import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from schemas import (
    IterationStage,
    IterationTaskDetail,
    StartTaskRequest,
    TaskLogLine,
    TaskLogsResponse,
    TaskMetricPoint,
    TaskRuntimeMetrics,
)
from services.dataset_service import DATA_ROOT, OUTPUT_ROOT, PROJECT_ROOT, PYTHON_BIN, REPO_ROOT, _display_path


STAGE_DEFS = [
    ("merge_index", "数据合并"),
    ("split_dataset", "数据切分"),
    ("stage1_train", "Stage1 训练"),
    ("stage2_train", "Stage2 训练"),
    ("replay_mining", "Replay Mining"),
    ("stage3_train", "Stage3 训练"),
    ("evaluation", "自动评估"),
    ("model_compare", "模型对比"),
    ("package", "模型保存"),
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bash_env_path(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def _iteration_output_dir(iteration_id: str) -> Path:
    return OUTPUT_ROOT / "iterations" / iteration_id


def _iteration_data_dir(iteration_id: str) -> Path:
    return DATA_ROOT / "iterations" / iteration_id


def _status_path(iteration_id: str) -> Path:
    return _iteration_output_dir(iteration_id) / "task_status.json"


def _config_path(iteration_id: str) -> Path:
    return _iteration_output_dir(iteration_id) / "iteration_config.json"


def _log_path(iteration_id: str) -> Path:
    return _iteration_output_dir(iteration_id) / "run.log"


def _read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _resolve_path(path_text: object, fallback: Path) -> Path:
    if not isinstance(path_text, str) or not path_text.strip():
        return fallback
    raw = Path(path_text)
    if raw.is_absolute():
        return raw.resolve()
    candidates = [REPO_ROOT / raw, PROJECT_ROOT / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (REPO_ROOT / raw).resolve()


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _bash_candidates() -> List[Path]:
    candidates: List[Path] = []
    env_bash = os.getenv("BASH_BIN", "").strip()
    if env_bash:
        candidates.append(Path(env_bash))
    which_bash = shutil.which("bash")
    if which_bash:
        candidates.append(Path(which_bash))
    candidates.extend(
        [
            Path(r"C:\Program Files\Git\bin\bash.exe"),
            Path(r"C:\Program Files\Git\usr\bin\bash.exe"),
            Path(r"C:\Program Files (x86)\Git\bin\bash.exe"),
        ]
    )
    seen = set()
    unique: List[Path] = []
    for candidate in candidates:
        text = str(candidate)
        if text.lower() not in seen:
            unique.append(candidate)
            seen.add(text.lower())
    return unique


def _resolve_bash_bin() -> Path:
    checked: List[str] = []
    for candidate in _bash_candidates():
        checked.append(str(candidate))
        if not candidate.exists():
            continue
        try:
            result = subprocess.run(
                [str(candidate), "--version"],
                text=True,
                capture_output=True,
                timeout=5,
                check=False,
            )
        except Exception:
            continue
        version_text = f"{result.stdout}\n{result.stderr}".lower()
        if result.returncode == 0 and "bash" in version_text:
            return candidate
    raise RuntimeError(
        "未找到可用的 bash。当前 Windows WSL bash 不可用，会跳转 Microsoft Store。"
        "请安装 Git for Windows，并在 backend/.env 设置 BASH_BIN=C:\\Program Files\\Git\\bin\\bash.exe，"
        f"然后重启后端。已检查: {', '.join(checked) or 'none'}"
    )


def _pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        result = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True, check=False)
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _load_config(iteration_id: str) -> Dict[str, object]:
    config = _read_json(_config_path(iteration_id))
    if not config:
        raise FileNotFoundError(f"iteration_config.json not found for {iteration_id}")
    return config


def _stage_success_files(iteration_id: str, config: Dict[str, object]) -> Dict[str, bool]:
    data_dir = _resolve_path(config.get("dataDir"), _iteration_data_dir(iteration_id))
    output_dir = _resolve_path(config.get("outputDir"), _iteration_output_dir(iteration_id))
    metrics_dir = output_dir / "eval" / "metrics"
    return {
        "merge_index": (data_dir / "all_samples.csv").exists(),
        "split_dataset": (data_dir / "split_summary.json").exists() or (data_dir / "split_summary.yaml").exists(),
        "stage1_train": (output_dir / "stage1" / "best.pt").exists(),
        "stage2_train": (output_dir / "stage2" / "best.pt").exists(),
        "replay_mining": (output_dir / "replay_buffer.csv").exists(),
        "stage3_train": (output_dir / "stage3" / "best.pt").exists(),
        "evaluation": metrics_dir.exists() and any(metrics_dir.glob("*.json")),
        "model_compare": False,
        "package": (output_dir / "release" / "manifest.json").exists(),
    }


def _infer_current_stage(log_text: str) -> Optional[str]:
    checks = [
        ("stage3_train", ["stage3", "Stage3"]),
        ("replay_mining", ["replay", "Replay"]),
        ("stage2_train", ["stage2", "Stage2"]),
        ("stage1_train", ["stage1", "Stage1"]),
        ("evaluation", ["eval", "评估", "evaluation"]),
    ]
    for stage, tokens in checks:
        if any(token in log_text for token in tokens):
            return stage
    return None


def _read_log_text(iteration_id: str) -> str:
    path = _log_path(iteration_id)
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _build_stages(iteration_id: str, config: Dict[str, object], status_payload: Dict[str, object]) -> List[IterationStage]:
    success = _stage_success_files(iteration_id, config)
    current_stage = str(status_payload.get("currentStage") or _infer_current_stage(_read_log_text(iteration_id)) or "")
    stages: List[IterationStage] = []
    for name, title in STAGE_DEFS:
        if name == "model_compare":
            stage_status = "skipped"
        elif name == "replay_mining" and config.get("runReplay") is False:
            stage_status = "skipped"
        elif name == "stage3_train" and config.get("runStage3") is False:
            stage_status = "skipped"
        elif success[name]:
            stage_status = "success"
        elif current_stage == name and status_payload.get("status") == "running":
            stage_status = "running"
        else:
            stage_status = "waiting"
        stages.append(IterationStage(name=name, title=title, status=stage_status))
    return stages


def _progress_from_stages(stages: List[IterationStage]) -> int:
    if not stages:
        return 0
    done = sum(1 for stage in stages if stage.status in ["success", "skipped"])
    return int(round(done / len(stages) * 100))


def _prepared_file_paths(iteration_id: str, config: Dict[str, object]) -> Dict[str, Path]:
    data_dir = _resolve_path(config.get("dataDir"), _iteration_data_dir(iteration_id))
    return {
        "allSamplesCsv": _resolve_path(config.get("allSamplesCsv"), data_dir / "all_samples.csv"),
        "trainStage1Csv": _resolve_path(config.get("trainStage1Csv"), data_dir / "train_stage1.csv"),
        "trainStage2Csv": _resolve_path(config.get("trainStage2Csv"), data_dir / "train_stage2.csv"),
        "trainStage3Csv": _resolve_path(config.get("trainStage3Csv"), data_dir / "train_stage3.csv"),
        "valCsv": _resolve_path(config.get("valCsv"), data_dir / "val.csv"),
        "testUnseenCsv": _resolve_path(config.get("testUnseenCsv"), data_dir / "test_unseen.csv"),
        "testAllCsv": _resolve_path(config.get("testAllCsv"), data_dir / "test_all.csv"),
        "reviewedPoolCsv": _resolve_path(config.get("reviewedPoolCsv"), data_dir / "reviewed_pool.csv"),
    }


def _missing_required_prepared_files(iteration_id: str, config: Dict[str, object]) -> List[str]:
    files = _prepared_file_paths(iteration_id, config)
    required_keys = ["allSamplesCsv", "trainStage1Csv", "trainStage2Csv", "trainStage3Csv", "valCsv"]
    return [_display_path(files[key]) for key in required_keys if not files[key].exists()]


def get_iteration_task(iteration_id: str) -> IterationTaskDetail:
    config = _load_config(iteration_id)
    status_payload = _read_json(_status_path(iteration_id))
    pid = int(status_payload.get("pid", 0) or 0)
    status = str(status_payload.get("status") or "created")
    if status == "running" and not _pid_running(pid):
        status = "failed"
        status_payload["status"] = status
        status_payload["finishedAt"] = status_payload.get("finishedAt") or _now_iso()
        status_payload["errorMessage"] = status_payload.get("errorMessage") or "后台进程已退出，请查看 run.log。"
        _write_json(_status_path(iteration_id), status_payload)

    stages = _build_stages(iteration_id, config, status_payload)
    if status == "running" and all(stage.status in ["success", "skipped"] for stage in stages):
        status = "success"
    started_at = status_payload.get("startedAt")
    elapsed = None
    if isinstance(started_at, str):
        try:
            elapsed = int((datetime.now(timezone.utc) - datetime.fromisoformat(started_at)).total_seconds())
        except Exception:
            elapsed = None
    data_dir = _resolve_path(config.get("dataDir"), _iteration_data_dir(iteration_id))
    output_dir = _resolve_path(config.get("outputDir"), _iteration_output_dir(iteration_id))
    prepared_files = _prepared_file_paths(iteration_id, config)
    missing_prepared = _missing_required_prepared_files(iteration_id, config)
    current_stage = next((stage.name for stage in stages if stage.status == "running"), None)
    return IterationTaskDetail(
        iterationId=iteration_id,
        status=status,
        trainPlan=str(config.get("trainPlan") or "hard_in_stage2"),
        description=config.get("description") if isinstance(config.get("description"), str) else None,
        dataDir=_display_path(data_dir),
        outputDir=_display_path(output_dir),
        allSamplesCsv=_display_path(prepared_files["allSamplesCsv"]),
        trainStage1Csv=_display_path(prepared_files["trainStage1Csv"]),
        trainStage2Csv=_display_path(prepared_files["trainStage2Csv"]),
        trainStage3Csv=_display_path(prepared_files["trainStage3Csv"]),
        valCsv=_display_path(prepared_files["valCsv"]),
        testUnseenCsv=_display_path(prepared_files["testUnseenCsv"]),
        testAllCsv=_display_path(prepared_files["testAllCsv"]),
        reviewedPoolCsv=_display_path(prepared_files["reviewedPoolCsv"]),
        dataPrepared=len(missing_prepared) == 0,
        missingPreparedFiles=missing_prepared,
        startedAt=started_at if isinstance(started_at, str) else None,
        finishedAt=status_payload.get("finishedAt") if isinstance(status_payload.get("finishedAt"), str) else None,
        elapsedSeconds=elapsed,
        progress=_progress_from_stages(stages),
        currentStage=current_stage,
        stages=stages,
        errorMessage=status_payload.get("errorMessage") if isinstance(status_payload.get("errorMessage"), str) else None,
    )


def start_iteration_task(iteration_id: str, req: StartTaskRequest) -> IterationTaskDetail:
    config = _load_config(iteration_id)
    current = get_iteration_task(iteration_id)
    if current.status == "running" and not req.force:
        return current
    if current.status == "success" and not req.force:
        return current
    missing_prepared = _missing_required_prepared_files(iteration_id, config)
    if missing_prepared:
        raise FileNotFoundError(
            "数据尚未准备完成，请先在新建迭代任务页面完成启动前检查/创建任务。缺失文件: "
            + ", ".join(missing_prepared)
        )

    data_dir = _resolve_path(config.get("dataDir"), _iteration_data_dir(iteration_id))
    output_dir = _resolve_path(config.get("outputDir"), _iteration_output_dir(iteration_id))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = _log_path(iteration_id)
    script = PROJECT_ROOT / "scripts" / "run_iteration_increment_manifest_only.sh"
    if not script.exists():
        raise FileNotFoundError(f"run_iteration_increment_manifest_only.sh not found: {script}")
    bash_bin = _resolve_bash_bin()

    env = os.environ.copy()
    env.update(
        {
            "USE_PREPARED_DATA": "1",
            "ITER_ID": iteration_id,
            "DATA_DIR": str(data_dir),
            "OUT_DIR": str(output_dir),
            "TRAIN_PLAN": str(config.get("trainPlan") or "hard_in_stage2"),
            "SEED": str(config.get("seed") or 3407),
            "NPROC": str(config.get("nproc") or 8),
            "RUN_STAGE1": "1" if config.get("runStage1") is not False else "0",
            "RUN_STAGE2": "1" if config.get("runStage2") is not False else "0",
            "RUN_REPLAY": "1" if config.get("runReplay") is not False else "0",
            "RUN_STAGE3": "1" if config.get("runStage3") is not False else "0",
            "RUN_EVAL": "1" if config.get("runEval") is not False else "0",
            "RUN_SCAN": "0",
            "RUN_SPLIT": "0",
            "RUN_FULL_SCAN": "0",
            "PYTHON_BIN": _bash_env_path(PYTHON_BIN),
        }
    )
    handle = log_file.open("a", encoding="utf-8")
    handle.write(f"\n[{_now_iso()}] INFO starting iteration {iteration_id}\n")
    handle.flush()
    process = subprocess.Popen(
        [str(bash_bin), str(script)],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    _write_json(
        _status_path(iteration_id),
        {
            "iterationId": iteration_id,
            "status": "running",
            "pid": process.pid,
            "startedAt": _now_iso(),
            "currentStage": "stage1_train",
            "progress": 0,
        },
    )
    return get_iteration_task(iteration_id)


def get_iteration_stages(iteration_id: str) -> List[IterationStage]:
    return get_iteration_task(iteration_id).stages


def _level_for_line(text: str):
    lowered = text.lower()
    if "error" in lowered or "traceback" in lowered or "failed" in lowered:
        return "ERROR"
    if "warn" in lowered or "warning" in lowered:
        return "WARN"
    if "debug" in lowered:
        return "DEBUG"
    return "INFO"


def get_task_logs(iteration_id: str, cursor: Optional[str] = None, level: Optional[str] = None) -> TaskLogsResponse:
    path = _log_path(iteration_id)
    if not path.exists():
        return TaskLogsResponse(iterationId=iteration_id, lines=[], nextCursor="0")
    raw_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    start = int(cursor) if cursor and cursor.isdigit() else max(0, len(raw_lines) - 500)
    selected = raw_lines[start:]
    lines: List[TaskLogLine] = []
    for offset, text in enumerate(selected, start=start):
        log_level = _level_for_line(text)
        if level and log_level != level:
            continue
        lines.append(
            TaskLogLine(
                id=str(offset),
                timestamp="",
                level=log_level,
                message=text,
            )
        )
    return TaskLogsResponse(iterationId=iteration_id, lines=lines, nextCursor=str(len(raw_lines)))


def get_runtime_metrics(iteration_id: str) -> TaskRuntimeMetrics:
    path = _log_path(iteration_id)
    if not path.exists():
        return TaskRuntimeMetrics(iterationId=iteration_id, points=[])
    pattern = re.compile(
        r"(?:step[=:\s]+(?P<step>\d+))?.*?(?:epoch[=:\s]+(?P<epoch>\d+))?.*?"
        r"(?:train_loss|loss)[=:\s]+(?P<loss>\d+(?:\.\d+)?).*?"
        r"(?:val_ap|ap)[=:\s]+(?P<ap>\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )
    points: List[TaskMetricPoint] = []
    for index, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        match = pattern.search(line)
        if not match:
            continue
        points.append(
            TaskMetricPoint(
                step=int(match.group("step") or len(points) + 1),
                epoch=int(match.group("epoch")) if match.group("epoch") else None,
                trainLoss=float(match.group("loss")),
                valAp=float(match.group("ap")),
            )
        )
        if len(points) >= 1000:
            break
    return TaskRuntimeMetrics(iterationId=iteration_id, points=points)
