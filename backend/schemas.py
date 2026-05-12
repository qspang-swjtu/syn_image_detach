from typing import Dict, List, Literal, Optional

from pydantic import BaseModel


TrainPlan = Literal["hard_in_stage1", "hard_in_stage2"]


class ApiResponse(BaseModel):
    success: bool
    data: Optional[object] = None
    error: Optional[str] = None


class DatasetInfo(BaseModel):
    csvPath: str
    totalRows: int
    realCount: int
    fakeCount: int
    seenCount: int
    hardCount: int
    unseenCount: int
    reviewedPoolCount: int
    lastModified: Optional[str] = None


class ManifestSource(BaseModel):
    name: str
    path: str
    label: Literal[0, 1]
    source: Optional[str] = None
    dataset: Optional[str] = None
    domain: Optional[str] = None
    generator: Optional[str] = None
    split_hint: Optional[str] = None
    sample_weight: Optional[float] = None
    is_hard_negative: Optional[Literal[0, 1]] = None
    recursive: Optional[bool] = None


class ManifestPreview(BaseModel):
    manifestPath: str
    sources: List[ManifestSource]
    estimatedRows: Optional[int] = None
    warnings: List[str]


class SaveManifestRequest(BaseModel):
    iterationId: str
    sources: List[ManifestSource]


class SaveManifestResponse(BaseModel):
    iterationId: str
    manifestPath: str
    sourceCount: int
    yamlText: str
    warnings: List[str]


class ScanIncrementRequest(BaseModel):
    iterationId: str
    incrementManifest: str


class ScanIncrementResponse(BaseModel):
    iterationId: str
    incrementCsv: str
    summaryYaml: str
    rows: int
    byLabel: Dict[str, int]
    bySplitHint: Dict[str, int]


class MergeIndexRequest(BaseModel):
    iterationId: str
    baseCsv: str
    incrementManifest: Optional[str] = None


class MergeIndexResponse(BaseModel):
    iterationId: str
    allSamplesCsv: str
    summaryYaml: str
    totalRows: int
    duplicateRemoved: int
    byLabel: Dict[str, int]
    bySplitHint: Dict[str, int]


class SplitFiles(BaseModel):
    trainBaseCsv: str
    trainHardCsv: str
    trainStage1Csv: str
    trainStage2Csv: str
    trainStage3Csv: str
    valCsv: str
    testUnseenCsv: str
    testAllCsv: str
    reviewedPoolCsv: str


class DatasetSplitRequest(BaseModel):
    iterationId: str
    inputCsv: str
    trainPlan: TrainPlan
    valRealTotal: int
    valFakeTotal: int
    seed: int


class DatasetSplitSummary(BaseModel):
    allInput: int
    seenForSplit: int
    trainBase: int
    trainHard: int
    trainStage1: int
    trainStage2: int
    trainStage3Initial: int
    val: int
    testUnseen: int
    testAll: int
    reviewedPool: int
    outputDir: str
    files: SplitFiles
    warnings: List[str]


class CreateIterationRequest(BaseModel):
    iterationId: str
    description: Optional[str] = None
    baseCsv: str
    incrementManifest: Optional[str] = None
    trainPlan: TrainPlan
    seed: int
    nproc: int
    valRealTotal: int
    valFakeTotal: int
    runStage1: bool
    runStage2: bool
    runReplay: bool
    runStage3: bool
    runEval: bool


class IterationCreateResponse(BaseModel):
    iterationId: str
    status: Literal["created"]
    dataDir: str
    outputDir: str
    allSamplesCsv: str
    splitSummary: DatasetSplitSummary
    configPath: Optional[str] = None


TaskStatus = Literal["created", "pending", "running", "success", "failed", "stopped"]
StageStatus = Literal["waiting", "running", "success", "failed", "skipped"]
IterationStageName = Literal[
    "merge_index",
    "split_dataset",
    "stage1_train",
    "stage2_train",
    "replay_mining",
    "stage3_train",
    "evaluation",
    "model_compare",
    "package",
]


class IterationStage(BaseModel):
    name: IterationStageName
    title: str
    status: StageStatus
    startedAt: Optional[str] = None
    finishedAt: Optional[str] = None
    durationSeconds: Optional[int] = None
    message: Optional[str] = None


class IterationTaskDetail(BaseModel):
    iterationId: str
    status: TaskStatus
    trainPlan: TrainPlan
    description: Optional[str] = None
    dataDir: str
    outputDir: str
    allSamplesCsv: Optional[str] = None
    trainStage1Csv: Optional[str] = None
    trainStage2Csv: Optional[str] = None
    trainStage3Csv: Optional[str] = None
    valCsv: Optional[str] = None
    testUnseenCsv: Optional[str] = None
    testAllCsv: Optional[str] = None
    reviewedPoolCsv: Optional[str] = None
    dataPrepared: bool
    missingPreparedFiles: List[str]
    startedAt: Optional[str] = None
    finishedAt: Optional[str] = None
    elapsedSeconds: Optional[int] = None
    progress: int
    currentStage: Optional[IterationStageName] = None
    stages: List[IterationStage]
    errorMessage: Optional[str] = None


class StartTaskRequest(BaseModel):
    force: bool = False


TaskLogLevel = Literal["INFO", "WARN", "ERROR", "DEBUG"]


class TaskLogLine(BaseModel):
    id: str
    timestamp: str
    level: TaskLogLevel
    stage: Optional[IterationStageName] = None
    message: str


class TaskLogsResponse(BaseModel):
    iterationId: str
    lines: List[TaskLogLine]
    nextCursor: Optional[str] = None


class TaskMetricPoint(BaseModel):
    step: int
    epoch: Optional[int] = None
    timestamp: Optional[str] = None
    trainLoss: Optional[float] = None
    valAp: Optional[float] = None
    recallP98: Optional[float] = None
    lr: Optional[float] = None


class TaskRuntimeMetrics(BaseModel):
    iterationId: str
    points: List[TaskMetricPoint]


EvalDatasetName = Literal["val", "test_unseen", "test_all", "hard", "replay"]


class EvalMetrics(BaseModel):
    dataset: EvalDatasetName
    exists: bool
    numSamples: Optional[int] = None
    acc: Optional[float] = None
    real_acc: Optional[float] = None
    fake_acc: Optional[float] = None
    ap: Optional[float] = None
    auroc: Optional[float] = None
    recall_p95: Optional[float] = None
    recall_p98: Optional[float] = None
    recall_p99: Optional[float] = None
    threshold_p95: Optional[float] = None
    threshold_p98: Optional[float] = None
    threshold_p99: Optional[float] = None
    real_fpr: Optional[float] = None
    fake_fnr: Optional[float] = None
    metricsPath: Optional[str] = None
    warning: Optional[str] = None


class EvaluationSummary(BaseModel):
    iterationId: str
    status: Literal["pending", "running", "success", "failed", "missing"]
    outputDir: str
    datasets: List[EvalMetrics]
    warnings: List[str]


class PredictionRecord(BaseModel):
    id: str
    path: str
    label: int
    probability: float
    prediction: int
    source: Optional[str] = None
    generator: Optional[str] = None
    split_hint: Optional[str] = None
    hard_type: Optional[str] = None
    is_error: bool
    error_type: Optional[Literal["false_positive", "false_negative", "correct"]] = None


class PredictionPage(BaseModel):
    iterationId: str
    dataset: EvalDatasetName
    total: int
    page: int
    pageSize: int
    records: List[PredictionRecord]
