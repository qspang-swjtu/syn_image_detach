import json

from schemas import CreateIterationRequest, DatasetSplitRequest, IterationCreateResponse, MergeIndexRequest
from services.dataset_service import DATA_ROOT, OUTPUT_ROOT, _display_path, merge_dataset_index, preview_dataset_split


def create_iteration(req: CreateIterationRequest) -> IterationCreateResponse:
    merge_resp = merge_dataset_index(
        MergeIndexRequest(
            iterationId=req.iterationId,
            baseCsv=req.baseCsv,
            incrementManifest=req.incrementManifest,
        )
    )
    split_summary = preview_dataset_split(
        DatasetSplitRequest(
            iterationId=req.iterationId,
            inputCsv=merge_resp.allSamplesCsv,
            trainPlan=req.trainPlan,
            valRealTotal=req.valRealTotal,
            valFakeTotal=req.valFakeTotal,
            seed=req.seed,
        )
    )
    out_dir = OUTPUT_ROOT / "iterations" / req.iterationId
    data_dir = DATA_ROOT / "iterations" / req.iterationId
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "iteration_config.json"
    config = req.model_dump()
    config.update(
        {
            "iterationId": req.iterationId,
            "allSamplesCsv": merge_resp.allSamplesCsv,
            "dataDir": _display_path(data_dir),
            "outputDir": _display_path(out_dir),
            "trainStage1Csv": split_summary.files.trainStage1Csv,
            "trainStage2Csv": split_summary.files.trainStage2Csv,
            "trainStage3Csv": split_summary.files.trainStage3Csv,
            "valCsv": split_summary.files.valCsv,
            "testUnseenCsv": split_summary.files.testUnseenCsv,
            "testAllCsv": split_summary.files.testAllCsv,
            "reviewedPoolCsv": split_summary.files.reviewedPoolCsv,
            "splitSummary": split_summary.model_dump(),
            "status": "created",
        }
    )
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    return IterationCreateResponse(
        iterationId=req.iterationId,
        status="created",
        dataDir=_display_path(data_dir),
        outputDir=_display_path(out_dir),
        allSamplesCsv=merge_resp.allSamplesCsv,
        splitSummary=split_summary,
        configPath=_display_path(config_path),
    )
