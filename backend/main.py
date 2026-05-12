from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
    ApiResponse,
    CreateIterationRequest,
    DatasetSplitRequest,
    MergeIndexRequest,
    StartTaskRequest,
    SaveManifestRequest,
    ScanIncrementRequest,
)
from services.dataset_service import (
    get_base_dataset_info,
    get_manifest_preview,
    merge_dataset_index,
    preview_dataset_split,
    save_increment_manifest,
    scan_increment_manifest,
)
from services.iteration_service import create_iteration
from services.task_service import (
    get_iteration_stages,
    get_iteration_task,
    get_runtime_metrics,
    get_task_logs,
    start_iteration_task,
)
from services.evaluation_service import get_eval_metrics, get_evaluation_summary, get_predictions


app = FastAPI(title="SafePP Iteration API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def ok(data: object) -> ApiResponse:
    return ApiResponse(success=True, data=data)


def fail(exc: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"success": False, "error": str(exc.detail)})


@app.get("/api/datasets/base-info")
def api_base_info(csv_path: str = Query(...)):
    try:
        return ok(get_base_dataset_info(csv_path))
    except Exception as exc:
        raise fail(exc)


@app.get("/api/datasets/manifest-preview")
def api_manifest_preview(manifest_path: str = Query(...)):
    try:
        return ok(get_manifest_preview(manifest_path))
    except Exception as exc:
        raise fail(exc)


@app.post("/api/datasets/scan-increment")
def api_scan_increment(req: ScanIncrementRequest):
    try:
        return ok(scan_increment_manifest(req.iterationId, req.incrementManifest))
    except Exception as exc:
        raise fail(exc)


@app.post("/api/datasets/save-increment-manifest")
def api_save_increment_manifest(req: SaveManifestRequest):
    try:
        return ok(save_increment_manifest(req))
    except Exception as exc:
        raise fail(exc)


@app.post("/api/datasets/merge-index")
def api_merge_index(req: MergeIndexRequest):
    try:
        return ok(merge_dataset_index(req))
    except Exception as exc:
        raise fail(exc)


@app.post("/api/datasets/split-preview")
def api_split_preview(req: DatasetSplitRequest):
    try:
        return ok(preview_dataset_split(req))
    except Exception as exc:
        raise fail(exc)


@app.post("/api/iterations")
def api_create_iteration(req: CreateIterationRequest):
    try:
        return ok(create_iteration(req))
    except Exception as exc:
        raise fail(exc)


@app.post("/api/iterations/{iteration_id}/start")
def api_start_iteration(iteration_id: str, req: StartTaskRequest):
    try:
        return ok(start_iteration_task(iteration_id, req))
    except Exception as exc:
        raise fail(exc)


@app.get("/api/iterations/{iteration_id}")
def api_get_iteration(iteration_id: str):
    try:
        return ok(get_iteration_task(iteration_id))
    except Exception as exc:
        raise fail(exc)


@app.get("/api/iterations/{iteration_id}/stages")
def api_get_iteration_stages(iteration_id: str):
    try:
        return ok({"iterationId": iteration_id, "stages": get_iteration_stages(iteration_id)})
    except Exception as exc:
        raise fail(exc)


@app.get("/api/iterations/{iteration_id}/logs")
def api_get_iteration_logs(iteration_id: str, cursor: str = "", level: str = ""):
    try:
        return ok(get_task_logs(iteration_id, cursor=cursor or None, level=level or None))
    except Exception as exc:
        raise fail(exc)


@app.get("/api/iterations/{iteration_id}/runtime-metrics")
def api_get_runtime_metrics(iteration_id: str):
    try:
        return ok(get_runtime_metrics(iteration_id))
    except Exception as exc:
        raise fail(exc)


@app.get("/api/iterations/{iteration_id}/evaluation/summary")
def api_get_evaluation_summary(iteration_id: str):
    try:
        return ok(get_evaluation_summary(iteration_id))
    except Exception as exc:
        raise fail(exc)


@app.get("/api/iterations/{iteration_id}/evaluation/metrics")
def api_get_eval_metrics(iteration_id: str, dataset: str):
    try:
        return ok(get_eval_metrics(iteration_id, dataset))
    except Exception as exc:
        raise fail(exc)


@app.get("/api/iterations/{iteration_id}/evaluation/predictions")
def api_get_predictions(
    iteration_id: str,
    dataset: str,
    page: int = 1,
    page_size: int = 20,
    error_type: str = "all",
    source: str = "",
    generator: str = "",
    split_hint: str = "",
):
    try:
        return ok(
            get_predictions(
                iteration_id=iteration_id,
                dataset=dataset,
                page=page,
                page_size=page_size,
                error_type=error_type,
                source=source or None,
                generator=generator or None,
                split_hint=split_hint or None,
            )
        )
    except Exception as exc:
        raise fail(exc)
