"""FastAPI application and REST endpoints. Run with `python -m backend.api`."""
import csv
import io
import json
import logging
import os
import threading
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from backend.models import ExperimentInput, PRESETS
from backend.benchmarking import run_experiment
from backend import repository

app = FastAPI(title="DriftGuard XR", version="0.1.0",
              description="Controlled synthetic visual-inertial estimation research prototype.",
              docs_url="/api/docs", openapi_url="/api/openapi.json")
logger = logging.getLogger("driftguard")
# OpenCV's RANSAC RNG is global. Serialize full runs to preserve seed semantics
# and bound compute usage; concurrent readers are unaffected.
run_lock = threading.Lock()


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    logger.exception("Request failed: %s", request.url.path)
    return JSONResponse(status_code=500, content={"error": "Numerical computation or experiment storage failed. Check server logs."})


@app.get("/api/healthz")
def health():
    return {"status": "ok"}


@app.get("/api/presets")
def presets():
    return PRESETS


@app.get("/api/experiments")
def list_experiments():
    return repository.history()


def execute(config):
    if not run_lock.acquire(blocking=False):
        raise HTTPException(409, "Another experiment is running. Please retry after it completes.")
    try:
        result, details = run_experiment(config)
        repository.save(result, details)
        return result
    finally:
        run_lock.release()


@app.post("/api/experiments")
def start_experiment(config: ExperimentInput):
    return execute(config)


@app.get("/api/experiments/{experiment_id}")
def get_experiment(experiment_id: str):
    result = repository.get_result(experiment_id)
    if result is None:
        raise HTTPException(404, "Experiment not found")
    return result


@app.patch("/api/experiments/{experiment_id}")
def update_experiment(experiment_id: str, config: ExperimentInput):
    if repository.get_result(experiment_id) is None:
        raise HTTPException(404, "Experiment not found")
    # Parameter changes are new immutable runs; the source run is retained.
    return execute(config)


@app.get("/api/experiments/{experiment_id}/frames/{index}")
def frame(experiment_id: str, index: int):
    if index < 0:
        raise HTTPException(404, "Frame not found")
    result = repository.get_detail(experiment_id, index)
    if result is None:
        raise HTTPException(404, "Frame not found")
    return result


@app.get("/api/experiments/{experiment_id}/trajectories")
def trajectories(experiment_id: str):
    return get_experiment(experiment_id)["frames"]


@app.get("/api/experiments/{experiment_id}/diagnostics")
def diagnostics(experiment_id: str):
    return [{k: v for k, v in f.items() if k not in ("poses", "truth", "errors")}
            for f in get_experiment(experiment_id)["frames"]]


@app.get("/api/experiments/{experiment_id}/goi")
def goi_diagnostics(experiment_id: str):
    keys = ("index", "time", "eigenvalues", "conditionNumber", "goiMean",
            "translationObservability", "rotationObservability", "keyframe", "keyframeReason", "informationGain")
    return [{k: f[k] for k in keys} for f in get_experiment(experiment_id)["frames"]]


@app.get("/api/experiments/{experiment_id}/benchmarks")
def benchmarks(experiment_id: str):
    return get_experiment(experiment_id)["metrics"]


@app.get("/api/experiments/{experiment_id}/export/{format}")
def export(experiment_id: str, format: str):
    result = get_experiment(experiment_id)
    if format not in ("json", "csv"):
        raise HTTPException(400, "Export format must be json or csv")
    headers = {"Content-Disposition": f'attachment; filename="driftguard-{experiment_id[:8]}.{format}"'}
    if format == "json":
        # Full per-feature diagnostics alongside trajectories, metrics and provenance.
        result = dict(result)
        result["frameDetails"] = repository.get_details(experiment_id)
        return Response(json.dumps(result, allow_nan=False), media_type="application/json", headers=headers)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["frame", "time_s", "method", "x_m", "y_m", "z_m",
                     "qx", "qy", "qz", "qw", "truth_x_m", "truth_y_m", "truth_z_m",
                     "position_error_m", "tracking_state", "position_sigma_m", "keyframe"])
    for f in result["frames"]:
        for key, pose in f["poses"].items():
            writer.writerow([f["index"], f["time"], key, *pose["position"], *pose["quaternion"],
                             *f["truth"]["position"], f["errors"][key], f["state"],
                             f["positionUncertainty"], f["keyframe"]])
    return Response(output.getvalue(), media_type="text/csv", headers=headers)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api:app", host="0.0.0.0", port=int(os.environ["PORT"]), workers=1)
