import os
import sys
import uuid
import json
import time
import shutil
import threading
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, JSONResponse

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────
# Router setup
# ─────────────────────────────────────────────────────────────

router = APIRouter()

UPLOAD_DIR = "api/uploads"
OUTPUT_DIR = "api/outputs"

# In-memory job registry  {job_id: {...status dict...}}
_job_registry: dict = {}
_registry_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _new_job_id() -> str:
    return str(uuid.uuid4())[:12]


def _job_dir(job_id: str) -> str:
    return os.path.join(UPLOAD_DIR, job_id)


def _out_dir(job_id: str) -> str:
    return os.path.join(OUTPUT_DIR, job_id)


def _set_status(job_id: str, **kwargs):
    with _registry_lock:
        if job_id not in _job_registry:
            _job_registry[job_id] = {}
        _job_registry[job_id].update(kwargs)
        _job_registry[job_id]["job_id"]     = job_id
        _job_registry[job_id]["updated_at"] = time.time()
    # Persist to disk
    out = _out_dir(job_id)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "status.json"), "w") as f:
        json.dump(_job_registry[job_id], f, indent=2)


def _get_status(job_id: str) -> Optional[dict]:
    # Try memory first
    with _registry_lock:
        if job_id in _job_registry:
            return dict(_job_registry[job_id])
    # Fallback: read from disk
    status_path = os.path.join(_out_dir(job_id), "status.json")
    if os.path.exists(status_path):
        with open(status_path) as f:
            return json.load(f)
    return None


def _detect_image_size(job_id: str) -> tuple:
    """Return (width, height) of first image in job directory."""
    import cv2
    job_d = _job_dir(job_id)
    for fname in sorted(os.listdir(job_d)):
        fpath = os.path.join(job_d, fname)
        img = cv2.imread(fpath)
        if img is not None:
            h, w = img.shape[:2]
            return w, h
    return 640, 480


# ─────────────────────────────────────────────────────────────
# Background workers
# ─────────────────────────────────────────────────────────────

def _run_sfm_background(job_id: str, focal_length: float):
    """
    Background thread function: runs SfM and saves outputs.
    """
    try:
        import cv2
        from src.phase2_sfm.incremental_sfm import IncrementalSfM

        _set_status(job_id, phase="sfm", status="running",
                    progress=0, message="Starting SfM...")

        job_d  = _job_dir(job_id)
        out_d  = _out_dir(job_id)
        os.makedirs(out_d, exist_ok=True)

        w, h = _detect_image_size(job_id)
        K = np.array([
            [focal_length, 0,            w / 2],
            [0,            focal_length, h / 2],
            [0,            0,            1    ]
        ], dtype=np.float64)

        _set_status(job_id, progress=10, message="Detecting features...")
        sfm = IncrementalSfM(K)
        sfm.load_images(job_d, max_dim=1024)

        _set_status(job_id, progress=20, message="Matching features...")
        sfm.detect_features()
        sfm.match_features()

        _set_status(job_id, progress=40, message="Reconstructing...")
        sfm.reconstruct(bundle_adjust_interval=3)

        _set_status(job_id, progress=80, message="Saving outputs...")

        # Save PLY
        ply_path = os.path.join(out_d, "point_cloud.ply")
        sfm.save_point_cloud(ply_path)

        # Save camera poses JSON
        poses_data = {}
        for idx, (R, t) in sfm.registered_cameras.items():
            poses_data[str(idx)] = {
                "R": R.tolist(),
                "t": t.tolist(),
                "image": sfm.image_names[idx]
            }

        poses_path = os.path.join(out_d, "camera_poses.json")
        with open(poses_path, "w") as f:
            json.dump({
                "K": K.tolist(),
                "poses": poses_data,
                "image_width": w,
                "image_height": h
            }, f, indent=2)

        # Save reprojection error visualization
        mean_err = sfm.compute_mean_reprojection_error()

        _set_status(
            job_id,
            status="complete",
            phase="sfm",
            progress=100,
            message="SfM complete",
            num_cameras=len(sfm.registered_cameras),
            total_images=len(sfm.images),
            num_points=len(sfm.points_3d),
            mean_reprojection_error=round(float(mean_err), 4),
            outputs={
                "point_cloud": f"/download/{job_id}/point_cloud.ply",
                "camera_poses": f"/download/{job_id}/camera_poses.json"
            }
        )

    except Exception as exc:
        import traceback
        _set_status(
            job_id,
            status="error",
            phase="sfm",
            message=str(exc),
            traceback=traceback.format_exc()
        )


def _run_nerf_background(
    job_id: str,
    scene_dir: str,
    num_iterations: int,
    img_scale: float
):
    """
    Background thread: trains NeRF on a scene and saves checkpoint.
    """
    try:
        from src.phase4_nerf.train import NeRFTrainer

        _set_status(job_id, phase="nerf", status="running",
                    progress=5, message="Initializing NeRF...")

        out_d = _out_dir(job_id)
        ckpt_dir = os.path.join(out_d, "nerf_checkpoints")

        config = {
            "num_iterations": num_iterations,
            "batch_size": 2048,
            "img_scale": img_scale,
            "num_coarse_samples": 64,
            "num_fine_samples": 128,
            "near": 2.0,
            "far": 6.0,
            "learning_rate": 5e-4,
            "log_every": 200,
            "val_every": max(1000, num_iterations // 10),
            "save_every": num_iterations,
            "use_small_model": True,   # Faster for API usage
        }

        _set_status(job_id, progress=10, message="Loading dataset...")
        trainer = NeRFTrainer(scene_dir, output_dir=ckpt_dir, config=config)

        _set_status(job_id, progress=15, message=f"Training for {num_iterations} iterations...")
        trainer.train()

        _set_status(
            job_id,
            status="complete",
            phase="nerf",
            progress=100,
            message="NeRF training complete",
            checkpoint_dir=ckpt_dir,
            outputs={
                "checkpoint": f"/download/{job_id}/nerf_checkpoints"
            }
        )

    except Exception as exc:
        import traceback
        _set_status(
            job_id,
            status="error",
            phase="nerf",
            message=str(exc),
            traceback=traceback.format_exc()
        )


# ─────────────────────────────────────────────────────────────
# Routes — Health
# ─────────────────────────────────────────────────────────────

@router.get("/health", tags=["system"])
def health_check():
    """API health check."""
    import torch, cv2
    return {
        "status": "healthy",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "opencv_version": cv2.__version__,
        "active_jobs": len(_job_registry)
    }


@router.get("/jobs", tags=["jobs"])
def list_jobs():
    """List all jobs and their statuses."""
    with _registry_lock:
        return {"jobs": list(_job_registry.values())}


# ─────────────────────────────────────────────────────────────
# Routes — Image Upload
# ─────────────────────────────────────────────────────────────

@router.post("/upload-images", tags=["pipeline"], openapi_extra={
    "requestBody": {
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "files": {
                            "type": "array",
                            "items": {"type": "string", "format": "binary"}
                        }
                    },
                    "required": ["files"]
                }
            }
        }
    }
})
async def upload_images(files: List[UploadFile] = File(...)):
    """
    Upload multiple images for 3D reconstruction.

    - Accepts JPG, JPEG, PNG
    - Returns job_id to use in subsequent calls
    - Min recommended: 15 images; Max: 200 images
    """
    valid_extensions = {".jpg", ".jpeg", ".png"}
    job_id  = _new_job_id()
    job_d   = _job_dir(job_id)
    os.makedirs(job_d, exist_ok=True)

    saved_files   = []
    skipped_files = []

    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in valid_extensions:
            skipped_files.append(file.filename)
            continue

        safe_name = os.path.basename(file.filename)
        dest_path = os.path.join(job_d, safe_name)

        content = await file.read()
        with open(dest_path, "wb") as f:
            f.write(content)

        saved_files.append(safe_name)

    if not saved_files:
        shutil.rmtree(job_d, ignore_errors=True)
        raise HTTPException(400, "No valid images uploaded. Use JPG or PNG.")

    _set_status(
        job_id,
        status="uploaded",
        phase="upload",
        num_images=len(saved_files),
        saved_files=saved_files,
        skipped_files=skipped_files
    )

    return {
        "job_id": job_id,
        "num_images": len(saved_files),
        "saved": saved_files,
        "skipped": skipped_files,
        "next_steps": {
            "run_sfm": f"POST /api/run-sfm/{job_id}",
            "check_status": f"GET /api/status/{job_id}"
        }
    }


# ─────────────────────────────────────────────────────────────
# Routes — SfM
# ─────────────────────────────────────────────────────────────

@router.post("/run-sfm/{job_id}", tags=["pipeline"])
async def run_sfm(
    job_id: str,
    background_tasks: BackgroundTasks,
    focal_length: float = Query(
        default=800.0,
        description="Camera focal length in pixels. Use image_width*1.2 as rough estimate.",
        gt=50.0
    )
):
    """
    Start Structure-from-Motion reconstruction in the background.

    The job runs asynchronously. Poll /status/{job_id} to track progress.
    """
    job_d = _job_dir(job_id)
    if not os.path.isdir(job_d):
        raise HTTPException(404, f"Job {job_id} not found. Upload images first.")

    # Check no job already running
    status = _get_status(job_id)
    if status and status.get("status") == "running":
        raise HTTPException(409, f"Job {job_id} is already running.")

    _set_status(job_id, status="queued", phase="sfm",
                message="SfM queued", focal_length=focal_length)

    background_tasks.add_task(_run_sfm_background, job_id, focal_length)

    return {
        "job_id": job_id,
        "status": "queued",
        "message": "SfM started in background",
        "poll": f"GET /api/status/{job_id}"
    }


# ─────────────────────────────────────────────────────────────
# Routes — NeRF
# ─────────────────────────────────────────────────────────────

@router.post("/run-nerf/{job_id}", tags=["pipeline"])
async def run_nerf(
    job_id: str,
    background_tasks: BackgroundTasks,
    scene_dir: str = Query(
        default="data/nerf_synthetic/lego",
        description="Path to scene directory with transforms_train.json"
    ),
    num_iterations: int = Query(default=10000, ge=1000, le=200000),
    img_scale: float = Query(default=0.25, ge=0.1, le=1.0,
                             description="Image downscale factor for speed")
):
    """
    Start NeRF training in the background.

    Requires a scene with transforms_*.json (NeRF Synthetic format).
    For custom scenes, run SfM first and convert poses.
    """
    if not os.path.isdir(scene_dir):
        raise HTTPException(404, f"Scene directory not found: {scene_dir}")

    if not os.path.exists(os.path.join(scene_dir, "transforms_train.json")):
        raise HTTPException(
            400,
            f"No transforms_train.json in {scene_dir}. "
            "Download a NeRF Synthetic scene first."
        )

    _set_status(job_id, status="queued", phase="nerf",
                message="NeRF training queued",
                num_iterations=num_iterations,
                img_scale=img_scale)

    background_tasks.add_task(
        _run_nerf_background, job_id, scene_dir, num_iterations, img_scale
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"NeRF training started ({num_iterations} iters)",
        "poll": f"GET /api/status/{job_id}"
    }


# ─────────────────────────────────────────────────────────────
# Routes — Status & Download
# ─────────────────────────────────────────────────────────────

@router.get("/status/{job_id}", tags=["jobs"])
def get_job_status(job_id: str):
    """
    Poll job status.

    Possible status values:
    - "uploaded"  — images received, waiting to start
    - "queued"    — job is queued
    - "running"   — processing in progress
    - "complete"  — finished successfully
    - "error"     — failed (check 'message' field)
    """
    status = _get_status(job_id)
    if status is None:
        raise HTTPException(404, f"Job {job_id} not found.")
    return status


@router.delete("/jobs/{job_id}", tags=["jobs"])
def delete_job(job_id: str):
    """Delete a job and all its files."""
    status = _get_status(job_id)
    if status is None:
        raise HTTPException(404, f"Job {job_id} not found.")

    if status.get("status") == "running":
        raise HTTPException(409, "Cannot delete a running job.")

    shutil.rmtree(_job_dir(job_id),  ignore_errors=True)
    shutil.rmtree(_out_dir(job_id),  ignore_errors=True)

    with _registry_lock:
        _job_registry.pop(job_id, None)

    return {"message": f"Job {job_id} deleted."}


@router.get("/download/{job_id}/{filename:path}", tags=["outputs"])
def download_file(job_id: str, filename: str):
    """
    Download an output file.

    Common filenames:
    - point_cloud.ply
    - camera_poses.json
    - status.json
    """
    filepath = os.path.join(_out_dir(job_id), filename)
    if not os.path.exists(filepath):
        raise HTTPException(
            404,
            f"File '{filename}' not found for job {job_id}. "
            "Make sure reconstruction is complete."
        )
    return FileResponse(filepath, filename=os.path.basename(filename))


@router.get("/jobs/{job_id}/point-cloud-stats", tags=["outputs"])
def point_cloud_stats(job_id: str):
    """Return statistics about the reconstructed point cloud."""
    ply_path = os.path.join(_out_dir(job_id), "point_cloud.ply")
    if not os.path.exists(ply_path):
        raise HTTPException(404, "Point cloud not found. Run SfM first.")

    try:
        from src.utils.io_utils import load_ply
        points, colors, _ = load_ply(ply_path)

        bbox_min = points.min(axis=0).tolist()
        bbox_max = points.max(axis=0).tolist()
        centroid  = points.mean(axis=0).tolist()

        return {
            "job_id": job_id,
            "num_points": len(points),
            "has_colors": colors is not None,
            "bounding_box": {"min": bbox_min, "max": bbox_max},
            "centroid": centroid,
            "scale": float(np.linalg.norm(
                np.array(bbox_max) - np.array(bbox_min)
            ))
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to read point cloud: {e}")