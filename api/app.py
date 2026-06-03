"""
FastAPI application entry point.
Mounts routes from api/routes.py and serves static files.
"""

import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.routes import router

# ── Create app ──────────────────────────────────────────────
app = FastAPI(
    title="3D Reconstruction Pipeline API",
    description=(
        "End-to-end 3D reconstruction: "
        "Feature Matching → SfM → Dense MVS → NeRF"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS (allow React frontend on localhost:3000) ────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",   # Vite dev server
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount routes ─────────────────────────────────────────────
app.include_router(router, prefix="/api")

# ── Static file serving (for rendered images etc.) ──────────
os.makedirs("api/outputs", exist_ok=True)
app.mount(
    "/static",
    StaticFiles(directory="api/outputs"),
    name="static"
)

# ── Root landing page ────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>3D Reconstruction API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px;
                   margin: 60px auto; padding: 20px; background: #f5f5f5; }
            h1   { color: #2c3e50; }
            .card { background: white; padding: 20px; border-radius: 8px;
                    margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,.1); }
            a    { color: #3498db; text-decoration: none; }
            code { background: #eef; padding: 2px 6px; border-radius: 3px; }
            .badge { display:inline-block; padding:3px 10px; border-radius:12px;
                     background:#27ae60; color:white; font-size:0.85em; }
        </style>
    </head>
    <body>
        <h1>🎯 3D Reconstruction Pipeline API</h1>
        <span class="badge">v1.0.0</span>

        <div class="card">
            <h2>📖 Documentation</h2>
            <p><a href="/docs">Swagger UI (interactive)</a></p>
            <p><a href="/redoc">ReDoc (readable)</a></p>
        </div>

        <div class="card">
            <h2>⚡ Quick Start</h2>
            <ol>
                <li>Upload images: <code>POST /api/upload-images</code></li>
                <li>Run SfM: <code>POST /api/run-sfm/{job_id}</code></li>
                <li>Poll status: <code>GET /api/status/{job_id}</code></li>
                <li>Download PLY: <code>GET /api/download/{job_id}/point_cloud.ply</code></li>
                <li>Run NeRF: <code>POST /api/run-nerf/{job_id}</code></li>
            </ol>
        </div>

        <div class="card">
            <h2>🔍 System Info</h2>
            <p><a href="/api/health">Health Check</a></p>
            <p><a href="/api/jobs">All Jobs</a></p>
        </div>
    </body>
    </html>
    """


# ── Entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,           # auto-reload on code change
        log_level="info"
    )