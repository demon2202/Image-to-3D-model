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
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta name="description" content="State-of-the-art interactive 3D Reconstruction Web Dashboard: Feature Matching, Structure-from-Motion, Dense MVS, and NeRF.">
        <title>3D Reconstruction Studio</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
        
        <style>
            :root {
                --bg-primary: #090d16;
                --bg-secondary: #0f1626;
                --card-bg: rgba(22, 33, 54, 0.65);
                --card-border: rgba(255, 255, 255, 0.08);
                --text-primary: #f1f5f9;
                --text-muted: #94a3b8;
                --accent-blue: #3b82f6;
                --accent-blue-glow: rgba(59, 130, 246, 0.4);
                --accent-green: #10b981;
                --accent-green-glow: rgba(16, 185, 129, 0.4);
                --accent-purple: #8b5cf6;
                --accent-red: #f43f5e;
                --transition-fast: 0.2s cubic-bezier(0.4, 0, 0.2, 1);
                --transition-normal: 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }

            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }

            body {
                font-family: 'Outfit', sans-serif;
                background-color: var(--bg-primary);
                background-image: 
                    radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.12) 0px, transparent 50%),
                    radial-gradient(at 100% 100%, rgba(139, 92, 246, 0.1) 0px, transparent 50%),
                    radial-gradient(at 50% 50%, rgba(16, 185, 129, 0.05) 0px, transparent 70%);
                background-attachment: fixed;
                color: var(--text-primary);
                min-height: 100vh;
                line-height: 1.5;
                padding-bottom: 60px;
            }

            header {
                backdrop-filter: blur(12px);
                border-bottom: 1px solid var(--card-border);
                padding: 1.5rem 2rem;
                position: sticky;
                top: 0;
                z-index: 100;
                display: flex;
                justify-content: space-between;
                align-items: center;
                background: rgba(9, 13, 22, 0.8);
            }

            .logo-container {
                display: flex;
                align-items: center;
                gap: 0.75rem;
            }

            .logo-icon {
                font-size: 2rem;
                background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                animation: rotateLogo 8s linear infinite;
            }

            @keyframes rotateLogo {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            .logo-text {
                font-size: 1.5rem;
                font-weight: 700;
                letter-spacing: -0.025em;
                background: linear-gradient(to right, #ffffff, #cbd5e1);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }

            .api-badge {
                background: rgba(59, 130, 246, 0.15);
                border: 1px solid rgba(59, 130, 246, 0.3);
                color: #60a5fa;
                padding: 0.25rem 0.75rem;
                border-radius: 9999px;
                font-size: 0.75rem;
                font-weight: 600;
            }

            .status-badge {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                font-size: 0.85rem;
                color: var(--text-muted);
            }

            .status-dot {
                width: 8px;
                height: 8px;
                background-color: var(--accent-green);
                border-radius: 50%;
                box-shadow: 0 0 10px var(--accent-green);
            }

            main {
                max-width: 1400px;
                margin: 2.5rem auto;
                padding: 0 2rem;
                display: grid;
                grid-template-columns: 1.6fr 1fr;
                gap: 2rem;
            }

            @media (max-width: 1024px) {
                main {
                    grid-template-columns: 1fr;
                }
            }

            .card {
                background: var(--card-bg);
                backdrop-filter: blur(16px);
                border: 1px solid var(--card-border);
                border-radius: 18px;
                padding: 2rem;
                margin-bottom: 2rem;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
                transition: transform var(--transition-normal), border-color var(--transition-normal);
            }

            .card:hover {
                border-color: rgba(255, 255, 255, 0.15);
            }

            .card-title {
                font-size: 1.25rem;
                font-weight: 600;
                margin-bottom: 1.5rem;
                display: flex;
                align-items: center;
                gap: 0.75rem;
                color: #ffffff;
            }

            .card-title i {
                color: var(--accent-blue);
            }

            /* --- Drag and drop upload zone --- */
            .upload-zone {
                border: 2px dashed rgba(255, 255, 255, 0.15);
                border-radius: 12px;
                padding: 3rem 1rem;
                text-align: center;
                cursor: pointer;
                transition: all var(--transition-fast);
                background: rgba(255, 255, 255, 0.02);
            }

            .upload-zone:hover, .upload-zone.dragover {
                border-color: var(--accent-blue);
                background: rgba(59, 130, 246, 0.05);
                box-shadow: inset 0 0 20px rgba(59, 130, 246, 0.1);
            }

            .upload-icon {
                font-size: 2.5rem;
                color: var(--accent-blue);
                margin-bottom: 1rem;
            }

            .upload-zone p {
                font-size: 1rem;
                color: var(--text-muted);
                margin-bottom: 0.5rem;
            }

            .upload-zone span {
                font-size: 0.8rem;
                color: rgba(148, 163, 184, 0.7);
            }

            /* --- Selected files list --- */
            .selected-files-container {
                margin-top: 1.5rem;
                max-height: 200px;
                overflow-y: auto;
                border-radius: 8px;
                background: rgba(0, 0, 0, 0.2);
            }

            .file-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0.75rem 1rem;
                border-bottom: 1px solid rgba(255, 255, 255, 0.05);
                font-size: 0.85rem;
            }

            .file-row:last-child {
                border-bottom: none;
            }

            .file-info {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                overflow: hidden;
                white-space: nowrap;
                text-overflow: ellipsis;
            }

            .file-size {
                color: var(--text-muted);
            }

            /* --- Upload button & progress bar --- */
            .btn {
                background: linear-gradient(135deg, var(--accent-blue), #1d4ed8);
                color: white;
                border: none;
                padding: 0.875rem 1.75rem;
                font-family: inherit;
                font-size: 0.95rem;
                font-weight: 600;
                border-radius: 10px;
                cursor: pointer;
                transition: all var(--transition-fast);
                width: 100%;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 0.5rem;
                margin-top: 1.5rem;
                box-shadow: 0 4px 14px var(--accent-blue-glow);
            }

            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px var(--accent-blue-glow);
                filter: brightness(1.1);
            }

            .btn:active {
                transform: translateY(0);
            }

            .btn:disabled {
                background: #1e293b;
                color: var(--text-muted);
                box-shadow: none;
                cursor: not-allowed;
                transform: none;
                filter: none;
            }

            .btn-secondary {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid var(--card-border);
                box-shadow: none;
            }

            .btn-secondary:hover {
                background: rgba(255, 255, 255, 0.1);
                box-shadow: none;
            }

            .btn-success {
                background: linear-gradient(135deg, var(--accent-green), #047857);
                box-shadow: 0 4px 14px var(--accent-green-glow);
            }

            .btn-success:hover {
                box-shadow: 0 6px 20px var(--accent-green-glow);
            }

            .progress-container {
                margin-top: 1.5rem;
                display: none;
            }

            .progress-bar-bg {
                background: rgba(255, 255, 255, 0.08);
                height: 8px;
                border-radius: 9999px;
                overflow: hidden;
                position: relative;
            }

            .progress-bar-fill {
                height: 100%;
                background: linear-gradient(to right, var(--accent-blue), var(--accent-purple));
                width: 0%;
                transition: width 0.1s ease;
                border-radius: 9999px;
            }

            .progress-text {
                display: flex;
                justify-content: space-between;
                font-size: 0.8rem;
                color: var(--text-muted);
                margin-bottom: 0.5rem;
            }

            /* --- Config & Control --- */
            .input-group {
                margin-bottom: 1.25rem;
            }

            .input-group label {
                display: block;
                font-size: 0.85rem;
                color: var(--text-muted);
                margin-bottom: 0.5rem;
                font-weight: 500;
            }

            .input-group input, .input-group select {
                width: 100%;
                background: rgba(0, 0, 0, 0.3);
                border: 1px solid var(--card-border);
                border-radius: 8px;
                padding: 0.75rem 1rem;
                color: white;
                font-family: inherit;
                font-size: 0.9rem;
                transition: border-color var(--transition-fast);
            }

            .input-group input:focus, .input-group select:focus {
                outline: none;
                border-color: var(--accent-blue);
            }

            .config-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1rem;
            }

            /* --- Status Details --- */
            .status-container {
                display: none;
            }

            .status-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1rem;
            }

            .phase-badge {
                font-size: 0.75rem;
                text-transform: uppercase;
                background: rgba(139, 92, 246, 0.15);
                border: 1px solid rgba(139, 92, 246, 0.3);
                color: #c084fc;
                padding: 0.2rem 0.6rem;
                border-radius: 6px;
                font-weight: 700;
            }

            .status-text-val {
                font-weight: 600;
                font-size: 0.9rem;
            }

            .status-text-val.status-running { color: var(--accent-blue); }
            .status-text-val.status-complete { color: var(--accent-green); }
            .status-text-val.status-error { color: var(--accent-red); }
            .status-text-val.status-queued { color: var(--accent-purple); }

            .job-msg {
                background: rgba(0, 0, 0, 0.2);
                border-left: 3px solid var(--accent-blue);
                padding: 0.75rem 1rem;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.85rem;
                border-radius: 0 6px 6px 0;
                margin-top: 1rem;
                color: #e2e8f0;
                word-break: break-all;
            }

            .downloads-container {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 0.75rem;
                margin-top: 1.5rem;
            }

            /* --- Jobs List Registry --- */
            .jobs-list-container {
                max-height: 480px;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
                gap: 0.75rem;
            }

            .job-item {
                background: rgba(255, 255, 255, 0.02);
                border: 1px solid rgba(255, 255, 255, 0.04);
                border-radius: 10px;
                padding: 1rem;
                cursor: pointer;
                transition: all var(--transition-fast);
                position: relative;
            }

            .job-item:hover {
                background: rgba(255, 255, 255, 0.05);
                border-color: rgba(255, 255, 255, 0.1);
            }

            .job-item.active {
                background: rgba(59, 130, 246, 0.08);
                border-color: rgba(59, 130, 246, 0.3);
                box-shadow: 0 0 15px rgba(59, 130, 246, 0.05);
            }

            .job-item-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 0.5rem;
            }

            .job-item-id {
                font-family: 'JetBrains Mono', monospace;
                font-weight: 600;
                font-size: 0.9rem;
            }

            .job-item-info {
                display: flex;
                gap: 1rem;
                font-size: 0.75rem;
                color: var(--text-muted);
            }

            .job-delete-btn {
                position: absolute;
                right: 1rem;
                bottom: 1rem;
                background: transparent;
                border: none;
                cursor: pointer;
                color: rgba(244, 63, 94, 0.5);
                transition: color var(--transition-fast);
                font-size: 1.1rem;
                padding: 0.25rem;
            }

            .job-delete-btn:hover {
                color: var(--accent-red);
            }

            .empty-jobs {
                text-align: center;
                color: var(--text-muted);
                padding: 2rem 0;
                font-size: 0.9rem;
            }

            .no-job-selected {
                text-align: center;
                color: var(--text-muted);
                padding: 2.5rem 0;
                font-size: 0.95rem;
            }

            /* Scrollbar styling */
            ::-webkit-scrollbar {
                width: 6px;
                height: 6px;
            }
            ::-webkit-scrollbar-track {
                background: transparent;
            }
            ::-webkit-scrollbar-thumb {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 9999px;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: rgba(255, 255, 255, 0.2);
            }
        </style>
    </head>
    <body>
        <header>
            <div class="logo-container">
                <span class="logo-icon">🪐</span>
                <div class="logo-text">3D Reconstruction Studio</div>
                <span class="api-badge">API Gateway</span>
            </div>
            <div class="status-badge" id="api-status">
                <span class="status-dot"></span>
                <span>System Online</span>
            </div>
        </header>

        <main>
            <!-- Left Side: Actions and Tracking -->
            <div>
                <!-- Upload Zone Card -->
                <div class="card">
                    <div class="card-title">
                        <span>📤</span> Upload Dataset
                    </div>
                    <div class="upload-zone" id="drop-zone">
                        <div class="upload-icon">📁</div>
                        <p>Drag & drop your files here</p>
                        <p style="font-size: 0.9rem; margin-top: 0.25rem;">Or click to select files from your computer</p>
                        <span style="display:block; margin-top: 0.75rem;">Supports JPG, PNG, and ZIP archives containing images.</span>
                        <input type="file" id="file-input" multiple accept="image/png, image/jpeg, image/jpg, .zip" style="display: none;">
                    </div>

                    <div class="selected-files-container" id="file-list-container" style="display: none;">
                        <div id="file-list"></div>
                    </div>

                    <button class="btn" id="upload-btn" disabled>
                        <span>🚀</span> Upload & Create Reconstruction Job
                    </button>

                    <div class="progress-container" id="upload-progress-container">
                        <div class="progress-text">
                            <span id="upload-status-label">Uploading files...</span>
                            <span id="upload-pct">0%</span>
                        </div>
                        <div class="progress-bar-bg">
                            <div class="progress-bar-fill" id="upload-progress-bar"></div>
                        </div>
                    </div>
                </div>

                <!-- Config & Actions Card -->
                <div class="card" id="config-card">
                    <div class="card-title">
                        <span>⚙️</span> Reconstruction Controller
                    </div>
                    
                    <div class="no-job-selected" id="config-fallback">
                        Select a job or upload images to unlock pipeline controls.
                    </div>

                    <div id="config-controls" style="display: none;">
                        <div style="margin-bottom: 1.5rem; display: flex; gap: 0.5rem; align-items: center;">
                            <span class="text-muted" style="font-size: 0.9rem;">Target Job:</span>
                            <span id="config-target-id" style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: var(--accent-blue);"></span>
                        </div>

                        <!-- SfM Controls -->
                        <div style="border: 1px solid rgba(255,255,255,0.05); padding: 1.25rem; border-radius: 12px; margin-bottom: 1.5rem; background: rgba(0,0,0,0.15);">
                            <h3 style="font-size: 1rem; margin-bottom: 1rem; color: #fff; display: flex; align-items: center; gap: 0.5rem;">
                                <span>🔍</span> Phase 1: Structure-from-Motion (SfM)
                            </h3>
                            <div class="input-group">
                                <label for="focal-length">Focal Length (pixels)</label>
                                <input type="number" id="focal-length" value="800.0" step="10" min="50">
                                <span style="font-size: 0.75rem; color: var(--text-muted); display: block; margin-top: 0.25rem;">Approximation formula: image_width * 1.2</span>
                            </div>
                            <button class="btn btn-success" id="run-sfm-btn">
                                Run SfM Reconstruction
                            </button>
                        </div>

                        <!-- NeRF Controls -->
                        <div style="border: 1px solid rgba(255,255,255,0.05); padding: 1.25rem; border-radius: 12px; background: rgba(0,0,0,0.15);">
                            <h3 style="font-size: 1rem; margin-bottom: 1rem; color: #fff; display: flex; align-items: center; gap: 0.5rem;">
                                <span>⚡</span> Phase 2: Neural Radiance Fields (NeRF)
                            </h3>
                            <div class="input-group">
                                <label for="nerf-scene-dir">Scene Directory Path (On Server)</label>
                                <input type="text" id="nerf-scene-dir" value="data/nerf_synthetic/lego">
                            </div>
                            <div class="config-grid">
                                <div class="input-group">
                                    <label for="nerf-iterations">Iterations</label>
                                    <input type="number" id="nerf-iterations" value="10000" step="1000" min="1000" max="200000">
                                </div>
                                <div class="input-group">
                                    <label for="nerf-scale">Image Downscale Factor</label>
                                    <select id="nerf-scale">
                                        <option value="0.1">0.10 (Fastest)</option>
                                        <option value="0.25" selected>0.25 (Default)</option>
                                        <option value="0.5">0.50 (High Quality)</option>
                                        <option value="1.0">1.00 (Original)</option>
                                    </select>
                                </div>
                            </div>
                            <button class="btn btn-success" id="run-nerf-btn" style="background: linear-gradient(135deg, var(--accent-purple), #6d28d9);">
                                Train NeRF Model
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Right Side: Job details and history -->
            <div>
                <!-- Tracking Card -->
                <div class="card">
                    <div class="card-title">
                        <span>📊</span> Active Job Monitor
                    </div>
                    
                    <div class="no-job-selected" id="monitor-fallback">
                        No job is currently being monitored. Select a job below to see progress.
                    </div>

                    <div class="status-container" id="monitor-container">
                        <div class="status-header">
                            <span id="monitor-job-id" style="font-family: 'JetBrains Mono', monospace; font-weight: 700;"></span>
                            <span class="phase-badge" id="monitor-phase">-</span>
                        </div>

                        <div class="progress-text" style="margin-top: 1rem;">
                            <span>Job Progress</span>
                            <span id="monitor-pct">0%</span>
                        </div>
                        <div class="progress-bar-bg" style="margin-bottom: 1rem;">
                            <div class="progress-bar-fill" id="monitor-progress-bar"></div>
                        </div>

                        <div style="display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 0.75rem;">
                            <span class="text-muted">Status:</span>
                            <span class="status-text-val" id="monitor-status">-</span>
                        </div>

                        <div style="display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 0.75rem;" id="images-count-row">
                            <span class="text-muted">Images In Dataset:</span>
                            <span id="monitor-num-images" style="font-weight: 600;">0</span>
                        </div>

                        <div class="job-msg" id="monitor-message">
                            Ready
                        </div>

                        <div class="downloads-container" id="monitor-downloads" style="display: none;">
                            <a id="dl-cloud-link" class="btn btn-secondary" href="#" style="margin-top: 0; font-size: 0.85rem;" target="_blank">
                                ☁️ Download PLY Cloud
                            </a>
                            <a id="dl-poses-link" class="btn btn-secondary" href="#" style="margin-top: 0; font-size: 0.85rem;" target="_blank">
                                📷 Download Poses
                            </a>
                        </div>
                    </div>
                </div>

                <!-- Jobs Registry -->
                <div class="card">
                    <div class="card-title" style="justify-content: space-between;">
                        <span>📋 Job History</span>
                        <button id="refresh-jobs-btn" style="background: transparent; border: none; cursor: pointer; color: var(--accent-blue); font-size: 0.9rem;">
                            🔄 Refresh
                        </button>
                    </div>

                    <div class="jobs-list-container" id="jobs-list">
                        <!-- Loaded dynamically -->
                    </div>
                </div>
            </div>
        </main>

        <script>
            // App state
            let selectedFiles = [];
            let activeJobId = null;
            let pollingInterval = null;

            // UI Elements
            const dropZone = document.getElementById('drop-zone');
            const fileInput = document.getElementById('file-input');
            const fileListContainer = document.getElementById('file-list-container');
            const fileList = document.getElementById('file-list');
            const uploadBtn = document.getElementById('upload-btn');
            const uploadProgressContainer = document.getElementById('upload-progress-container');
            const uploadProgressBar = document.getElementById('upload-progress-bar');
            const uploadStatusLabel = document.getElementById('upload-status-label');
            const uploadPct = document.getElementById('upload-pct');
            
            const configCard = document.getElementById('config-card');
            const configFallback = document.getElementById('config-fallback');
            const configControls = document.getElementById('config-controls');
            const configTargetId = document.getElementById('config-target-id');
            const focalLengthInput = document.getElementById('focal-length');
            const runSfmBtn = document.getElementById('run-sfm-btn');
            const nerfSceneDirInput = document.getElementById('nerf-scene-dir');
            const nerfIterationsInput = document.getElementById('nerf-iterations');
            const nerfScaleSelect = document.getElementById('nerf-scale');
            const runNerfBtn = document.getElementById('run-nerf-btn');

            const monitorFallback = document.getElementById('monitor-fallback');
            const monitorContainer = document.getElementById('monitor-container');
            const monitorJobId = document.getElementById('monitor-job-id');
            const monitorPhase = document.getElementById('monitor-phase');
            const monitorPct = document.getElementById('monitor-pct');
            const monitorProgressBar = document.getElementById('monitor-progress-bar');
            const monitorStatus = document.getElementById('monitor-status');
            const monitorNumImages = document.getElementById('monitor-num-images');
            const monitorMessage = document.getElementById('monitor-message');
            const monitorDownloads = document.getElementById('monitor-downloads');
            const dlCloudLink = document.getElementById('dl-cloud-link');
            const dlPosesLink = document.getElementById('dl-poses-link');

            const jobsList = document.getElementById('jobs-list');
            const refreshJobsBtn = document.getElementById('refresh-jobs-btn');

            // --- Drag & Drop Event Listeners ---
            ['dragenter', 'dragover'].forEach(eventName => {
                dropZone.addEventListener(eventName, (e) => {
                    e.preventDefault();
                    dropZone.classList.add('dragover');
                }, false);
            });

            ['dragleave', 'drop'].forEach(eventName => {
                dropZone.addEventListener(eventName, (e) => {
                    e.preventDefault();
                    dropZone.classList.remove('dragover');
                }, false);
            });

            dropZone.addEventListener('drop', (e) => {
                const dt = e.dataTransfer;
                const files = dt.files;
                handleFiles(files);
            });

            dropZone.addEventListener('click', () => {
                fileInput.click();
            });

            fileInput.addEventListener('change', (e) => {
                handleFiles(e.target.files);
            });

            function handleFiles(files) {
                selectedFiles = Array.from(files);
                if (selectedFiles.length > 0) {
                    fileList.innerHTML = '';
                    selectedFiles.forEach((file) => {
                        const row = document.createElement('div');
                        row.className = 'file-row';
                        
                        const info = document.createElement('div');
                        info.className = 'file-info';
                        
                        const isZip = file.name.endsWith('.zip');
                        info.innerHTML = `<span>${isZip ? '📦' : '📷'}</span> <span>${file.name}</span>`;
                        
                        const size = document.createElement('span');
                        size.className = 'file-size';
                        size.textContent = formatBytes(file.size);
                        
                        row.appendChild(info);
                        row.appendChild(size);
                        fileList.appendChild(row);
                    });
                    fileListContainer.style.display = 'block';
                    uploadBtn.disabled = false;
                } else {
                    fileListContainer.style.display = 'none';
                    uploadBtn.disabled = true;
                }
            }

            function formatBytes(bytes, decimals = 2) {
                if (bytes === 0) return '0 Bytes';
                const k = 1024;
                const dm = decimals < 0 ? 0 : decimals;
                const sizes = ['Bytes', 'KB', 'MB', 'GB'];
                const i = Math.floor(Math.log(bytes) / Math.log(k));
                return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
            }

            // --- Upload Logic ---
            uploadBtn.addEventListener('click', () => {
                if (selectedFiles.length === 0) return;

                const formData = new FormData();
                selectedFiles.forEach((file) => {
                    formData.append('files', file);
                });

                uploadBtn.disabled = true;
                uploadProgressContainer.style.display = 'block';
                uploadProgressBar.style.width = '0%';
                uploadPct.textContent = '0%';
                uploadStatusLabel.textContent = 'Uploading files to server...';

                const xhr = new XMLHttpRequest();
                xhr.open('POST', '/api/upload-images', true);

                xhr.upload.addEventListener('progress', (e) => {
                    if (e.lengthComputable) {
                        const percentage = Math.round((e.loaded * 100) / e.total);
                        uploadProgressBar.style.width = percentage + '%';
                        uploadPct.textContent = percentage + '%';
                        uploadStatusLabel.textContent = `Uploading: ${formatBytes(e.loaded)} / ${formatBytes(e.total)}`;
                    }
                });

                xhr.onload = function () {
                    if (xhr.status >= 200 && xhr.status < 300) {
                        const response = JSON.parse(xhr.responseText);
                        uploadStatusLabel.textContent = 'Upload complete!';
                        uploadProgressBar.style.width = '100%';
                        uploadPct.textContent = '100%';
                        
                        // Set active job and transition
                        setTimeout(() => {
                            uploadProgressContainer.style.display = 'none';
                            selectedFiles = [];
                            fileListContainer.style.display = 'none';
                            fileInput.value = '';
                            
                            selectJob(response.job_id);
                            fetchJobsList();
                        }, 1000);
                    } else {
                        let errMsg = 'Upload failed';
                        try {
                            const resObj = JSON.parse(xhr.responseText);
                            if (resObj.detail) errMsg += ': ' + resObj.detail;
                        } catch(e) {}
                        uploadStatusLabel.textContent = errMsg;
                        uploadProgressBar.style.backgroundColor = 'var(--accent-red)';
                        uploadBtn.disabled = false;
                    }
                };

                xhr.onerror = function () {
                    uploadStatusLabel.textContent = 'Network error during upload.';
                    uploadBtn.disabled = false;
                };

                xhr.send(formData);
            });

            // --- Job Selection and Polling ---
            function selectJob(jobId) {
                activeJobId = jobId;
                
                // Show controls
                configFallback.style.display = 'none';
                configControls.style.display = 'block';
                configTargetId.textContent = jobId;

                // Show monitor
                monitorFallback.style.display = 'none';
                monitorContainer.style.display = 'block';
                monitorJobId.textContent = `Job: ${jobId}`;

                // Update registry active state
                document.querySelectorAll('.job-item').forEach(el => {
                    if (el.dataset.id === jobId) {
                        el.classList.add('active');
                    } else {
                        el.classList.remove('active');
                    }
                });

                // Clear previous poll
                if (pollingInterval) clearInterval(pollingInterval);
                
                // Fetch immediately and poll
                pollStatus();
                pollingInterval = setInterval(pollStatus, 2000);
            }

            function pollStatus() {
                if (!activeJobId) return;

                fetch(`/api/status/${activeJobId}`)
                    .then(response => {
                        if (!response.ok) throw new Error('Job not found');
                        return response.json();
                    })
                    .then(data => {
                        // Update Monitor UI
                        monitorPhase.textContent = data.phase || 'upload';
                        
                        const progress = data.progress !== undefined ? data.progress : 0;
                        monitorPct.textContent = progress + '%';
                        monitorProgressBar.style.width = progress + '%';

                        const statusVal = data.status || 'uploaded';
                        monitorStatus.textContent = statusVal.toUpperCase();
                        monitorStatus.className = 'status-text-val status-' + statusVal;

                        monitorNumImages.textContent = data.num_images || 0;
                        monitorMessage.textContent = data.message || 'Waiting to start...';

                        // Show downloads if completed SfM
                        if ((data.phase === 'sfm' || data.phase === 'nerf') && (data.status === 'complete' || (data.phase === 'nerf' && data.status === 'running'))) {
                            monitorDownloads.style.display = 'grid';
                            dlCloudLink.href = `/api/download/${activeJobId}/point_cloud.ply`;
                            dlPosesLink.href = `/api/download/${activeJobId}/camera_poses.json`;
                        } else {
                            monitorDownloads.style.display = 'none';
                        }

                        // Stop polling if complete or error
                        if (statusVal === 'complete' || statusVal === 'error') {
                            clearInterval(pollingInterval);
                            pollingInterval = null;
                            fetchJobsList(); // Refresh list to catch up-to-date statuses
                        }
                    })
                    .catch(err => {
                        monitorStatus.textContent = 'DISCONNECTED';
                        monitorStatus.className = 'status-text-val status-error';
                        monitorMessage.textContent = 'Unable to poll job status: ' + err.message;
                        clearInterval(pollingInterval);
                        pollingInterval = null;
                    });
            }

            // --- Control Actions ---
            runSfmBtn.addEventListener('click', () => {
                if (!activeJobId) return;
                const focal = focalLengthInput.value || 800.0;
                
                runSfmBtn.disabled = true;
                
                fetch(`/api/run-sfm/${activeJobId}?focal_length=${focal}`, { method: 'POST' })
                    .then(res => res.json())
                    .then(data => {
                        runSfmBtn.disabled = false;
                        selectJob(activeJobId); // Restart polling
                    })
                    .catch(err => {
                        alert('Error initiating SfM: ' + err.message);
                        runSfmBtn.disabled = false;
                    });
            });

            runNerfBtn.addEventListener('click', () => {
                if (!activeJobId) return;
                const sceneDir = nerfSceneDirInput.value || 'data/nerf_synthetic/lego';
                const iters = nerfIterationsInput.value || 10000;
                const scale = nerfScaleSelect.value || 0.25;

                runNerfBtn.disabled = true;

                fetch(`/api/run-nerf/${activeJobId}?scene_dir=${encodeURIComponent(sceneDir)}&num_iterations=${iters}&img_scale=${scale}`, { method: 'POST' })
                    .then(res => {
                        if (!res.ok) return res.json().then(e => { throw new Error(e.detail || 'Failed to start') });
                        return res.json();
                    })
                    .then(data => {
                        runNerfBtn.disabled = false;
                        selectJob(activeJobId); // Restart polling
                    })
                    .catch(err => {
                        alert('Error initiating NeRF training: ' + err.message);
                        runNerfBtn.disabled = false;
                    });
            });

            // --- Job History Registry list ---
            function fetchJobsList() {
                fetch('/api/jobs')
                    .then(res => res.json())
                    .then(data => {
                        const jobs = data.jobs || [];
                        jobsList.innerHTML = '';
                        
                        if (jobs.length === 0) {
                            jobsList.innerHTML = '<div class="empty-jobs">No jobs created yet.</div>';
                            return;
                        }

                        // Sort jobs by updated_at descending
                        jobs.sort((a, b) => (b.updated_at || 0) - (a.updated_at || 0));

                        jobs.forEach(job => {
                            const item = document.createElement('div');
                            item.className = 'job-item' + (activeJobId === job.job_id ? ' active' : '');
                            item.dataset.id = job.job_id;
                            
                            const updateTime = job.updated_at ? new Date(job.updated_at * 1000).toLocaleTimeString() : 'Unknown';
                            
                            item.innerHTML = `
                                <div class="job-item-header">
                                    <span class="job-item-id">${job.job_id}</span>
                                    <span class="status-text-val status-${job.status || 'uploaded'}" style="font-size:0.8rem;">
                                        ${(job.status || 'uploaded').toUpperCase()}
                                    </span>
                                </div>
                                <div class="job-item-info">
                                    <span>Phase: <b>${job.phase || 'upload'}</b></span>
                                    <span>Images: <b>${job.num_images || 0}</b></span>
                                    <span>Updated: <b>${updateTime}</b></span>
                                </div>
                                <button class="job-delete-btn" title="Delete job">🗑️</button>
                            `;

                            // Select click
                            item.addEventListener('click', (e) => {
                                if (e.target.classList.contains('job-delete-btn')) {
                                    e.stopPropagation();
                                    deleteJob(job.job_id);
                                    return;
                                }
                                selectJob(job.job_id);
                            });

                            jobsList.appendChild(item);
                        });
                    })
                    .catch(err => {
                        jobsList.innerHTML = `<div class="empty-jobs" style="color:var(--accent-red)">Error loading jobs: ${err.message}</div>`;
                    });
            }

            function deleteJob(jobId) {
                if (!confirm(`Are you sure you want to delete job ${jobId}?`)) return;

                fetch(`/api/jobs/${jobId}`, { method: 'DELETE' })
                    .then(res => {
                        if (!res.ok) return res.json().then(e => { throw new Error(e.detail || 'Delete failed') });
                        return res.json();
                    })
                    .then(() => {
                        if (activeJobId === jobId) {
                            activeJobId = null;
                            if (pollingInterval) clearInterval(pollingInterval);
                            
                            // Hide panels
                            configControls.style.display = 'none';
                            configFallback.style.display = 'block';
                            monitorContainer.style.display = 'none';
                            monitorFallback.style.display = 'block';
                        }
                        fetchJobsList();
                    })
                    .catch(err => {
                        alert('Error deleting job: ' + err.message);
                    });
            }

            // --- Event Bindings ---
            refreshJobsBtn.addEventListener('click', fetchJobsList);

            // Initial load
            fetchJobsList();
        </script>
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