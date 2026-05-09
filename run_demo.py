from __future__ import annotations

import importlib.util
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
DATA_DIR = BACKEND_DIR / "data"
SQLITE_PATH = DATA_DIR / "training_metrics.db"
RUNS_DIR = ROOT / "runs"
PROCESSED_DIR = ROOT / "processed"
CV_OUTPUT_PROCESSED_DIR = ROOT / "cvagent_output" / "processed"
IS_WIN = os.name == "nt"


def ok(message: str) -> None:
    print(f"[OK] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def fail(message: str) -> None:
    print(f"[ERROR] {message}")
    raise SystemExit(1)


def ensure_prereqs() -> None:
    if sys.version_info < (3, 10):
        fail(f"Python 3.10+ required. Found {sys.version.split()[0]}")
    if shutil.which("node") is None:
        fail("Node.js was not found on PATH.")
    if shutil.which("npm") is None and shutil.which("npm.cmd") is None:
        fail("npm was not found on PATH.")
    if not (ROOT / "package.json").exists():
        fail("package.json not found. Run this script from the VisCurator repo root.")
    if not (ROOT / "node_modules").exists():
        fail("node_modules is missing. Run `npm install` first.")
    if not (BACKEND_DIR / "requirements.txt").exists():
        fail("backend/requirements.txt is missing.")
    if importlib.util.find_spec("uvicorn") is None:
        fail("Python package `uvicorn` is missing. Install backend dependencies first.")
    if importlib.util.find_spec("fastapi") is None:
        fail("Python package `fastapi` is missing. Install backend dependencies first.")


def ensure_paths() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    CV_OUTPUT_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ok(f"Directories ready: {RUNS_DIR.name}/, {PROCESSED_DIR.name}/")


def ensure_sqlite() -> None:
    with sqlite3.connect(SQLITE_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT,
                epoch INTEGER,
                loss REAL,
                accuracy REAL,
                precision REAL,
                recall REAL,
                map REAL,
                timestamp TEXT
            )
            """
        )
        conn.commit()
    ok(f"SQLite initialized at {SQLITE_PATH}")


def wait_for_http(url: str, timeout_s: float, proc: subprocess.Popen | None = None) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if 200 <= response.status < 500:
                    return True
        except urllib.error.URLError:
            time.sleep(0.5)
        except Exception:
            time.sleep(0.5)
    return False


def start_backend() -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        cwd=str(ROOT),
        env=env,
    )
    if wait_for_http("http://127.0.0.1:8000/api/health", 20, proc):
        ok("Backend started")
        return proc

    if proc.poll() is not None:
        fail("Backend exited during startup. Check Python dependencies, FastAPI imports, or backend/.env.")
    fail("Backend did not become healthy within 20 seconds.")


def start_frontend() -> subprocess.Popen:
    npm_cmd = "npm.cmd" if IS_WIN and shutil.which("npm.cmd") else "npm"
    proc = subprocess.Popen(
        [npm_cmd, "run", "dev", "--", "--host", "127.0.0.1", "--port", "5173"],
        cwd=str(ROOT),
    )
    if wait_for_http("http://127.0.0.1:5173", 25, proc):
        ok("Frontend started")
        return proc

    if proc.poll() is not None:
        fail("Frontend exited during startup. Check Node/Vite dependencies and the local npm install.")
    fail("Frontend did not become ready within 25 seconds.")


def print_summary() -> None:
    ok("WebSocket endpoints ready")
    print("[INFO] Backend:  http://127.0.0.1:8000")
    print("[INFO] Frontend: http://127.0.0.1:5173")
    print("[INFO] Press Ctrl+C to stop both servers.")


def stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
    except Exception:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def main() -> None:
    print("VisCurator demo launcher")
    ensure_prereqs()
    ensure_paths()
    ensure_sqlite()

    backend = start_backend()
    frontend = start_frontend()
    print_summary()

    try:
        while True:
            if backend.poll() is not None:
                fail("Backend stopped unexpectedly during demo run.")
            if frontend.poll() is not None:
                fail("Frontend stopped unexpectedly during demo run.")
            time.sleep(1)
    except KeyboardInterrupt:
        warn("Shutting down demo services...")
    finally:
        stop_process(frontend)
        stop_process(backend)
        ok("Demo services stopped")


if __name__ == "__main__":
    main()
