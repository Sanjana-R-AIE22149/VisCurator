#!/usr/bin/env python3
"""
viscurator_launch.py
====================
Drop this in your VisCurator root folder and run:

    python viscurator_launch.py

It will:
  1. Check Python version and system tools
  2. Validate backend/.env (API key, no spaces, no placeholders)
  3. Check every required Python package — install missing ones
  4. Check node_modules — run npm install if missing
  5. Create all required directories and the SQLite DB
  6. Start the FastAPI backend and wait until it's healthy
  7. Start the Vite frontend and wait until it's healthy
  8. Print the URL and open the browser
  9. Watch both processes — restart backend if it crashes
 10. Clean shutdown on Ctrl-C

No arguments needed.
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

# ── Terminal colours (Windows-safe) ──────────────────────────────────────────

IS_WIN = platform.system() == "Windows"

if IS_WIN:
    # Enable ANSI on Windows 10+
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleMode(
            ctypes.windll.kernel32.GetStdHandle(-11), 7
        )
    except Exception:
        pass

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m"

def G(t: str) -> str: return _c("32;1", t)   # green bold
def Y(t: str) -> str: return _c("33;1", t)   # yellow bold
def R(t: str) -> str: return _c("31;1", t)   # red bold
def C(t: str) -> str: return _c("36;1", t)   # cyan bold
def D(t: str) -> str: return _c("2", t)      # dim
def B(t: str) -> str: return _c("1", t)      # bold


# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT         = Path(__file__).resolve().parent
BACKEND_DIR  = ROOT / "backend"
ENV_FILE     = BACKEND_DIR / ".env"
ENV_EXAMPLE  = BACKEND_DIR / ".env.example"
REQ_FILE     = BACKEND_DIR / "requirements.txt"
DATA_DIR     = BACKEND_DIR / "data"
SQLITE_PATH  = DATA_DIR / "training_metrics.db"
NODE_MODULES = ROOT / "node_modules"
RUNS_DIR     = ROOT / "runs"
OUTPUT_DIR   = ROOT / "cvagent_output"

BACKEND_URL  = "http://127.0.0.1:8000"
FRONTEND_URL = "http://127.0.0.1:5173"
HEALTH_URL   = f"{BACKEND_URL}/api/health"

# Required Python packages: (import_name, pip_name)
REQUIRED_PACKAGES = [
    ("fastapi",        "fastapi>=0.110.0"),
    ("uvicorn",        "uvicorn[standard]>=0.29.0"),
    ("dotenv",         "python-dotenv>=1.0.0"),
    ("openai",         "openai>=1.30.0"),
    ("aiohttp",        "aiohttp>=3.9.0"),
    ("PIL",            "Pillow>=10.3.0"),
    ("imagehash",      "imagehash>=4.3.1"),
    ("numpy",          "numpy>=1.26.0"),
    ("cv2",            "opencv-python-headless>=4.9.0"),
    ("torch",          "torch>=2.3.0"),
    ("albumentations", "albumentations>=1.4.0"),
    ("psutil",         "psutil>=5.9.0"),
    ("jose",           "python-jose[cryptography]>=3.3.0"),
    ("passlib",        "passlib[bcrypt]>=1.7.4"),
    ("datasets",       "datasets>=2.19.0"),
    ("multipart",      "python-multipart>=0.0.9"),
]


# ── Utilities ─────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    width = 56
    print(f"\n{C('─' * width)}")
    print(f"  {B(title)}")
    print(f"{C('─' * width)}")


def _ok(msg: str) -> None:
    print(f"  {G('✓')} {msg}")


def _warn(msg: str) -> None:
    print(f"  {Y('⚠')} {msg}")


def _fail(msg: str) -> None:
    print(f"\n  {R('✗')} {msg}\n")
    sys.exit(1)


def _info(msg: str) -> None:
    print(f"  {D('·')} {msg}")


def _wait_http(url: str, timeout: float, proc: subprocess.Popen | None = None) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if 200 <= r.status < 500:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


# ── Step 1 — Python & system checks ──────────────────────────────────────────

def check_python() -> None:
    _section("Step 1 · System checks")
    if sys.version_info < (3, 10):
        _fail(f"Python 3.10+ required. You have {sys.version.split()[0]}.\nDownload: https://python.org/downloads")
    _ok(f"Python {sys.version.split()[0]}")

    if not shutil.which("node"):
        _fail("Node.js not found.\nInstall from: https://nodejs.org  (LTS version, 18+)")
    node_ver = subprocess.check_output(["node", "--version"], text=True).strip()
    _ok(f"Node.js {node_ver}")

    npm = "npm.cmd" if IS_WIN else "npm"
    if not shutil.which(npm) and not shutil.which("npm"):
        _fail("npm not found. It ships with Node.js — reinstall Node.")
    _ok("npm found")


# ── Step 2 — .env validation ──────────────────────────────────────────────────

def check_env() -> None:
    _section("Step 2 · Environment (.env)")

    if not ENV_FILE.exists():
        if ENV_EXAMPLE.exists():
            shutil.copy(ENV_EXAMPLE, ENV_FILE)
            _warn("Created backend/.env from .env.example")
        else:
            ENV_FILE.write_text(
                "NVIDIA_API_KEY=your_key_here\n"
                "HF_TOKEN=\n"
                "HOST=0.0.0.0\n"
                "PORT=8000\n"
                "LOG_LEVEL=info\n"
            )
            _warn("Created blank backend/.env")

    # Parse .env manually so we can validate before loading dotenv
    env_vars: dict[str, str] = {}
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env_vars[k.strip()] = v.strip()

    api_key = env_vars.get("NVIDIA_API_KEY", "")

    # Check for leading/trailing spaces (the most common mistake)
    raw_line = next(
        (l for l in ENV_FILE.read_text().splitlines() if l.startswith("NVIDIA_API_KEY")),
        "",
    )
    if "= " in raw_line or raw_line.endswith(" "):
        _fail(
            "NVIDIA_API_KEY has a space after '=' in backend/.env\n"
            "  Fix: NVIDIA_API_KEY=nvapi-xxx  (no space)\n"
            f"  Current line: {raw_line!r}"
        )

    if not api_key or api_key in ("your_key_here", ""):
        _warn(
            "NVIDIA_API_KEY is not set in backend/.env\n"
            "    The app will start but the CVAgent will not work.\n"
            f"    Get a free key at {C('https://build.nvidia.com')}"
        )
    elif not api_key.startswith("nvapi-"):
        _warn(f"NVIDIA_API_KEY doesn't look right (should start with 'nvapi-'): {api_key[:12]}…")
    else:
        _ok(f"NVIDIA_API_KEY set ({api_key[:8]}…)")

    hf = env_vars.get("HF_TOKEN", "")
    if hf and hf != "optional_huggingface_token":
        _ok("HF_TOKEN set (gated datasets enabled)")
    else:
        _info("HF_TOKEN not set — only public HuggingFace datasets available")


# ── Step 3 — Python packages ──────────────────────────────────────────────────

def check_packages() -> None:
    _section("Step 3 · Python packages")

    missing_pip: list[str] = []
    for import_name, pip_spec in REQUIRED_PACKAGES:
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            print(f"  {R('✗')} {import_name}  {D('(missing)')}")
            missing_pip.append(pip_spec)
        else:
            _ok(import_name)

    if missing_pip:
        print(f"\n  {Y('Installing')} {len(missing_pip)} missing package(s)…")
        cmd = [sys.executable, "-m", "pip", "install", "--quiet"] + missing_pip
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            _fail(
                "pip install failed.\n"
                "Try manually:\n"
                f"  pip install -r {REQ_FILE}"
            )
        _ok(f"Installed: {', '.join(p.split('>=')[0].split('[')[0] for p in missing_pip)}")


# ── Step 4 — Node modules ─────────────────────────────────────────────────────

def check_node_modules() -> None:
    _section("Step 4 · Node modules")

    if not (ROOT / "package.json").exists():
        _fail(f"package.json not found in {ROOT}\nRun this script from the VisCurator root folder.")

    if not NODE_MODULES.exists():
        _warn("node_modules not found — running npm install (this may take a minute)…")
        npm = "npm.cmd" if IS_WIN else "npm"
        rc = subprocess.run([npm, "install"], cwd=str(ROOT)).returncode
        if rc != 0:
            _fail("npm install failed. Check your Node.js version (18+ required).")
        _ok("npm install complete")
    else:
        _ok("node_modules present")


# ── Step 5 — Directories & SQLite ────────────────────────────────────────────

def setup_dirs() -> None:
    _section("Step 5 · Directories & database")

    for d in [DATA_DIR, RUNS_DIR, OUTPUT_DIR, OUTPUT_DIR / "processed"]:
        d.mkdir(parents=True, exist_ok=True)
    _ok("Output directories ready")

    with sqlite3.connect(SQLITE_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id    TEXT,
                epoch     INTEGER,
                loss      REAL,
                accuracy  REAL,
                precision REAL,
                recall    REAL,
                map       REAL,
                timestamp TEXT
            )
        """)
        conn.commit()
    _ok(f"SQLite ready at {SQLITE_PATH.relative_to(ROOT)}")


# ── Step 6 — Start backend ────────────────────────────────────────────────────

def start_backend() -> subprocess.Popen:
    _section("Step 6 · Backend (FastAPI + uvicorn)")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"

    # Load .env into the subprocess env so uvicorn sees all vars
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip()

    proc = subprocess.Popen(
        [
                    sys.executable, "-m", "uvicorn",
                    "backend.main:app",
                    "--loop", "asyncio",
            "--host", "127.0.0.1",
            "--port", "8000",
            "--log-level", "warning",   # quieter — errors still show
        ],
        cwd=str(ROOT),
        env=env,
    )

    _info(f"Waiting for backend to be healthy (PID {proc.pid})…")
    if not _wait_http(HEALTH_URL, timeout=30, proc=proc):
        if proc.poll() is not None:
            _fail(
                "Backend process died during startup.\n"
                "Common causes:\n"
                "  • Missing Python package (re-run this script)\n"
                "  • Syntax error in backend code\n"
                "  • Port 8000 already in use\n"
                "Run manually to see the full traceback:\n"
                f"  python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"
            )
        _fail(
            "Backend didn't respond within 30 seconds.\n"
            "Check if port 8000 is blocked or another process is using it."
        )

    # Read the health response to show NIM status
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=3) as r:
            health = json.loads(r.read().decode())
        nim_ok = health.get("nim_connected", False)
        pkgs_ok = all(health.get("python_packages", {}).values())

        _ok(f"Backend healthy at {BACKEND_URL}")
        if nim_ok:
            _ok("NIM connected — CVAgent is fully operational")
        else:
            _warn("NIM not connected — set NVIDIA_API_KEY in backend/.env")

        if not pkgs_ok:
            bad = [k for k, v in health.get("python_packages", {}).items() if not v]
            _warn(f"Missing packages detected by backend: {', '.join(bad)}")
    except Exception:
        _ok(f"Backend healthy at {BACKEND_URL}")

    return proc


# ── Step 7 — Start frontend ───────────────────────────────────────────────────

def start_frontend() -> subprocess.Popen:
    _section("Step 7 · Frontend (Vite)")

    npm = "npm.cmd" if IS_WIN and shutil.which("npm.cmd") else "npm"
    proc = subprocess.Popen(
        [npm, "run", "dev", "--", "--host", "127.0.0.1", "--port", "5173"],
        cwd=str(ROOT),
    )

    _info(f"Waiting for Vite dev server (PID {proc.pid})…")
    if not _wait_http(FRONTEND_URL, timeout=120, proc=proc):
        if proc.poll() is not None:
            _fail(
                "Frontend process died during startup.\n"
                "Common causes:\n"
                "  • node_modules corrupted — delete it and re-run\n"
                "  • Port 5173 already in use\n"
                "Run manually to see errors:\n"
                "  npm run dev"
            )
        _fail("Frontend didn't respond within 30 seconds.")

    _ok(f"Frontend ready at {FRONTEND_URL}")
    return proc


# ── Step 8 — Print summary & open browser ────────────────────────────────────

def print_ready(nim_connected: bool) -> None:
    print(f"""
{C('╔══════════════════════════════════════════════════════╗')}
{C('║')}  {G('VisCurator is running!')}                              {C('║')}
{C('╠══════════════════════════════════════════════════════╣')}
{C('║')}                                                      {C('║')}
{C('║')}   {B('App')}        →  {C('http://localhost:5173')}             {C('║')}
{C('║')}   {B('API')}        →  {C('http://localhost:8000')}             {C('║')}
{C('║')}   {B('API docs')}   →  {C('http://localhost:8000/docs')}        {C('║')}
{C('║')}   {B('Health')}     →  {C('http://localhost:8000/api/health')}  {C('║')}
{C('║')}                                                      {C('║')}
{C('║')}   {B('Login')}      →  admin / viscurator                {C('║')}
{C('║')}   {B('NIM')}        →  {"" + G("connected") + "   CVAgent ready           " if nim_connected else Y("not connected") + "  Set NVIDIA_API_KEY      "}{C('║')}
{C('║')}                                                      {C('║')}
{C('║')}   {D('Press Ctrl-C to stop all servers')}                 {C('║')}
{C('╚══════════════════════════════════════════════════════╝')}""")


# ── Step 9 — Watch loop with auto-restart ────────────────────────────────────

def watch(backend: subprocess.Popen, frontend: subprocess.Popen) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip()

    backend_restarts = 0

    def _shutdown(sig=None, frame=None) -> None:
        print(f"\n  {Y('Shutting down…')}")
        for p in (frontend, backend):
            try:
                if IS_WIN:
                    p.terminate()
                else:
                    p.terminate()
            except Exception:
                pass
        for p in (frontend, backend):
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        print(f"  {G('All servers stopped. Goodbye!')}\n")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    if not IS_WIN:
        signal.signal(signal.SIGTERM, _shutdown)

    while True:
        time.sleep(2)

        # Frontend crash — fatal, user needs to see the error
        if frontend.poll() is not None:
            print(f"\n  {R('Frontend process exited unexpectedly.')}")
            _shutdown()

        # Backend crash — try to restart once, then give up
        if backend.poll() is not None:
            backend_restarts += 1
            if backend_restarts > 3:
                print(f"\n  {R('Backend crashed 3 times. Giving up.')}")
                _shutdown()

            print(f"\n  {Y(f'Backend crashed (attempt {backend_restarts}/3). Restarting in 3s…')}")
            time.sleep(3)

            backend = subprocess.Popen(
                [
                    sys.executable, "-m", "uvicorn",
                    "backend.main:app",
                    "--loop", "asyncio",
                    "--host", "127.0.0.1",
                    "--port", "8000",
                    "--log-level", "warning",
                ],
                cwd=str(ROOT),
                env=env,
            )

            if _wait_http(HEALTH_URL, timeout=20, proc=backend):
                print(f"  {G('Backend restarted successfully.')}")
            else:
                print(f"  {R('Backend failed to restart.')}")
                _shutdown()


# ── Banner ────────────────────────────────────────────────────────────────────

def print_banner() -> None:
    print(f"""
{C('  ╦  ╦╦╔═╗╔═╗╦ ╦╦═╗╔═╗╔╦╗╔═╗╦═╗')}
{C('  ╚╗╔╝║╚═╗║  ║ ║╠╦╝╠═╣ ║ ║ ║╠╦╝')}
{C('   ╚╝ ╩╚═╝╚═╝╚═╝╩╚═╩ ╩ ╩ ╚═╝╩╚═')}
  {D('AI Computer Vision Platform · v0.2')}
""")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print_banner()

    check_python()
    check_env()
    check_packages()
    check_node_modules()
    setup_dirs()

    backend  = start_backend()
    frontend = start_frontend()

    # Check NIM status for the summary banner
    nim_ok = False
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=3) as r:
            nim_ok = json.loads(r.read().decode()).get("nim_connected", False)
    except Exception:
        pass

    print_ready(nim_ok)

    # Open browser after a short delay
    try:
        time.sleep(1.5)
        webbrowser.open(FRONTEND_URL)
    except Exception:
        pass

    watch(backend, frontend)


if __name__ == "__main__":
    main()