"""
VisCurator — Dev Launcher
=========================
Starts both the FastAPI backend and the Vite frontend in one command.

Usage:
    python start.py               # start both servers
    python start.py --backend     # backend only
    python start.py --frontend    # frontend only
    python start.py --setup       # install all dependencies first, then start

Requirements: Python 3.10+, Node.js 18+, npm
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# ── Force UTF-8 on Windows for Unicode Banner ────────────────

if platform.system() == "Windows":
    # On Windows, the default console encoding (cp1252) often fails 
    # to encode the box-drawing characters in the banner.
    sys.stdout.reconfigure(encoding='utf-8')

# ── Colours (Windows-safe) ───────────────────────────────────

IS_WIN = platform.system() == "Windows"

def _ansi(code: str, text: str) -> str:
    if IS_WIN:
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleMode(  # type: ignore[attr-defined]
                ctypes.windll.kernel32.GetStdHandle(-11), 7
            )
        except Exception:
            return text
    return f"\033[{code}m{text}\033[0m"

def green(t: str)  -> str: return _ansi("32;1", t)
def yellow(t: str) -> str: return _ansi("33;1", t)
def cyan(t: str)   -> str: return _ansi("36;1", t)
def red(t: str)    -> str: return _ansi("31;1", t)
def dim(t: str)    -> str: return _ansi("2", t)
def bold(t: str)   -> str: return _ansi("1", t)

# ── Paths ────────────────────────────────────────────────────

ROOT = Path(__file__).parent.resolve()
BACKEND_DIR  = ROOT / "backend"
ENV_FILE     = BACKEND_DIR / ".env"
ENV_EXAMPLE  = BACKEND_DIR / ".env.example"
REQ_FILE     = BACKEND_DIR / "requirements.txt"
PKG_JSON     = ROOT / "package.json"
NODE_MODULES = ROOT / "node_modules"

# ── Banner ────────────────────────────────────────────────────

BANNER = f"""
{cyan('╔══════════════════════════════════════════════════╗')}
{cyan('║')}  {bold('VisCurator')}  ·  AI Computer Vision Platform       {cyan('║')}
{cyan('║')}  {dim('Autonomous dataset curation & model building')}   {cyan('║')}
{cyan('╚══════════════════════════════════════════════════╝')}
"""

# ── Helpers ──────────────────────────────────────────────────

def _run(cmd: list[str] | str, **kwargs) -> int:
    """Run a command, inherit stdio, return exit code."""
    shell = isinstance(cmd, str) or IS_WIN
    result = subprocess.run(cmd, shell=shell, **kwargs)
    return result.returncode


def _check_tool(name: str, version_flag: str = "--version") -> bool:
    """Return True if *name* is on PATH."""
    return shutil.which(name) is not None


def _pip_install() -> None:
    print(f"\n{yellow('▶')} Installing Python dependencies …")
    rc = _run([sys.executable, "-m", "pip", "install", "-r", str(REQ_FILE), "-q"])
    if rc != 0:
        print(red("  ✗ pip install failed. Check requirements.txt and your Python env."))
        sys.exit(1)
    print(green("  ✓ Python dependencies installed."))


def _npm_install() -> None:
    print(f"\n{yellow('▶')} Installing Node dependencies …")
    rc = _run("npm install", cwd=str(ROOT))
    if rc != 0:
        print(red("  ✗ npm install failed. Ensure Node.js 18+ and npm are installed."))
        sys.exit(1)
    print(green("  ✓ Node dependencies installed."))


def _ensure_env() -> bool:
    """Create .env from .env.example if it doesn't exist.
    Returns True if the key looks configured."""
    if not ENV_FILE.exists():
        if ENV_EXAMPLE.exists():
            import shutil as sh
            sh.copy(ENV_EXAMPLE, ENV_FILE)
            print(yellow(
                f"\n  ⚠  Created backend/.env from .env.example\n"
                f"     {bold('Set NVIDIA_API_KEY in backend/.env before running the agent.')}"
            ))
        else:
            ENV_FILE.write_text("NVIDIA_API_KEY=your_key_here\n")
            print(yellow("\n  ⚠  Created a blank backend/.env — set NVIDIA_API_KEY inside it."))
        return False

    content = ENV_FILE.read_text()
    has_key = (
        "NVIDIA_API_KEY=" in content
        and "your_key_here" not in content
        and content.split("NVIDIA_API_KEY=")[-1].strip()[:5] not in ("", "your_")
    )
    if not has_key:
        print(yellow(
            f"\n  ⚠  NVIDIA_API_KEY is not set in backend/.env\n"
            f"     The frontend will run but agent features will be disabled.\n"
            f"     Get your free key at {cyan('https://build.nvidia.com')}"
        ))
    return has_key


def _preflight() -> None:
    """Check system requirements and abort early if something is missing."""
    ok = True

    # Python version
    if sys.version_info < (3, 10):
        print(red(f"  ✗ Python 3.10+ required (found {sys.version})"))
        ok = False
    else:
        print(green(f"  ✓ Python {sys.version.split()[0]}"))

    # Node
    if not _check_tool("node"):
        print(red("  ✗ Node.js not found — install from https://nodejs.org"))
        ok = False
    else:
        v = subprocess.check_output(["node", "--version"], text=True).strip()
        print(green(f"  ✓ Node.js {v}"))

    # npm
    if not _check_tool("npm"):
        print(red("  ✗ npm not found — it ships with Node.js"))
        ok = False
    else:
        print(green("  ✓ npm found"))

    if not ok:
        sys.exit(1)


# ── Process management ───────────────────────────────────────

_procs: list[subprocess.Popen] = []


def _start_backend() -> subprocess.Popen:
    """Launch the FastAPI server via uvicorn."""
    print(f"\n{yellow('▶')} Starting backend  {dim('(FastAPI · http://localhost:8000)')}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)          # so `from backend.xxx` works
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable, "-m", "uvicorn",
        "backend.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
        "--reload-dir", str(BACKEND_DIR),
        "--log-level", "info",
    ]

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        # Don't capture — let logs stream to the terminal
    )
    print(green(f"  ✓ Backend started  (PID {proc.pid})"))
    print(dim(  f"    API docs → http://localhost:8000/docs"))
    print(dim(  f"    Health   → http://localhost:8000/api/health"))
    return proc


def _start_frontend() -> subprocess.Popen:
    """Launch the Vite dev server."""
    print(f"\n{yellow('▶')} Starting frontend {dim('(Vite · http://localhost:5173)')}")

    npm_cmd = "npm.cmd" if IS_WIN else "npm"
    proc = subprocess.Popen(
        [npm_cmd, "run", "dev"],
        cwd=str(ROOT),
    )
    print(green(f"  ✓ Frontend started (PID {proc.pid})"))
    print(dim(  f"    App URL  → http://localhost:5173"))
    return proc


def _wait_and_watch(procs: list[subprocess.Popen]) -> None:
    """Block until all processes exit or the user hits Ctrl-C."""
    print(f"\n{dim('─' * 52)}")
    print(bold("  Both servers are running.  Press Ctrl-C to stop all."))
    print(f"{dim('─' * 52)}\n")

    def _stop(sig=None, frame=None):
        print(f"\n{yellow('Shutting down …')}")
        for p in procs:
            try:
                if IS_WIN:
                    p.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    p.terminate()
            except Exception:
                pass
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        print(green("  All servers stopped.  Goodbye!\n"))
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop)
    if not IS_WIN:
        signal.signal(signal.SIGTERM, _stop)

    while True:
        for p in procs:
            rc = p.poll()
            if rc is not None:
                print(red(f"\n  Process (PID {p.pid}) exited with code {rc}."))
                _stop()
        time.sleep(1)


# ── Main ─────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VisCurator dev launcher — starts backend and/or frontend.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--backend",  action="store_true", help="Start backend only")
    parser.add_argument("--frontend", action="store_true", help="Start frontend only")
    parser.add_argument("--setup",    action="store_true", help="Install deps then start")
    args = parser.parse_args()

    print(BANNER)

    # ── Pre-flight ──
    print(bold("System checks:"))
    _preflight()

    # ── Dependency install ──
    if args.setup:
        _pip_install()
        _npm_install()
    else:
        # Gentle nudge if node_modules is missing
        if not NODE_MODULES.exists() and not args.backend:
            print(yellow("\n  node_modules not found — running npm install …"))
            _npm_install()

    # ── .env check ──
    _ensure_env()

    procs: list[subprocess.Popen] = []

    # ── Start servers ──
    run_backend  = not args.frontend   # True unless --frontend only
    run_frontend = not args.backend    # True unless --backend only

    if run_backend:
        procs.append(_start_backend())
        if run_frontend:
            # Give the backend a moment to bind its port
            time.sleep(2)

    if run_frontend:
        procs.append(_start_frontend())

    if not procs:
        print(red("Nothing to start. Use --backend, --frontend, or neither for both."))
        sys.exit(1)

    _wait_and_watch(procs)


if __name__ == "__main__":
    main()
