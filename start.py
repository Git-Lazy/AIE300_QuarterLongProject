"""Convenience script to install deps and start the app + demo.

Usage:
  python start.py           # install, start uvicorn, open browser
  python start.py --no-install --no-open  # skip install and browser
  python start.py --demo    # also run function_call_demo in parallel

Notes:
- Run this with the virtualenv's Python (activate venv first) so packages
  install into the correct environment. The script will call pip via
  the running interpreter (`sys.executable`).
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import webbrowser
from typing import Optional

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

APP_URL = "http://127.0.0.1:8000/"


def run_install(requirements: str = "requirements.txt") -> None:
    if not os.path.exists(requirements):
        print(f"No {requirements} found, skipping install.")
        return

    cmd = [sys.executable, "-m", "pip", "install", "-r", requirements]
    print("Installing dependencies:", " ".join(cmd))
    subprocess.check_call(cmd)


def load_env_file(dotenv_path: Optional[str] = ".env") -> None:
    if load_dotenv is None:
        print("python-dotenv not available; make sure OPENAI_API_KEY is set in environment.")
        return
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
        print(f"Loaded environment from {dotenv_path}")


def start_uvicorn() -> subprocess.Popen:
    cmd = [sys.executable, "-m", "uvicorn", "main:app", "--reload", "--host", "127.0.0.1", "--port", "8000"]
    print("Starting server:", " ".join(cmd))
    env = os.environ.copy()
    # Start uvicorn in a subprocess; inherit stdout/stderr
    p = subprocess.Popen(cmd, env=env)
    return p


def start_demo() -> subprocess.Popen:
    cmd = [sys.executable, "function_call_demo.py"]
    print("Starting demo:", " ".join(cmd))
    env = os.environ.copy()
    p = subprocess.Popen(cmd, env=env)
    return p


def wait_for_server(url: str = APP_URL, timeout: int = 12) -> bool:
    try:
        import requests
    except Exception:
        # If requests isn't available just sleep a bit and hope for the best
        time.sleep(2)
        return True

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=1)
            if r.status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-install", action="store_true", help="Skip installing requirements")
    parser.add_argument("--no-open", action="store_true", help="Do not open the browser")
    parser.add_argument("--demo", action="store_true", help="Also run function_call_demo.py alongside the server")
    args = parser.parse_args()

    if not args.no_install:
        try:
            run_install()
        except subprocess.CalledProcessError as e:
            print("Failed to install requirements:", e)
            sys.exit(1)

    # load .env into this process so child inherits variables
    load_env_file()

    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY is not set. Set it in .env or export it in your shell.")

    server_proc = start_uvicorn()
    demo_proc = None
    try:
        # wait for server to be reachable
        ok = wait_for_server()
        if ok:
            print(f"Server appears up at {APP_URL}")
            if not args.no_open:
                try:
                    webbrowser.open(APP_URL)
                except Exception:
                    print("Could not open browser automatically.")
        else:
            print("Server did not respond within timeout. Check logs.")

        if args.demo:
            demo_proc = start_demo()

        # Wait and forward signals
        def _signal_handler(signum, frame):
            print("Received signal, stopping child processes...")
            if demo_proc and demo_proc.poll() is None:
                demo_proc.terminate()
            if server_proc and server_proc.poll() is None:
                server_proc.terminate()
            sys.exit(0)

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        # Wait for server process to exit
        while True:
            rc = server_proc.poll()
            if rc is not None:
                print(f"Server exited with code {rc}")
                break
            time.sleep(0.5)

    finally:
        if demo_proc and demo_proc.poll() is None:
            demo_proc.terminate()
        if server_proc and server_proc.poll() is None:
            server_proc.terminate()


if __name__ == "__main__":
    main()
