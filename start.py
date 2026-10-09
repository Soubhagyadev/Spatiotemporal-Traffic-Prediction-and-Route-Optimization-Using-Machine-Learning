"""Start all three services: Flask API → Dash dashboard → Django web UI."""
import os, subprocess, sys, time
from pathlib import Path

# Use the venv interpreter if it exists, otherwise fall back to current Python
_venv = Path(__file__).parent / ".venv" / "bin" / "python"
py = str(_venv) if _venv.exists() else sys.executable

# Bind to 0.0.0.0 in Docker so ports are reachable from outside the container
host = "0.0.0.0" if os.environ.get("RUNNING_IN_DOCKER") else "127.0.0.1"

services = [
    (["scripts/run_api.py",       "--host", host], f"Flask API      → http://{host}:5001"),
    (["scripts/run_dashboard.py", "--host", host], f"Dash dashboard → http://{host}:8050"),
    (["scripts/run_web.py",       "--host", host], f"Django web UI  → http://{host}:8000"),
]

procs = []
for args, label in services:
    print(f"Starting {label}")
    procs.append(subprocess.Popen([py] + args))
    time.sleep(2)

print("\nAll services running. Ctrl+C to stop.\n")
try:
    for p in procs:
        p.wait()
except KeyboardInterrupt:
    for p in procs:
        p.terminate()
