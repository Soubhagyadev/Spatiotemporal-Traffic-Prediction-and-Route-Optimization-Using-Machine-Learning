"""Start all three services: Flask API → Dash dashboard → Django web UI."""
import subprocess, sys, time
from pathlib import Path

# Use the venv interpreter if it exists, otherwise fall back to current Python
_venv = Path(__file__).parent / ".venv" / "bin" / "python"
py = str(_venv) if _venv.exists() else sys.executable
services = [
    (["scripts/run_api.py"],       "Flask API      → http://127.0.0.1:5001"),
    (["scripts/run_dashboard.py"], "Dash dashboard → http://127.0.0.1:8050"),
    (["scripts/run_web.py"],       "Django web UI  → http://127.0.0.1:8000"),
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
