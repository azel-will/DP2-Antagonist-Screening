#!/usr/bin/env python
"""
env_check.py
Python + core libs + hardware + GPU
Output: results/env_summary.json 
"""
import sys, platform, subprocess, json, psutil, torch
from pathlib import Path

# ---------- 1. Python & OS ----------
print("Python:", sys.version)
print("Platform:", platform.platform())
print("CPU:", platform.processor() or "N/A")
print("Python executable:", sys.executable)

# ---------- 2. Hardware ----------
mem = psutil.virtual_memory()
print(f"Memory: {mem.total // (1024**3)} GB, used {mem.percent}%")
print(f"CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")

# ---------- 3. GPU ----------
if torch.cuda.is_available():
    print("GPU detected:", torch.cuda.get_device_name(0))
    print("GPU memory:", torch.cuda.get_device_properties(0).total_memory // (1024**3), "GB")
    print("CUDA version:", torch.version.cuda)
else:
    print("GPU: not available")

# ---------- 4. Core Libraries ----------
subprocess.run(["python", "-m", "pip", "list"], check=True)

# ---------- 5. Export to JSON (SI reference) ----------
env = {
    "python": sys.version,
    "platform": platform.platform(),
    "cpu_cores": psutil.cpu_count(),
    "memory_GB": mem.total // (1024**3),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "cuda": torch.version.cuda if torch.cuda.is_available() else None,
}
out_path = Path(__file__).parent.parent / "env_summary.json"
out_path.parent.mkdir(exist_ok=True)
with open(out_path, "w") as f:
    json.dump(env, f, indent=2)

print(f"\n✅ env_summary.json saved → {out_path}  (Git-tracked)")