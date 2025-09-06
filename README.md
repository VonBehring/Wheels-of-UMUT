# Wheels of UMUT — FlashAttention 2.8.2 (CUDA 12.6 • Python 3.12 • SM80/A100)

Unofficial community wheel for **FlashAttention 2.8.2**.  
Built for **Python 3.12 (cp312)**, **PyTorch 2.8.x**, **CUDA 12.6**, and **NVIDIA A100 (SM80)**.

> **Note:** This wheel is **not official**. Upstream project: **Dao-AILab/flash-attention** (BSD-3-Clause).

---

## 🔽 One-liner install

Release asset:  
`flash_attn-2.8.2-cp312-cp312-linux_x86_64.whl`

```bash
pip install "https://github.com/VonBehring/Wheels-of-UMUT/releases/download/fa2.8.2-cu126-py312-sm80/flash_attn-2.8.2-cp312-cp312-linux_x86_64.whl"   --hash=sha256:6d491b37fd351d7d4ee998f3281faa4667b4e2f22024e11e160a5adb1f390916
```

Verify the installed version:
```bash
python - << 'PY'
import importlib.metadata as im
print("flash-attn =", im.version("flash-attn"))
PY
```

---

## ✅ Compatibility

- **Python:** 3.12 (cp312)  
- **PyTorch:** 2.8.x  
- **CUDA runtime:** 12.6  *(the wheel does **not** bundle CUDA libs; it links to your system CUDA)*  
- **GPU architecture:** **SM80 (NVIDIA A100)**  
  > Other architectures (e.g., SM86/RTX 30xx, SM90/H100) may not work with this build.

---

## 🔐 SHA256

```
6d491b37fd351d7d4ee998f3281faa4667b4e2f22024e11e160a5adb1f390916
```

Integrity check:
```bash
sha256sum flash_attn-2.8.2-cp312-cp312-linux_x86_64.whl
```

---

## 🛠️ Build from source (Colab — single cell)

The cell below prefers **CUDA 12.6**, uses **Ninja**, prints streaming logs + a simple heartbeat, builds for **A100 (SM80)**, installs the wheel, and prints its SHA256.

```python
# Colab: flash-attn 2.8.2 — ULTIMATE build (prefer CUDA 12.6) + live log + heartbeat
import os, sys, time, glob, subprocess, threading, importlib.metadata as im, shutil, hashlib

def run(cmd, env=None, cwd=None, log_path=None):
    print(f"\n$ {cmd}", flush=True)
    p = subprocess.Popen(cmd, shell=True, cwd=cwd, executable="/bin/bash",
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, env=env)
    alive = True
    def heartbeat():
        while alive:
            print(f"[heartbeat {time.strftime('%H:%M:%S')}]"); sys.stdout.flush()
            time.sleep(30)
    hb = threading.Thread(target=heartbeat, daemon=True); hb.start()
    log_f = open(log_path, "a", encoding="utf-8") if log_path else None
    try:
        for line in p.stdout:
            if log_f: log_f.write(line)
            print(line, end="")
    finally:
        if log_f: log_f.flush(); log_f.close()
    p.wait(); alive = False; hb.join(timeout=0.1)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {cmd}")

# [0] Environment info
print("[0] Environment info")
try:
    import torch
    print("torch =", torch.__version__)
    print("cuda.is_available =", torch.cuda.is_available())
    print("torch.version.cuda =", torch.version.cuda)
except Exception as e:
    print("Torch check error:", e)

# [1] Prefer CUDA 12.6 if available
env = os.environ.copy()
for c in ["/usr/local/cuda-12.6", "/usr/local/cuda"]:
    if os.path.isdir(c):
        env["CUDA_HOME"] = c; break
else:
    env["CUDA_HOME"] = "/usr/local/cuda"

env["PATH"] = f"{env['CUDA_HOME']}/bin:" + env.get("PATH","")
env["LD_LIBRARY_PATH"] = f"{env['CUDA_HOME']}/lib64:" + env.get("LD_LIBRARY_PATH","")
env["TORCH_CUDA_ARCH_LIST"] = "8.0"  # A100 (SM80)
env["MAX_JOBS"] = str(os.cpu_count() or 1)
env["USE_NINJA"] = "1"
env["CMAKE_BUILD_PARALLEL_LEVEL"] = env["MAX_JOBS"]
env["NINJA_STATUS"] = "[ninja %f/%t | %o/s] "
env["PYTHONUNBUFFERED"] = "1"
print("[env]", {k: env[k] for k in ["CUDA_HOME","TORCH_CUDA_ARCH_LIST","MAX_JOBS","USE_NINJA"]})

# [2] Tooling
run("pip install -U pip setuptools wheel ninja cmake packaging", env=env)
run("ninja --version || (pip uninstall -y ninja && pip install ninja)", env=env)

# [3] Clean & fetch sources
run("pip uninstall -y flash-attn flash_attn || true", env=env)
shutil.rmtree("/content/flash-attention-src", ignore_errors=True)
run("git clone --progress --branch v2.8.2 --depth 1 https://github.com/Dao-AILab/flash-attention.git /content/flash-attention-src", env=env)
run("cd /content/flash-attention-src && git submodule update --init --recursive || true", env=env)

# [4] Build (streamed) + log
log_file = "/content/flash-attn-build.log"
run("cd /content/flash-attention-src && python -u setup.py bdist_wheel -v", env=env, log_path=log_file)

# [5] Find wheel, install, verify (+ SHA256)
wheels = sorted(glob.glob("/content/flash-attention-src/dist/flash_attn-2.8.2-*.whl"))
if not wheels:
    raise FileNotFoundError("Built wheel not found.")
wheel_path = wheels[-1]
print("\n[wheel]", wheel_path)
run(f"pip install '{wheel_path}'", env=env)
print("flash-attn =", im.version("flash-attn"))
sha = hashlib.sha256(open(wheel_path,"rb").read()).hexdigest()
print("SHA256:", sha)
print("✅ flash-attn 2.8.2 installed. Log:", log_file)
```

### Multi-arch build (optional)
Wider compatibility (longer build, larger wheel):
```bash
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"  # A100; RTX 30xx; H100
```

---

## 🧩 Troubleshooting

- **Accidentally upgraded to 2.8.3 (import error with xFormers):**  
  Remove and install **this** pinned wheel:
  ```bash
  pip uninstall -y flash-attn flash_attn
  pip install "https://github.com/VonBehring/Wheels-of-UMUT/releases/download/fa2.8.2-cu126-py312-sm80/flash_attn-2.8.2-cp312-cp312-linux_x86_64.whl"     --hash=sha256:6d491b37fd351d7d4ee998f3281faa4667b4e2f22024e11e160a5adb1f390916
  ```
- **ComfyUI port already in use (8188):**
  ```bash
  fuser -n tcp 8188 -k || true
  ```

---

## 📜 License

- **Upstream:** [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) — **BSD-3-Clause**.  
- This repository preserves upstream copyright and license notices; provided **AS IS / NO WARRANTY**.  
- CUDA libraries are **not** bundled; the wheel dynamically links to the CUDA runtime present on the user’s system.

---

## 🙌 Contributing

Issues and PRs are welcome. Please include your Python, PyTorch, CUDA versions, GPU model, and full error logs when reporting problems.
