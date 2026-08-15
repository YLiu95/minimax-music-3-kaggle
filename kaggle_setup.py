from __future__ import annotations

import json
import os
import shutil
import site
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SGLANG_REPO = "https://github.com/sgl-project/sglang-omni.git"
SGLANG_REVISION = "2d2ff5056f8c321f1dbc2ff6584baf05996ce150"
MODEL_REPO = "MiniMaxAI/MiniMax-Music3"
MODEL_REVISION = "fbdf52fbaaca799592917417eb05f1899f1255ec"
WORK_DIR = Path("/kaggle/working/minimax-music3-runtime")
SOURCE_DIR = WORK_DIR / "sglang-omni"
VENV_DIR = WORK_DIR / ".venv"
MODEL_DIR = WORK_DIR / "model"
LOG_PATH = WORK_DIR / "server.log"
HEALTH_URL = "http://127.0.0.1:8000/health"

SOURCE_PATCHES = (
    "sglang-omni-gpu-only.patch",
    "sglang-omni-bitsandbytes-loader.patch",
    "sglang-omni-t4-rmsnorm.patch",
    "sglang-omni-t4-ar-dtype.patch",
    "sglang-omni-t4-attention-dtype.patch",
    "sglang-omni-t4-triton-attention.patch",
    "sglang-omni-shm-transport.patch",
)


def _run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("+", " ".join(command))
    return subprocess.run(command, check=True, **kwargs)


def _healthy() -> bool:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=2) as response:
            payload = json.load(response)
        return response.status == 200 and payload.get("status") == "healthy"
    except (OSError, ValueError, urllib.error.URLError):
        return False


def _tail(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return "(server log was not created)"
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def _ensure_two_gpus() -> None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    if len(rows) != 2:
        raise RuntimeError(f"MiniMax Music3 requires exactly two visible GPUs; found {rows}")
    for row in rows:
        memory_mib = int(row.rsplit(",", 1)[1].strip())
        if memory_mib < 15000:
            raise RuntimeError(f"Each GPU needs at least 15,000 MiB VRAM; found {row}")
    print("GPUs:", *rows, sep="\n  ")


def _ensure_uv() -> str:
    uv = shutil.which("uv")
    if uv:
        return uv
    _run([sys.executable, "-m", "pip", "install", "uv==0.11.13"])
    uv = shutil.which("uv") or str(Path(sys.executable).parent / "uv")
    if not Path(uv).exists():
        raise RuntimeError("uv installation completed but the executable was not found")
    return uv


def _ensure_source() -> None:
    if not (SOURCE_DIR / ".git").exists():
        SOURCE_DIR.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--filter=blob:none", SGLANG_REPO, str(SOURCE_DIR)])
    revision = subprocess.check_output(
        ["git", "-C", str(SOURCE_DIR), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != SGLANG_REVISION:
        _run(["git", "-C", str(SOURCE_DIR), "fetch", "origin", SGLANG_REVISION])
        _run(["git", "-C", str(SOURCE_DIR), "checkout", "--detach", SGLANG_REVISION])


def _apply_source_patch(root: Path, patch_path: Path) -> None:
    with patch_path.open("rb") as patch_file:
        reverse = subprocess.run(
            ["patch", "--dry-run", "--reverse", "--strip=1", f"--directory={root}"],
            stdin=patch_file,
            capture_output=True,
        )
    if reverse.returncode == 0:
        return
    with patch_path.open("rb") as patch_file:
        _run(
            ["patch", "--forward", "--strip=1", f"--directory={root}"],
            stdin=patch_file,
        )


def _apply_site_patch(site_packages: Path, patch_path: Path) -> None:
    reverse = subprocess.run(
        ["patch", "--dry-run", "--reverse", "--strip=1", f"--directory={site_packages}"],
        stdin=patch_path.open("rb"),
        capture_output=True,
    )
    if reverse.returncode == 0:
        return
    with patch_path.open("rb") as patch_file:
        _run(
            ["patch", "--forward", "--strip=1", f"--directory={site_packages}"],
            stdin=patch_file,
        )


def _install_runtime(uv: str, support_dir: Path) -> Path:
    venv_python = VENV_DIR / "bin" / "python"
    if not venv_python.exists():
        _run([uv, "venv", str(VENV_DIR), "--python", sys.executable])
    install_env = os.environ.copy()
    install_env.update(
        {
            "UV_CONCURRENT_DOWNLOADS": "1",
            "UV_CONCURRENT_BUILDS": "1",
            "MAX_JOBS": "2",
        }
    )
    _run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(venv_python),
            "-e",
            str(SOURCE_DIR),
            "bitsandbytes==0.49.2",
        ],
        env=install_env,
    )
    site_packages = Path(
        subprocess.check_output(
            [str(venv_python), "-c", "import site; print(site.getsitepackages()[0])"],
            text=True,
        ).strip()
    )
    _apply_site_patch(site_packages, support_dir / "patches" / "sglang-nf4-weight-name.patch")
    return venv_python


def _download_model(venv_python: Path, hf_token: str) -> None:
    code = f"""
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id={MODEL_REPO!r},
    revision={MODEL_REVISION!r},
    local_dir={str(MODEL_DIR)!r},
    token=True,
    max_workers=2,
    allow_patterns=[
        'config.json',
        'dav.pth',
        'flowmatching_vae.pth',
        'qwen_7B/qwen_7B/*',
        'qwen_7B/qwen3-8B-tokenizer-music/*',
    ],
)
"""
    download_env = os.environ.copy()
    download_env.update(
        {
            "HF_TOKEN": hf_token,
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_DOWNLOAD_TIMEOUT": "600",
        }
    )
    _run([str(venv_python), "-c", code], env=download_env)

    index_path = MODEL_DIR / "qwen_7B" / "qwen_7B" / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    qwen_dir = index_path.parent
    missing = sorted(
        name for name in set(index["weight_map"].values()) if not (qwen_dir / name).is_file()
    )
    if missing:
        raise RuntimeError(f"Model download is incomplete; missing shards: {missing}")
    print(f"Checkpoint ready: {len(set(index['weight_map'].values()))} Qwen shards")


def setup(hf_token: str) -> subprocess.Popen | None:
    """Install, download, start, and health-check MiniMax Music3 on two T4 GPUs."""
    if _healthy():
        print("MiniMax Music3 is already healthy at http://127.0.0.1:8000")
        return None
    if not hf_token:
        raise ValueError("HF_TOKEN is empty")

    _ensure_two_gpus()
    support_dir = Path(__file__).resolve().parent
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_source()
    for patch_name in SOURCE_PATCHES:
        _apply_source_patch(SOURCE_DIR, support_dir / "patches" / patch_name)

    uv = _ensure_uv()
    venv_python = _install_runtime(uv, support_dir)
    _download_model(venv_python, hf_token)

    server_env = os.environ.copy()
    server_env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "MINIMAX_MUSIC3_AR_CONCURRENCY": "1",
            "SGLANG_OMNI_INTRA_NODE_TRANSPORT": "shm",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "LIBRARY_PATH": "/usr/local/nvidia/lib64",
            "LD_LIBRARY_PATH": "/usr/local/nvidia/lib64:"
            + server_env.get("LD_LIBRARY_PATH", ""),
            "OMP_NUM_THREADS": "2",
            "MAX_JOBS": "2",
        }
    )
    command = [
        str(VENV_DIR / "bin" / "sgl-omni"),
        "serve",
        "--model-path",
        str(MODEL_DIR),
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--quantization",
        "bitsandbytes",
        "--cpu-offload-gb",
        "0",
        "--mem-fraction-static",
        "0.70",
        "--max-running-requests",
        "1",
    ]
    log_file = LOG_PATH.open("w")
    process = subprocess.Popen(
        command,
        cwd=SOURCE_DIR,
        env=server_env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_file.close()

    deadline = time.monotonic() + 20 * 60
    while time.monotonic() < deadline:
        if _healthy():
            print("MiniMax Music3 is healthy at http://127.0.0.1:8000")
            print(f"Server log: {LOG_PATH}")
            return process
        if process.poll() is not None:
            raise RuntimeError(f"Server exited with code {process.returncode}:\n{_tail(LOG_PATH)}")
        time.sleep(5)

    process.terminate()
    raise TimeoutError(f"Server did not become healthy within 20 minutes:\n{_tail(LOG_PATH)}")
