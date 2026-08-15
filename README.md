# MiniMax Music 3 on Kaggle 2xT4

Run [MiniMax Music 3](https://github.com/MiniMax-AI/MiniMax-Music3) on a Kaggle notebook with two Tesla T4 GPUs. The tested runtime uses SGLang-Omni, NF4 weight-only quantization for the autoregressive model, FP32 acoustic synthesis, Triton attention, and shared-memory transfer between stages.

## Before You Start

1. Create a Kaggle notebook.
2. In **Notebook options**, enable **GPU T4 x2** and **Internet**.
3. Add these Kaggle secrets:
   - `HF_TOKEN`: a Hugging Face token that can download `MiniMaxAI/MiniMax-Music3`.
   - `GITHUB_TOKEN`: a GitHub token that can read this repository.
4. Paste Cell 1 and run it once. Keep the notebook session alive after it reports healthy.
5. Edit the prompt in Cell 2 and run it whenever you want a new song.

The first run downloads only the 58 required model files, about 28.8 GB, and installs everything under `/root/minimax-music3-runtime`. Nothing is downloaded or installed under `/kaggle/working`. Startup typically takes several minutes. A 10-second clip takes roughly 2-4 minutes on two T4s.

## Cell 1: Setup and Start

```python
import base64
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
github_token = secrets.get_secret("GITHUB_TOKEN")
hf_token = secrets.get_secret("HF_TOKEN")
if not github_token or not hf_token:
    raise RuntimeError("Add GITHUB_TOKEN and HF_TOKEN in Kaggle Secrets, then rerun Cell 1.")

for legacy_dir in (
    Path("/kaggle/working/minimax-music3-runtime"),
    Path("/kaggle/working/minimax-music-3-kaggle"),
):
    if legacy_dir.exists():
        print(f"Removing legacy runtime folder: {legacy_dir}")
        shutil.rmtree(legacy_dir)

support_dir = Path("/root/minimax-music-3-kaggle")

if not (support_dir / ".git").exists():
    auth = base64.b64encode(f"x-access-token:{github_token}".encode()).decode()
    git_env = os.environ.copy()
    git_env.update(
        {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "http.extraHeader",
            "GIT_CONFIG_VALUE_0": f"Authorization: Basic {auth}",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    subprocess.run(
        [
            "git",
            "clone",
            "--depth=1",
            "https://github.com/YLiu95/minimax-music-3-kaggle.git",
            str(support_dir),
        ],
        env=git_env,
        check=True,
    )

sys.path.insert(0, str(support_dir))
kaggle_setup = importlib.import_module("kaggle_setup")
SERVER_PROCESS = kaggle_setup.setup(hf_token)
```

Wait for:

```text
MiniMax Music3 is healthy at http://127.0.0.1:8000
```

Rerunning Cell 1 is safe. It resumes model downloads, skips applied patches, and reuses a healthy server.

Cell 1 also removes only the two legacy folders created by the previous `/kaggle/working`-based version, reclaiming any space consumed by its interrupted download.

## Cell 2: Generate Music

All editable configuration is in the section at the bottom of the cell. Edit it, then run:

```python
import wave
from datetime import datetime, timezone
from pathlib import Path

import requests
from IPython.display import Audio, display


def generate_music(
    *,
    server_url,
    model_name,
    lyrics,
    caption,
    seed,
    max_new_tokens,
    output_dir,
    output_prefix,
    request_timeout,
):
    response = requests.post(
        f"{server_url}/v1/audio/speech",
        json={
            "model": model_name,
            "input": lyrics,
            "instructions": caption,
            "response_format": "wav",
            "seed": seed,
            "max_new_tokens": max_new_tokens,
            "stream": False,
        },
        timeout=request_timeout,
    )
    if not response.ok:
        raise RuntimeError(
            f"Generation failed ({response.status_code}): {response.text[:2000]}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    output = output_dir / f"{output_prefix}_{timestamp}_seed{seed}.wav"
    output.write_bytes(response.content)

    with wave.open(str(output), "rb") as wav_file:
        assert wav_file.getnchannels() == 2
        assert wav_file.getframerate() == 32000
        duration = wav_file.getnframes() / wav_file.getframerate()
    print(f"Saved {output} ({duration:.2f} seconds, {output.stat().st_size:,} bytes)")
    display(Audio(filename=str(output)))
    return output


# Configuration: edit values only in this section.
SERVER_URL = "http://127.0.0.1:8000"
MODEL_NAME = "MiniMaxAI/MiniMax-Music3"
LYRICS = """[Verse]
Morning light is filtering through the pines
[Chorus]
Softly the world begins to breathe"""
CAPTION = (
    "A warm acoustic pop song at 92 BPM with intimate female vocals, "
    "fingerpicked guitar, soft piano, and a gradual build into a wide final chorus."
)
SEED = 7
MAX_NEW_TOKENS = 250  # 25 frames/second: 250 is about 10 seconds.
OUTPUT_DIR = Path("/kaggle/working/minimax-music3-output")
OUTPUT_PREFIX = "minimax_music3"
REQUEST_TIMEOUT = 6 * 60 * 60

OUTPUT = generate_music(
    server_url=SERVER_URL,
    model_name=MODEL_NAME,
    lyrics=LYRICS,
    caption=CAPTION,
    seed=SEED,
    max_new_tokens=MAX_NEW_TOKENS,
    output_dir=OUTPUT_DIR,
    output_prefix=OUTPUT_PREFIX,
    request_timeout=REQUEST_TIMEOUT,
)
```

Each WAV is displayed in the notebook and saved under `/kaggle/working/minimax-music3-output` with a UTC timestamp and seed, for example `minimax_music3_20260815_181900_123456Z_seed7.wav`.

## Prompt Guide

- Put section tags on their own lines: `[Intro]`, `[Verse]`, `[Pre-Chorus]`, `[Chorus]`, `[Bridge]`, `[Instrumental]`, `[Solo]`, and `[Outro]`.
- Do not write lyrics on the same line as a section tag. Use `[Verse]\nFirst lyric line`, not `[Verse] First lyric line`.
- Describe genre, BPM, instruments, vocal style, mood, arrangement, and production in `CAPTION`.
- For instrumental music, use minimal lyrics such as `[Intro]\n(instrumental)` and explicitly request no vocals in the caption.
- The same prompt and seed are deterministic. Change only `SEED` to create another take.

`MAX_NEW_TOKENS` is an audio-frame cap at 25 frames per second:

| Frames | Maximum duration |
|---:|---:|
| 250 | 10 seconds |
| 500 | 20 seconds |
| 750 | 30 seconds |
| 1500 | 60 seconds |
| 9000 | 5 minutes |

Start with 250 frames. Long generations on T4 are slow. The model may end naturally before the cap.

## Memory and Placement

- GPU 0: Qwen3/RVQ autoregressive stage in on-load NF4 with FP16 compute.
- GPU 1: Flow Matching and DAV waveform decoder in FP32.
- `--cpu-offload-gb 0` is always set. Model weights are not offloaded to CPU or disk during inference.
- Source, Python environment, package caches, and checkpoint files are stored under `/root`, not `/kaggle/working`.
- Only timestamped WAV files and server logs are written under `/kaggle/working`, in separate dedicated folders.
- Kaggle blocks the CUDA IPC syscall used by the default relay. Only streamed hidden-state chunks cross host shared memory; model weights remain GPU-resident.
- Installation and model download concurrency are deliberately capped to protect Kaggle host RAM.

## Troubleshooting

- Server log: `/kaggle/working/minimax-music3-logs/server.log`
- Generated WAV files: `/kaggle/working/minimax-music3-output/`
- Health check: `http://127.0.0.1:8000/health`
- If the Kaggle session restarts, rerun Cell 1. Existing files under `/root/minimax-music3-runtime` are reused when available.
- If Cell 2 says connection refused, rerun Cell 1 and wait for the healthy message.
- Do not raise `--mem-fraction-static` or `--max-running-requests` on T4; the tested values leave required room for adapters and CUDA graphs.

The same two cells are also available in `kaggle_cells.py`, separated by `# %%` markers.
