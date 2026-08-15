# %% Kaggle Cell 1: install, download, and start the two-GPU service
import base64
import importlib
import os
import subprocess
import sys
from pathlib import Path

from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
github_token = secrets.get_secret("GITHUB_TOKEN")
hf_token = secrets.get_secret("HF_TOKEN")
if not github_token or not hf_token:
    raise RuntimeError("Add GITHUB_TOKEN and HF_TOKEN in Kaggle Secrets, then rerun Cell 1.")
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

# %% Kaggle Cell 2: edit the prompt and generate a WAV
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
