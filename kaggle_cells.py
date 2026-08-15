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
support_dir = Path("/kaggle/working/minimax-music-3-kaggle")

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
from pathlib import Path

import requests
from IPython.display import Audio, display

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
OUTPUT = Path("/kaggle/working/minimax_music3.wav")

response = requests.post(
    "http://127.0.0.1:8000/v1/audio/speech",
    json={
        "model": "MiniMaxAI/MiniMax-Music3",
        "input": LYRICS,
        "instructions": CAPTION,
        "response_format": "wav",
        "seed": SEED,
        "max_new_tokens": MAX_NEW_TOKENS,
        "stream": False,
    },
    timeout=6 * 60 * 60,
)
if not response.ok:
    raise RuntimeError(f"Generation failed ({response.status_code}): {response.text[:2000]}")
OUTPUT.write_bytes(response.content)

with wave.open(str(OUTPUT), "rb") as wav_file:
    assert wav_file.getnchannels() == 2
    assert wav_file.getframerate() == 32000
    duration = wav_file.getnframes() / wav_file.getframerate()
print(f"Saved {OUTPUT} ({duration:.2f} seconds, {OUTPUT.stat().st_size:,} bytes)")
display(Audio(filename=str(OUTPUT)))
