# Resume Notes

Last updated: 2026-08-15

## Current state

- Kaggle host has two Tesla T4 GPUs, each with 15,360 MiB VRAM.
- SGLang Omni source is at `/root/sglang-omni`, commit `2d2ff5056f8c321f1dbc2ff6584baf05996ce150`.
- Isolated environment is at `/root/minimax-music3-venv`.
- Model files are at `/root/minimax-music3-model` and occupy about 27 GB.
- The Qwen index references 47 shards (`00000` through `00046`); all are present.
- The MiniMax Music3 focused unit suite previously passed 89 tests.
- No model or download process was running after the VS Code tunnel dropped.

## Verified checkpoint sizes

- Qwen BF16 tensors: 18,461,001,728 bytes total.
- Audio adapter tensors filtered from the base SGLang load: 1,292,050,432 bytes.
- Remaining backbone, token embeddings, and LM head: 17,168,951,296 bytes before KV cache and CUDA workspaces.
- `flowmatching_vae.pth`: 9,828,468,476 bytes; loaded directly on GPU 1.
- `dav.pth`: 491,817,450 bytes; loaded directly on GPU 1.

## Active blocker under investigation

The unquantized AR stage cannot fit on a 15,360 MiB T4: its retained BF16 weights alone are about 15.99 GiB, and the backend rejects tensor parallelism. The next step is to validate a supported weight-only quantized SGLang load that stays entirely on GPU at runtime. CPU/disk offload must remain disabled.

The current upstream `load_audio_state` helper also stages about 1.29 GB of adapter tensors in CPU RAM before copying them to CUDA. If the quantized route fits, patch this helper locally to load selected safetensors directly onto the AR CUDA device before finalizing the Kaggle cells.

## Useful commands

```bash
nvidia-smi
git -C /root/sglang-omni status --short --branch
git -C /root/minimax-music-3-kaggle status --short --branch
find /root/minimax-music3-model/qwen_7B/qwen_7B -name 'model-*.safetensors' | wc -l
```

Do not load the full checkpoint into CPU RAM. Use safetensors metadata inspection or direct CUDA loading only.