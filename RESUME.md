# Resume Notes

Last updated: 2026-08-15

## Current state

- Kaggle host has two Tesla T4 GPUs, each with 15,360 MiB VRAM.
- SGLang Omni source is at `/root/sglang-omni`, commit `2d2ff5056f8c321f1dbc2ff6584baf05996ce150`.
- Isolated environment is at `/root/minimax-music3-venv`.
- Model files are at `/root/minimax-music3-model` and occupy about 27 GB.
- The Qwen index references 47 shards (`00000` through `00046`); all are present.
- The MiniMax Music3 focused unit suite previously passed 89 tests.
- A GPU-only adapter-loading patch is saved at `patches/sglang-omni-gpu-only.patch`.
- A BitsAndBytes loader-selection patch is saved at `patches/sglang-omni-bitsandbytes-loader.patch`.
- An SGLang NF4 weight-name patch is saved at `patches/sglang-nf4-weight-name.patch`.
- A T4 RMSNorm fallback patch is saved at `patches/sglang-omni-t4-rmsnorm.patch`.
- A T4 AR dtype patch is saved at `patches/sglang-omni-t4-ar-dtype.patch`.
- A T4 attention dtype patch is saved at `patches/sglang-omni-t4-attention-dtype.patch`.
- A T4 Triton attention patch is saved at `patches/sglang-omni-t4-triton-attention.patch`.
- A restricted-host SHM transport patch is saved at `patches/sglang-omni-shm-transport.patch`.
- The patched focused suite passes: `54 passed, 15 warnings`.
- The loader regression test passes with its neighboring builder test: `2 passed`.
- Both Music3 BitsAndBytes regressions pass after filtering audio keys before quantization: `2 passed`.
- A tiny on-GPU NF4 iterator probe passes and returns CUDA `uint8` weights under the expected `.weight` name.
- A T4 native RMSNorm probe passes with maximum absolute error `0.0`.
- The T4/Ampere AR dtype selection and BitsAndBytes regressions pass: `4 passed`.
- A T4 attention probe confirms BF16 input is normalized to FP16 before QKV and returns FP16 Q/K/V.
- T4/Ampere dtype/backend selection plus BitsAndBytes regressions pass: `6 passed`.
- CUDA IPC default, SHM override, and invalid transport rejection are validated.
- No model or download process was running after the VS Code tunnel dropped.
- Final deliverables are `README.md`, `kaggle_cells.py`, and `kaggle_setup.py`.
- End-to-end validation passed with Triton attention and SHM relay: HTTP 200, stereo 32 kHz 16-bit WAV, 9.996 seconds, 1,279,560 bytes.

## Verified checkpoint sizes

- Qwen BF16 tensors: 18,461,001,728 bytes total.
- Audio adapter tensors filtered from the base SGLang load: 1,292,050,432 bytes.
- Remaining backbone, token embeddings, and LM head: 17,168,951,296 bytes before KV cache and CUDA workspaces.
- `flowmatching_vae.pth`: 9,828,468,476 bytes; loaded directly on GPU 1.
- `dav.pth`: 491,817,450 bytes; loaded directly on GPU 1.

## Validated runtime

The unquantized AR stage cannot fit on a 15,360 MiB T4, so the validated setup uses BitsAndBytes 0.49.2 NF4 with FP16 compute on GPU 0. GPU 1 runs the FP32 acoustic stage. `--mem-fraction-static 0.70` allocates 26,781 KV tokens and leaves room for adapters and CUDA graphs. Triton attention avoids T4 FlashInfer limits. `SGLANG_OMNI_INTRA_NODE_TRANSPORT=shm` avoids Kaggle's blocked `pidfd_getfd` syscall and stages only streamed hidden chunks. Model weights remain GPU-resident and `--cpu-offload-gb 0` is explicit.

The final setup patch sequence was also applied to a fresh detached checkout at the pinned SGLang revision, and all touched Python modules compiled successfully.

The upstream `load_audio_state` helper staged about 1.29 GB of adapter tensors in CPU RAM. The saved patch changes safetensors loading to target the AR model's CUDA device directly and passes the focused tests.

## Useful commands

```bash
nvidia-smi
git -C /root/sglang-omni status --short --branch
git -C /root/minimax-music-3-kaggle status --short --branch
find /root/minimax-music3-model/qwen_7B/qwen_7B -name 'model-*.safetensors' | wc -l
```

Do not load the full checkpoint into CPU RAM. Use safetensors metadata inspection or direct CUDA loading only.