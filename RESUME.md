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
- The patched focused suite passes: `54 passed, 15 warnings`.
- The loader regression test passes with its neighboring builder test: `2 passed`.
- Both Music3 BitsAndBytes regressions pass after filtering audio keys before quantization: `2 passed`.
- A tiny on-GPU NF4 iterator probe passes and returns CUDA `uint8` weights under the expected `.weight` name.
- A T4 native RMSNorm probe passes with maximum absolute error `0.0`.
- The T4/Ampere AR dtype selection and BitsAndBytes regressions pass: `4 passed`.
- A T4 attention probe confirms BF16 input is normalized to FP16 before QKV and returns FP16 Q/K/V.
- T4/Ampere dtype/backend selection plus BitsAndBytes regressions pass: `6 passed`.
- No model or download process was running after the VS Code tunnel dropped.

## Verified checkpoint sizes

- Qwen BF16 tensors: 18,461,001,728 bytes total.
- Audio adapter tensors filtered from the base SGLang load: 1,292,050,432 bytes.
- Remaining backbone, token embeddings, and LM head: 17,168,951,296 bytes before KV cache and CUDA workspaces.
- `flowmatching_vae.pth`: 9,828,468,476 bytes; loaded directly on GPU 1.
- `dav.pth`: 491,817,450 bytes; loaded directly on GPU 1.

## Current runtime plan

The unquantized AR stage cannot fit on a 15,360 MiB T4: its retained BF16 weights alone are about 15.99 GiB, and the backend rejects tensor parallelism. BitsAndBytes 0.49.2 is installed and its NF4 CUDA probe passed on T4. Use `--mem-fraction-static 0.70`: it allocates 26,781 KV tokens, leaves 4.34 GiB before adapter attachment, and the adapter and both CUDA graph sets attach successfully. Set both `LIBRARY_PATH` and `LD_LIBRARY_PATH` to include `/usr/local/nvidia/lib64`. The FP16 request path reaches FlashInfer prefill, but its head-128 kernel needs 65,616 bytes shared memory while T4 permits 65,536. The T4 Triton patch selects SGLang's Triton attention backend on pre-Ampere GPUs and preserves the default on Ampere+. Restart with all seven patches, `--quantization bitsandbytes --cpu-offload-gb 0`, and one AR request slot, then repeat the 250-frame request.

The upstream `load_audio_state` helper staged about 1.29 GB of adapter tensors in CPU RAM. The saved patch changes safetensors loading to target the AR model's CUDA device directly and passes the focused tests.

## Useful commands

```bash
nvidia-smi
git -C /root/sglang-omni status --short --branch
git -C /root/minimax-music-3-kaggle status --short --branch
find /root/minimax-music3-model/qwen_7B/qwen_7B -name 'model-*.safetensors' | wc -l
```

Do not load the full checkpoint into CPU RAM. Use safetensors metadata inspection or direct CUDA loading only.