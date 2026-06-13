#!/usr/bin/env python
"""
Compatibility launcher for the progressive teacher server.

The old implementation created two vLLM AsyncLLMEngine instances in the same
Python process. That cannot reliably place one engine on GPU 0 and the other on
GPU 1: vLLM/CUDA device visibility is process-scoped. This launcher keeps the
old filename and CLI, but delegates to progressive_teacher_proxy.py, which starts
one vLLM backend process per model.
"""

import argparse
import asyncio
import os
import sys

from progressive_teacher_proxy import main as proxy_main


def visible_gpu_ids() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return []
    return [gpu.strip() for gpu in visible.split(",") if gpu.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="GPU-separated progressive teacher launcher")
    parser.add_argument("--sft-model", type=str, required=True)
    parser.add_argument("--initial-ref-model", type=str, required=True)
    parser.add_argument("--mixing-ratio", type=float, default=0.5)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--sft-gpu", type=str, default=None)
    parser.add_argument("--ref-gpu", type=str, default=None)
    parser.add_argument("--sft-port", type=int, default=8002)
    parser.add_argument("--ref-port", type=int, default=8003)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    return parser.parse_args()


def infer_gpu_pair(args) -> tuple[str, str]:
    visible = visible_gpu_ids()
    sft_gpu = args.sft_gpu
    ref_gpu = args.ref_gpu

    if sft_gpu is None and visible:
        sft_gpu = visible[0]
    if ref_gpu is None and len(visible) >= 2:
        ref_gpu = visible[1]

    if sft_gpu is None:
        sft_gpu = "0"
    if ref_gpu is None:
        ref_gpu = "1"

    if sft_gpu == ref_gpu:
        raise ValueError(
            f"SFT and reference backends must use different GPUs, got {sft_gpu!r}. "
            "Set CUDA_VISIBLE_DEVICES to at least two IDs or pass --sft-gpu and --ref-gpu."
        )

    return sft_gpu, ref_gpu


def main():
    args = parse_args()
    sft_gpu, ref_gpu = infer_gpu_pair(args)

    sys.argv = [
        "progressive_teacher_proxy.py",
        "--sft-model",
        args.sft_model,
        "--initial-ref-model",
        args.initial_ref_model,
        "--mixing-ratio",
        str(args.mixing_ratio),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--sft-gpu",
        sft_gpu,
        "--ref-gpu",
        ref_gpu,
        "--sft-port",
        str(args.sft_port),
        "--ref-port",
        str(args.ref_port),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--dtype",
        args.dtype,
    ]
    asyncio.run(proxy_main())


if __name__ == "__main__":
    main()
