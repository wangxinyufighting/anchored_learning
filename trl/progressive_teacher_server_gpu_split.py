#!/usr/bin/env python
"""
Progressive Teacher Server - GPU-separated version
Loads SFT on one GPU, ref on another GPU to avoid OOM.
"""

import argparse
import asyncio
import json
import logging
import os
from typing import Dict, List

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Progressive Teacher Server")

# Global state
sft_engine = None
ref_engine = None
mixing_ratio = 0.5
tokenizer = None
config = None
sft_model_path = None
ref_model_path = None


class UpdateReferenceRequest(BaseModel):
    model_path: str


async def create_engine(model_path: str, gpu_id: int, gpu_memory_util: float):
    """Create engine on specific GPU."""
    logger.info(f"Loading model on GPU {gpu_id}: {model_path}")

    engine_args = AsyncEngineArgs(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_util,
        trust_remote_code=True,
        disable_log_stats=True,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    logger.info(f"Model loaded on GPU {gpu_id}")
    return engine


async def initialize_engines(args):
    global sft_engine, ref_engine, mixing_ratio, tokenizer, config, sft_model_path, ref_model_path

    mixing_ratio = args.mixing_ratio
    config = args
    sft_model_path = args.sft_model
    ref_model_path = args.initial_ref_model

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model, trust_remote_code=True)

    # Load SFT on GPU 0, ref on GPU 1
    sft_engine = await create_engine(args.sft_model, 0, args.gpu_memory_utilization)
    ref_engine = await create_engine(args.initial_ref_model, 1, args.gpu_memory_utilization)

    logger.info(f"Both engines ready with mixing_ratio={mixing_ratio}")


async def mix_logprobs(sft_logprobs: List, ref_logprobs: List, alpha: float) -> List:
    """Mix logprobs from SFT and reference models."""
    mixed = []
    for sft_lp, ref_lp in zip(sft_logprobs, ref_logprobs):
        if sft_lp is None or ref_lp is None:
            mixed.append(sft_lp if sft_lp else ref_lp)
            continue

        sft_dict = {token_id: logprob for token_id, logprob in sft_lp.items()}
        ref_dict = {token_id: logprob for token_id, logprob in ref_lp.items()}
        all_tokens = set(sft_dict.keys()) | set(ref_dict.keys())

        mixed_dict = {}
        for tid in all_tokens:
            sft_prob = torch.exp(torch.tensor(sft_dict.get(tid, -30.0)))
            ref_prob = torch.exp(torch.tensor(ref_dict.get(tid, -30.0)))
            mixed_prob = alpha * sft_prob + (1 - alpha) * ref_prob
            mixed_dict[tid] = torch.log(mixed_prob + 1e-10).item()

        mixed.append(mixed_dict)

    return mixed


@app.post("/v1/completions")
async def create_completion(request: dict):
    """Completion endpoint."""
    if sft_engine is None or ref_engine is None:
        raise HTTPException(status_code=503, detail="Engines not initialized")

    prompt = request.get("prompt", "")
    max_tokens = request.get("max_tokens", 16)
    temperature = request.get("temperature", 1.0)
    logprobs = request.get("logprobs", 5)

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        logprobs=logprobs,
    )

    # Generate from both models
    sft_result = await sft_engine.generate(prompt, sampling_params, f"sft_{id(request)}")
    ref_result = await ref_engine.generate(prompt, sampling_params, f"ref_{id(request)}")

    sft_output = sft_result.outputs[0]
    ref_output = ref_result.outputs[0]

    mixed_logprobs = await mix_logprobs(sft_output.logprobs, ref_output.logprobs, mixing_ratio)

    return {
        "id": f"cmpl-{id(request)}",
        "object": "text_completion",
        "choices": [{
            "index": 0,
            "text": sft_output.text,
            "logprobs": {
                "tokens": sft_output.token_ids,
                "token_logprobs": [lp[tid] if lp and tid in lp else -30.0
                                  for lp, tid in zip(mixed_logprobs, sft_output.token_ids)],
                "top_logprobs": mixed_logprobs,
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": len(sft_result.prompt_token_ids),
            "completion_tokens": len(sft_output.token_ids),
            "total_tokens": len(sft_result.prompt_token_ids) + len(sft_output.token_ids),
        }
    }


@app.post("/update_reference")
async def update_reference_model(request: UpdateReferenceRequest):
    """Update reference model on GPU 1."""
    global ref_engine, ref_model_path

    if not os.path.exists(request.model_path):
        raise HTTPException(status_code=400, detail=f"Path not found: {request.model_path}")

    logger.info(f"Updating reference model to: {request.model_path}")

    old_engine = ref_engine
    ref_engine = None

    if old_engine is not None:
        del old_engine

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Reload on GPU 1
    ref_engine = await create_engine(request.model_path, 1, config.gpu_memory_utilization)

    # Track current reference model
    ref_model_path = request.model_path

    logger.info("Reference model updated")
    return {"status": "success", "new_model": request.model_path}


@app.get("/status")
async def get_status():
    return {
        "status": "ready" if (sft_engine is not None and ref_engine is not None) else "initializing",
        "mixing_ratio": mixing_ratio,
        "sft_model": sft_model_path,
        "ref_model": ref_model_path,
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-model", type=str, required=True)
    parser.add_argument("--initial-ref-model", type=str, required=True)
    parser.add_argument("--mixing-ratio", type=float, default=0.5)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    return parser.parse_args()


async def main():
    args = parse_args()
    await initialize_engines(args)

    import uvicorn
    uvicorn_config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
    server = uvicorn.Server(uvicorn_config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
