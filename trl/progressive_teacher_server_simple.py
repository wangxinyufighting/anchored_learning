#!/usr/bin/env python
"""
Simplified Progressive Teacher Server for vLLM 0.10+

This is a minimal implementation that works with newer vLLM versions.
For production use, consider using vLLM's built-in OpenAI server with custom logit processors.
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


class UpdateReferenceRequest(BaseModel):
    model_path: str


async def create_engine(model_path: str, tensor_parallel_size: int, gpu_memory_util: float, disable_custom_all_reduce: bool = False):
    logger.info(f"Loading model: {model_path}")
    engine_args = AsyncEngineArgs(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_util,
        trust_remote_code=True,
        disable_log_stats=True,
        disable_custom_all_reduce=disable_custom_all_reduce,
    )
    return AsyncLLMEngine.from_engine_args(engine_args)


async def initialize_engines(args):
    global sft_engine, ref_engine, mixing_ratio, tokenizer, config

    mixing_ratio = args.mixing_ratio
    config = args

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model, trust_remote_code=True)

    sft_engine = await create_engine(args.sft_model, args.tensor_parallel_size, args.gpu_memory_utilization, args.disable_custom_all_reduce)
    ref_engine = await create_engine(args.initial_ref_model, args.tensor_parallel_size, args.gpu_memory_utilization, args.disable_custom_all_reduce)

    logger.info(f"Server ready with mixing_ratio={mixing_ratio}")


async def mix_logprobs(sft_logprobs: List, ref_logprobs: List, alpha: float) -> List:
    """Mix logprobs from SFT and reference models."""
    mixed = []
    for sft_lp, ref_lp in zip(sft_logprobs, ref_logprobs):
        if sft_lp is None or ref_lp is None:
            mixed.append(sft_lp if sft_lp else ref_lp)
            continue

        # Get all tokens
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
    """Simplified completion endpoint."""
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

    # Mix logprobs
    sft_output = sft_result.outputs[0]
    ref_output = ref_result.outputs[0]

    mixed_logprobs = await mix_logprobs(sft_output.logprobs, ref_output.logprobs, mixing_ratio)

    # Return simple response
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
    """Update reference model."""
    global ref_engine

    if not os.path.exists(request.model_path):
        raise HTTPException(status_code=400, detail=f"Path not found: {request.model_path}")

    logger.info(f"Updating reference model to: {request.model_path}")

    # Properly shutdown old engine
    old_engine = ref_engine
    ref_engine = None  # Clear global first

    if old_engine is not None:
        try:
            # vLLM doesn't have explicit shutdown, but we can try to cleanup
            del old_engine
        except Exception as e:
            logger.warning(f"Error cleaning up old engine: {e}")

    # Force GPU memory cleanup
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Create new engine
    ref_engine = await create_engine(
        request.model_path,
        config.tensor_parallel_size,
        config.gpu_memory_utilization,
        config.disable_custom_all_reduce
    )

    logger.info("Reference model updated successfully")
    return {"status": "success", "new_model": request.model_path}


@app.get("/status")
async def get_status():
    return {
        "status": "ready" if sft_engine and ref_engine else "initializing",
        "mixing_ratio": mixing_ratio,
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
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--disable-custom-all-reduce", action="store_true", help="Disable custom all-reduce (fixes flash attention issues)")
    return parser.parse_args()


async def main():
    args = parse_args()
    await initialize_engines(args)

    import uvicorn
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
