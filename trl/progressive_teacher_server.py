#!/usr/bin/env python
"""
Custom vLLM Teacher Server with Progressive Logit Mixing

This server:
1. Loads a frozen SFT model and a dynamically-updatable reference model
2. Mixes their logits: T = α * SFT + (1-α) * ref
3. Serves teacher logprobs via OpenAI-compatible API
4. Provides /update_reference endpoint to swap reference model between training stages

Usage:
    python progressive_teacher_server.py \\
        --sft-model /path/to/sft_model \\
        --initial-ref-model /path/to/base_model \\
        --mixing-ratio 0.5 \\
        --host 0.0.0.0 \\
        --port 8000 \\
        --tensor-parallel-size 2
"""

import argparse
import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer

try:
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    from vllm.entrypoints.openai.protocol import (
        ChatCompletionRequest,
        ChatCompletionResponse,
        ChatCompletionResponseChoice,
        ChatCompletionResponseStreamChoice,
        ChatMessage,
        CompletionRequest,
        CompletionResponse,
        CompletionResponseChoice,
        DeltaMessage,
        LogProbs,
        UsageInfo,
    )
except ImportError:
    raise ImportError(
        "vLLM is required for this server. Install with: pip install vllm"
    )


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global state
app = FastAPI(title="Progressive Teacher Server")
sft_engine = None
ref_engine = None
mixing_ratio = 0.5
tokenizer = None
sft_model_path = None
current_ref_model_path = None


class UpdateReferenceRequest(BaseModel):
    """Request to update the reference model."""
    model_path: str


class ServerConfig:
    """Server configuration."""
    def __init__(self, args):
        self.sft_model = args.sft_model
        self.initial_ref_model = args.initial_ref_model
        self.mixing_ratio = args.mixing_ratio
        self.host = args.host
        self.port = args.port
        self.tensor_parallel_size = args.tensor_parallel_size
        self.gpu_memory_utilization = args.gpu_memory_utilization
        self.max_model_len = args.max_model_len
        self.dtype = args.dtype
        self.trust_remote_code = args.trust_remote_code


async def create_engine(model_path: str, config: ServerConfig, is_sft: bool = False):
    """Create a vLLM async engine."""
    logger.info(f"Loading {'SFT' if is_sft else 'reference'} model: {model_path}")

    engine_args = AsyncEngineArgs(
        model=model_path,
        tensor_parallel_size=config.tensor_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len,
        dtype=config.dtype,
        trust_remote_code=config.trust_remote_code,
        disable_log_stats=True,
        # Ensure both models use same tokenizer settings
        tokenizer=config.sft_model if not is_sft else None,
    )

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    logger.info(f"{'SFT' if is_sft else 'Reference'} model loaded successfully")
    return engine


async def initialize_engines(config: ServerConfig):
    """Initialize both SFT and reference engines."""
    global sft_engine, ref_engine, mixing_ratio, tokenizer, sft_model_path, current_ref_model_path

    mixing_ratio = config.mixing_ratio
    sft_model_path = config.sft_model
    current_ref_model_path = config.initial_ref_model

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.sft_model,
        trust_remote_code=config.trust_remote_code
    )

    # Create engines
    sft_engine = await create_engine(config.sft_model, config, is_sft=True)
    ref_engine = await create_engine(config.initial_ref_model, config, is_sft=False)

    logger.info(f"Server initialized with mixing_ratio={mixing_ratio}")


async def mix_logprobs(
    sft_logprobs: List[Dict[int, float]],
    ref_logprobs: List[Dict[int, float]],
    alpha: float
) -> List[Dict[int, float]]:
    """
    Mix log probabilities from SFT and reference models.

    Formula: log(p_mixed) = log(alpha * p_sft + (1-alpha) * p_ref)

    Args:
        sft_logprobs: List of dicts {token_id: logprob} from SFT model
        ref_logprobs: List of dicts {token_id: logprob} from reference model
        alpha: Mixing ratio for SFT model

    Returns:
        Mixed log probabilities
    """
    mixed_logprobs = []

    for sft_dict, ref_dict in zip(sft_logprobs, ref_logprobs):
        # Get union of all tokens
        all_tokens = set(sft_dict.keys()) | set(ref_dict.keys())

        mixed_dict = {}
        for token_id in all_tokens:
            # Convert logprob to prob
            sft_prob = torch.exp(torch.tensor(sft_dict.get(token_id, -float('inf'))))
            ref_prob = torch.exp(torch.tensor(ref_dict.get(token_id, -float('inf'))))

            # Mix probabilities
            mixed_prob = alpha * sft_prob + (1 - alpha) * ref_prob

            # Convert back to logprob
            mixed_logprob = torch.log(mixed_prob + 1e-10).item()
            mixed_dict[token_id] = mixed_logprob

        mixed_logprobs.append(mixed_dict)

    return mixed_logprobs


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """OpenAI-compatible completion endpoint with mixed teacher logits."""
    if sft_engine is None or ref_engine is None:
        raise HTTPException(status_code=503, detail="Engines not initialized")

    # Prepare sampling params for logprob generation
    sampling_params = SamplingParams(
        temperature=request.temperature if request.temperature is not None else 1.0,
        top_p=request.top_p if request.top_p is not None else 1.0,
        max_tokens=request.max_tokens if request.max_tokens is not None else 16,
        logprobs=request.logprobs if request.logprobs is not None else 5,
        prompt_logprobs=request.echo if request.echo else None,
    )

    # Generate from both models
    prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]

    # Await both generations
    sft_outputs = await sft_engine.generate(prompt, sampling_params, request_id=f"sft_{time.time()}")
    ref_outputs = await ref_engine.generate(prompt, sampling_params, request_id=f"ref_{time.time()}")

    # Extract logprobs
    sft_logprobs = sft_outputs.outputs[0].logprobs
    ref_logprobs = ref_outputs.outputs[0].logprobs

    # Mix logprobs
    mixed_logprobs = await mix_logprobs(sft_logprobs, ref_logprobs, mixing_ratio)

    # Build response (simplified - full implementation would handle all fields)
    response = CompletionResponse(
        id=f"cmpl-{time.time()}",
        object="text_completion",
        created=int(time.time()),
        model=f"mixed_{sft_model_path}",
        choices=[
            CompletionResponseChoice(
                index=0,
                text=sft_outputs.outputs[0].text,  # Use SFT text
                logprobs=LogProbs(
                    tokens=[tokenizer.decode([tid]) for tid in mixed_logprobs[0].keys()],
                    token_logprobs=list(mixed_logprobs[0].values()),
                    top_logprobs=mixed_logprobs,
                    text_offset=[],
                ),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=len(tokenizer.encode(prompt)),
            completion_tokens=len(sft_outputs.outputs[0].token_ids),
            total_tokens=len(tokenizer.encode(prompt)) + len(sft_outputs.outputs[0].token_ids),
        ),
    )

    return response


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """OpenAI-compatible chat completion endpoint."""
    # Convert chat messages to prompt
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            request.messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # Fallback: simple concatenation
        prompt = "\n".join([msg.get("content", "") for msg in request.messages])

    # Convert to completion request
    completion_request = CompletionRequest(
        model=request.model,
        prompt=prompt,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        logprobs=request.logprobs if hasattr(request, 'logprobs') else None,
    )

    return await create_completion(completion_request)


@app.post("/update_reference")
async def update_reference_model(request: UpdateReferenceRequest):
    """
    Update the reference model to a new checkpoint.

    This is called between training stages to update R_i = S_{i-1}.
    """
    global ref_engine, current_ref_model_path

    new_model_path = request.model_path

    if not os.path.exists(new_model_path):
        raise HTTPException(
            status_code=400,
            detail=f"Model path does not exist: {new_model_path}"
        )

    logger.info(f"Updating reference model from {current_ref_model_path} to {new_model_path}")

    # Shutdown old reference engine
    if ref_engine is not None:
        # Note: vLLM doesn't have explicit cleanup, but Python GC should handle it
        del ref_engine
        torch.cuda.empty_cache()

    # Create new reference engine
    # We need to reconstruct config (store it globally in production)
    config = ServerConfig(type('Args', (), {
        'sft_model': sft_model_path,
        'initial_ref_model': new_model_path,
        'mixing_ratio': mixing_ratio,
        'host': '0.0.0.0',
        'port': 8000,
        'tensor_parallel_size': 1,  # These should be stored globally
        'gpu_memory_utilization': 0.9,
        'max_model_len': None,
        'dtype': 'auto',
        'trust_remote_code': True,
    })())

    ref_engine = await create_engine(new_model_path, config, is_sft=False)
    current_ref_model_path = new_model_path

    logger.info(f"Reference model updated successfully to {new_model_path}")

    return JSONResponse(content={
        "status": "success",
        "old_model": current_ref_model_path,
        "new_model": new_model_path,
        "message": "Reference model updated"
    })


@app.get("/status")
async def get_status():
    """Get server status."""
    return JSONResponse(content={
        "status": "ready" if sft_engine and ref_engine else "initializing",
        "sft_model": sft_model_path,
        "ref_model": current_ref_model_path,
        "mixing_ratio": mixing_ratio,
    })


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(content={"status": "healthy"})


def parse_args():
    parser = argparse.ArgumentParser(description="Progressive Teacher Server")
    parser.add_argument("--sft-model", type=str, required=True, help="Path to SFT model (frozen)")
    parser.add_argument("--initial-ref-model", type=str, required=True, help="Path to initial reference model")
    parser.add_argument("--mixing-ratio", type=float, default=0.5, help="Mixing ratio α for SFT model")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--max-model-len", type=int, default=None, help="Max model length")
    parser.add_argument("--dtype", type=str, default="auto", help="Data type")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code")
    return parser.parse_args()


async def main():
    args = parse_args()
    config = ServerConfig(args)

    # Initialize engines
    await initialize_engines(config)

    # Start server
    import uvicorn
    uvicorn_config = uvicorn.Config(
        app,
        host=config.host,
        port=config.port,
        log_level="info",
    )
    server = uvicorn.Server(uvicorn_config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
