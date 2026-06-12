#!/usr/bin/env python
"""
Progressive Teacher Proxy.

Fixes the GPU-placement bug by running each model in its OWN process, pinned to
one GPU at launch via CUDA_VISIBLE_DEVICES (the only place that env var reliably
works). This proxy itself never initializes CUDA — it only forwards HTTP requests
to two `vllm serve` backends and mixes their logprobs:

    GPU `sft_gpu`:  vllm serve <SFT>  -> sft_port   (frozen)
    GPU `ref_gpu`:  vllm serve <ref>  -> ref_port   (swapped by /update_reference)
                         proxy (this process) -> --port, mixes the two

Teacher logits:  T = alpha * SFT + (1 - alpha) * ref
"""

import argparse
import asyncio
import logging
import os
import subprocess
import time
from typing import Optional

import httpx
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Progressive Teacher Proxy")

# Global state
sft_proc: Optional[subprocess.Popen] = None
ref_proc: Optional[subprocess.Popen] = None
config = None
client: Optional[httpx.AsyncClient] = None
ref_model_path = None


class UpdateReferenceRequest(BaseModel):
    model_path: str


def spawn_vllm_server(model_path: str, gpu_id: int, port: int, gpu_memory_util: float) -> subprocess.Popen:
    """Launch `vllm serve` in its own process, pinned to one GPU.

    CUDA_VISIBLE_DEVICES is set in the child's environment BEFORE it starts, so
    the child sees exactly one GPU. This is the only reliable way to place a
    vLLM engine on a specific device.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        "vllm", "serve", model_path,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory_util),
        "--tensor-parallel-size", "1",
        "--trust-remote-code",
        "--disable-log-requests",
    ]
    logger.info(f"Spawning vllm serve on GPU {gpu_id}, port {port}: {model_path}")
    return subprocess.Popen(cmd, env=env)


async def wait_for_server(port: int, timeout: float = 600.0):
    """Poll a vllm serve backend until its /health returns 200."""
    url = f"http://127.0.0.1:{port}/health"
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            r = await client.get(url, timeout=5.0)
            if r.status_code == 200:
                logger.info(f"Backend on port {port} is ready")
                return
        except Exception:
            pass
        await asyncio.sleep(3.0)
    raise RuntimeError(f"vllm serve on port {port} did not become ready within {timeout}s")


def mix_top_logprobs(sft_lps: list, ref_lps: list, alpha: float) -> list:
    """Mix two lists of per-position {token_id: logprob} dicts in probability space.

    T = alpha * P_sft + (1 - alpha) * P_ref, returned back in log space.
    Missing tokens in either distribution are floored at logprob -30.
    """
    mixed = []
    for sft_d, ref_d in zip(sft_lps, ref_lps):
        if sft_d is None or ref_d is None:
            mixed.append(sft_d if sft_d else ref_d)
            continue
        all_tokens = set(sft_d.keys()) | set(ref_d.keys())
        out = {}
        for tid in all_tokens:
            p_sft = torch.exp(torch.tensor(sft_d.get(tid, -30.0)))
            p_ref = torch.exp(torch.tensor(ref_d.get(tid, -30.0)))
            out[tid] = torch.log(alpha * p_sft + (1 - alpha) * p_ref + 1e-10).item()
        mixed.append(out)
    return mixed


def to_logprob_dicts(choice_logprobs: dict) -> list:
    """Convert a vllm /v1/completions logprobs payload into a list of
    {token_id: logprob} dicts, one per generated position.

    vllm returns `top_logprobs` as a list of {token_str: logprob}. We key by the
    token string here; both backends share the same tokenizer (same base model),
    so the keys are directly comparable for mixing.
    """
    top = choice_logprobs.get("top_logprobs") or []
    return [dict(pos) if pos else {} for pos in top]


@app.post("/v1/completions")
async def create_completion(request: dict):
    """Forward the prompt to both backends, then return SFT text with mixed logprobs."""
    if sft_proc is None or ref_proc is None:
        raise HTTPException(status_code=503, detail="Backends not initialized")

    # Force logprobs on so we can mix them.
    payload = dict(request)
    payload.setdefault("logprobs", 20)
    payload["model"] = config.sft_model  # backend ignores, but field is required

    sft_url = f"http://127.0.0.1:{config.sft_port}/v1/completions"
    ref_payload = dict(payload)
    ref_payload["model"] = ref_model_path
    ref_url = f"http://127.0.0.1:{config.ref_port}/v1/completions"

    # Query both backends concurrently.
    sft_r, ref_r = await asyncio.gather(
        client.post(sft_url, json=payload, timeout=300.0),
        client.post(ref_url, json=ref_payload, timeout=300.0),
    )
    sft_r.raise_for_status()
    ref_r.raise_for_status()
    sft_json = sft_r.json()
    ref_json = ref_r.json()

    sft_choice = sft_json["choices"][0]
    ref_choice = ref_json["choices"][0]

    sft_lps = to_logprob_dicts(sft_choice.get("logprobs") or {})
    ref_lps = to_logprob_dicts(ref_choice.get("logprobs") or {})
    mixed = mix_top_logprobs(sft_lps, ref_lps, config.mixing_ratio)

    # Return SFT's text/tokens, but with mixed top_logprobs.
    out = dict(sft_json)
    out["choices"][0]["logprobs"]["top_logprobs"] = mixed
    return out


@app.post("/update_reference")
async def update_reference_model(request: UpdateReferenceRequest):
    """Restart ONLY the reference backend on its GPU with a new checkpoint.

    The SFT backend is never touched. Because the ref backend is its own process
    pinned to `ref_gpu`, killing and respawning it fully frees that GPU's memory
    before the new model loads — no accumulation, no cross-GPU leakage.
    """
    global ref_proc, ref_model_path

    if not os.path.exists(request.model_path):
        raise HTTPException(status_code=400, detail=f"Path not found: {request.model_path}")

    logger.info(f"Updating reference model -> {request.model_path}")

    # Kill the old ref backend and wait for the GPU to be released.
    if ref_proc is not None:
        ref_proc.terminate()
        try:
            ref_proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            ref_proc.kill()
            ref_proc.wait()

    # Spawn the new ref backend on the same GPU/port.
    ref_proc = spawn_vllm_server(
        request.model_path, config.ref_gpu, config.ref_port, config.gpu_memory_utilization
    )
    await wait_for_server(config.ref_port)
    ref_model_path = request.model_path

    logger.info("Reference model updated")
    return {"status": "success", "new_model": request.model_path}


@app.get("/status")
async def get_status():
    ready = (
        sft_proc is not None and sft_proc.poll() is None
        and ref_proc is not None and ref_proc.poll() is None
    )
    return {
        "status": "ready" if ready else "initializing",
        "mixing_ratio": config.mixing_ratio if config else None,
        "sft_model": config.sft_model if config else None,
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
    parser.add_argument("--port", type=int, default=8001, help="Proxy port (training connects here)")
    parser.add_argument("--sft-gpu", type=int, default=0, help="GPU id for the frozen SFT backend")
    parser.add_argument("--ref-gpu", type=int, default=1, help="GPU id for the reference backend")
    parser.add_argument("--sft-port", type=int, default=8002, help="Internal port for SFT backend")
    parser.add_argument("--ref-port", type=int, default=8003, help="Internal port for ref backend")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    return parser.parse_args()


async def main():
    global config, client, sft_proc, ref_proc, ref_model_path
    config = parse_args()
    ref_model_path = config.initial_ref_model
    client = httpx.AsyncClient()

    # Each backend is its own process pinned to one GPU at launch.
    sft_proc = spawn_vllm_server(config.sft_model, config.sft_gpu, config.sft_port, config.gpu_memory_utilization)
    ref_proc = spawn_vllm_server(config.initial_ref_model, config.ref_gpu, config.ref_port, config.gpu_memory_utilization)

    await asyncio.gather(
        wait_for_server(config.sft_port),
        wait_for_server(config.ref_port),
    )
    logger.info(f"Both backends ready. Proxy listening on {config.host}:{config.port}, mixing_ratio={config.mixing_ratio}")

    import uvicorn
    uvicorn_config = uvicorn.Config(app, host=config.host, port=config.port, log_level="info")
    server = uvicorn.Server(uvicorn_config)
    try:
        await server.serve()
    finally:
        for p in (sft_proc, ref_proc):
            if p is not None:
                p.terminate()
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())





