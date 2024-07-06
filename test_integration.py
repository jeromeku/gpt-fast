# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch._dynamo.config
import torch._inductor.config
from torch.nn.attention import SDPBackend

import profiling_utils
from device_specs import CUDADeviceSpec
from profiling_utils import (
    FlopCounterManager,
    FLOPMode,
    FlopsTimer,
    SpeedOfLightStats,
    TransformerConfig,
    total_model_params,
)

NUM_PARAMS = None
MODEL_CFG: TransformerConfig
DEVICE_SPEC: CUDADeviceSpec
SOL: SpeedOfLightStats
FLOPCOUNTER: FlopCounterManager = FlopCounterManager(depth=2)

def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet supported")


torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import Transformer
from tokenizer import get_tokenizer


def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[0, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

def prefill(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> torch.Tensor:
    # input_pos: [B, S]
    seqlen = input_pos.shape[-1]
    # seqlen = len(input_pos.reshape(-1))
    num_tokens = input_pos.numel()
    assert num_tokens == seqlen
    
    with FLOPCOUNTER.count("prefill", num_tokens=num_tokens):
        logits = model(x, input_pos)
        next_token = sample(logits, **sampling_kwargs)[0]

    # with open("prefill.txt", "w") as f:
    #     print(FLOPCOUNTER.counts["prefill"], file=f)
    FLOPCOUNTER.print_summary(labels=['prefill'])
    flops_per_token = MODEL_CFG.flops_per_token(context_len=seqlen, mode=FLOPMode.FORWARD)
    flops_total = flops_per_token * num_tokens
    mem_lat_ms = SOL.memory_latency(unit="ms")
    compute_lat_ms = SOL.compute_latency(context_len=seqlen, num_tokens=num_tokens, mode=FLOPMode.FORWARD, unit="ms")
    print(f"Flop Check, prefill: {round(flops_total / 1e9, 1)}GFLOPs")    
    print(f"Memory Latency: {round(mem_lat_ms, 3)}ms, Compute Latency: {round(compute_lat_ms, 3)}ms")
    
    return next_token
def decode_one_token(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    context_len = input_pos[-1].item()
    num_tokens = input_pos.numel()
    assert input_pos.shape[-1] == 1
    assert num_tokens == 1

    step_name = "decode-" + str(context_len) + "-" + str(num_tokens)
    with FLOPCOUNTER.count(step_name, num_tokens=num_tokens):
        logits = model(x, input_pos)
        next_token = sample(logits, **sampling_kwargs)
    # with open(f"{step_name}.txt", "w") as f:
    FLOPCOUNTER.print_summary(labels=[step_name])

    flops_per_token = MODEL_CFG.flops_per_token(context_len=context_len, mode=FLOPMode.FORWARD)
    mem_lat_ms = SOL.memory_latency(unit="ms")
    compute_lat_ms = SOL.compute_latency(context_len=context_len, num_tokens=num_tokens, mode=FLOPMode.FORWARD, unit="ms")

    print(f"FLOPS decode {num_tokens} token with context {context_len}: {round(flops_per_token * num_tokens / 1e9, 1)}GFLOPs")
    print(f"Memory Latency: {round(mem_lat_ms, 2)}ms, Compute Latency: {round(compute_lat_ms, 2)}ms")
    
    return next_token

def decode_n_tokens(model: Transformer, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, callback=lambda _: _, **sampling_kwargs):
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        with torch.nn.attention.sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]): # Actually better for Inductor to codegen attention here
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, **sampling_kwargs
            )
            input_pos += 1
            new_tokens.append(next_token.clone())
            callback(new_tokens[-1])
            new_probs.append(next_prob.clone())
            cur_token = next_token.view(1, -1)

    return new_tokens, new_probs


def model_forward(model, x, input_pos):
    return model(x, input_pos)

@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    *,
    interactive: bool = False,
    draft_model: Transformer = None,
    speculate_k: Optional[int] = 8,
    callback = lambda x: x,
    **sampling_kwargs
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(0)
    T_new = T + max_new_tokens
    max_seq_length = min(T_new, model.config.block_size)

    device, dtype = prompt.device, prompt.dtype
    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)

    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(T_new, dtype=dtype, device=device)
    empty[:T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    next_token = prefill(model, prompt.view(1, -1), input_pos, **sampling_kwargs).clone()
    seq[T] = next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    accept_counts = [0] * (speculate_k + 1)

    generated_tokens, _ = decode_n_tokens(model, next_token.view(1, -1), input_pos, max_new_tokens - 1, callback=callback, **sampling_kwargs)
    seq[T + 1:] = torch.cat(generated_tokens)

    generate_stats = {
        'accept_counts': accept_counts
    }
    return seq, generate_stats

def encode_tokens(tokenizer, string, bos=True, device=default_device):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)

def _load_model(checkpoint_path, device, precision, use_tp, use_cuda=True):
    
    with torch.device('meta'):
        model = Transformer.from_name(checkpoint_path.parent.name)

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, assign=True)

    model = model.to(device=device, dtype=precision)
    return model.eval()

B_INST, E_INST = "[INST]", "[/INST]"

def main(
    prompt: str = "Hello, my name is",
    num_samples: int = 1,
    max_new_tokens: int = 100,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_path: Path = Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"),
    device=0,
    precision=torch.bfloat16
) -> None:
    """Generates text samples based on a pre-trained Transformer model and tokenizer.
    """
    assert checkpoint_path.is_file(), checkpoint_path

    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    assert tokenizer_path.is_file(), str(tokenizer_path)

    print("Loading model ...")
    t0 = time.time()
    model = _load_model(checkpoint_path, device, precision, False)
    
    global DEVICE_SPEC
    DEVICE_SPEC = CUDADeviceSpec(dtype=precision)
    print(f"Using {DEVICE_SPEC}")
    print(f"Model Config: {model.config}")

    num_active_params = total_model_params(model, exclude_embeddings=True)
    num_params = total_model_params(model, exclude_embeddings=False)
    print(f"Active params, Total Params: {num_active_params}, {num_params}")

    global MODEL_CFG
    MODEL_CFG = TransformerConfig(num_hidden_layers=model.config.n_layer, 
                                  num_attention_heads=model.config.n_head, 
                                  num_key_value_heads=model.config.n_local_heads, 
                                  model_dtype=precision, 
                                  kv_cache_dtype=precision, 
                                  hidden_size=model.config.dim, 
                                  intermediate_size=model.config.intermediate_size,
                                  vocab_size=model.config.vocab_size,
                                  num_params=num_params, 
                                  num_active_params=num_active_params,
                                  name=model.__class__.__name__)
    
    model_size = MODEL_CFG.model_size
    print("Transformer config: ", MODEL_CFG)
    global SOL
    SOL = SpeedOfLightStats(device_spec=DEVICE_SPEC, model_config=MODEL_CFG)
        
    device_sync(device=device) # MKG
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    tokenizer = get_tokenizer(tokenizer_path, checkpoint_path)

    encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)
    prompt_length = encoded.size(0)

    torch.manual_seed(1234)
    
    global NUM_PARAMS
    NUM_PARAMS = profiling_utils.total_model_params(model)

    aggregate_metrics = {
        'tokens_per_sec': [],
        'accept_counts': [],
    }
    
    start = 0

    for i in range(start, num_samples):
        t0 = time.perf_counter()
        
        y, metrics = generate(
            model,
            encoded,
            max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        aggregate_metrics['accept_counts'].append(metrics['accept_counts'])
        
        t = time.perf_counter() - t0

        print(tokenizer.decode(y.tolist()))

        tokens_generated = y.size(0) - prompt_length
        tokens_sec = tokens_generated / t
        aggregate_metrics['tokens_per_sec'].append(tokens_sec)
        print(f"Time for inference {i + 1}: {prompt_length} prompt tokens {tokens_generated} tokens generated, {t:.02f} sec total, {tokens_sec:.02f} tokens/sec")
        print(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")

    print("==========")

    print(f"Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}")
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
    FLOPCOUNTER.print_summary()
    with open("flop_counts.json", "w") as f:
        f.write(FLOPCOUNTER.to_json())

