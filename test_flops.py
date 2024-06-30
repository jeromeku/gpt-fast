import itertools
import time
from contextlib import contextmanager
from unittest.mock import patch

import torch
from torch.utils.flop_counter import FlopCounterMode
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM
from triton.testing import do_bench

from device_specs import AVAILABLE_GPU_SPECS, CUDADeviceSpec
from profiling_utils import (
    FLOPMode,
    TransformerConfig,
    compute_latency,
    memory_latency,
    total_model_params,
    flops_per_token,
    _flops_per_token_precise
)



def timeit(name):
    start = time.time()
    yield
    end = time.time()
    print(f"{name}: {end - start:.2f} s")


def test_flops(
    num_hidden_layers,
    num_attention_heads,
    num_key_value_heads,
    intermediate_size,
    hidden_size,
    vocab_size,
    dtype,
):
    batch_size = 1
    seq_len = 128
    model_config = LlamaConfig()
    model_config.num_hidden_layers = num_hidden_layers
    model_config.num_attention_heads = num_attention_heads
    model_config.num_key_value_heads = num_key_value_heads
    model_config.intermediate_size = intermediate_size
    model_config.hidden_size = hidden_size
    model_config.vocab_size = vocab_size
    print(model_config)
    model = LlamaForCausalLM(config=model_config).to(dtype)

    input_ids = torch.randint(
        0, model_config.vocab_size, (batch_size, seq_len), dtype=torch.int64
    )
    with FlopCounterMode(display=False) as m:
        _ = model(input_ids, labels=input_ids)
    ref_flops = m.get_total_flops()
    print(f"ref_flops = {ref_flops}")

    with patch("torch.cuda.get_device_name", return_value="A100"):
        device_spec = CUDADeviceSpec(device=0, bandwidth=1.555e3, vram=40e9)
        transformer_config = TransformerConfig(
            name="Llama",
            num_params=total_model_params(model, exclude_embedding=False),
            num_active_params=total_model_params(model, exclude_embedding=True),
            num_hidden_layers=model_config.num_hidden_layers,
            hidden_size=model_config.hidden_size,
            intermediate_size=model_config.intermediate_size,
            num_attention_heads=model_config.num_attention_heads,
            num_key_value_heads=model_config.num_key_value_heads,
            model_dtype=dtype,
            kv_cache_dtype=dtype,
        )

        flops_per_token = transformer_config.flops_per_token(
            context_len=seq_len, mode=FLOPMode.FORWARD
        )
        num_tokens = batch_size * seq_len
        test_flops = flops_per_token * num_tokens
        print(f"flops = {test_flops}")
        print(f"diff = {test_flops - ref_flops}")
        mem_lat = memory_latency(device_spec, transformer_config)
        compute_lat = compute_latency(
            device_spec, transformer_config, num_tokens=num_tokens, context_len=seq_len
        )

        print(f"mem_lat = {mem_lat}")
        print(f"compute_lat = {compute_lat}")
        print(f"diff = {compute_lat - mem_lat}")


TEST_CONFIG = [
    TransformerConfig(
        "llama",
        num_params=None,
        num_active_params=None,
        num_hidden_layers=2,
        hidden_size=128,
        intermediate_size=int(2.5 * 128),
        num_attention_heads=4,
        num_key_value_heads=4,
        model_dtype=torch.float16,
        kv_cache_dtype=torch.float16,
    ),
]
# test_flops(
#     num_hidden_layers=2,
#     num_attention_heads=4,
#     num_key_value_heads=4,
#     hidden_size=128,
#     intermediate_size=int(2.5 * 128),
#     vocab_size=3200,
#     dtype=torch.float16,
# )
def _test_flop_per_token(
    *, n_layers, kv_seq_len, hidden_dim, num_params=7e9, mode=FLOPMode.FORWARD, **kwargs
):
    flop_per_token = flops_per_token(
        num_params=num_params,
        n_layers=n_layers,
        kv_seq_len=kv_seq_len,
        hidden_dim=hidden_dim,
        mode=mode,
    )
    flop_check = (6 * num_params + 12 * n_layers * kv_seq_len * hidden_dim) / 3

    if mode == FLOPMode.FORWARD_BACKWARD:
        flop_check *= 3
    elif mode == FLOPMode.ACTIVATION_CHECKPOINT:
        flop_check *= 4

    assert (
        flop_per_token == flop_check
    ), f"({flop_per_token / 1e9} per token) != ({flop_check / 1e9} check)"


def _test_flop_precise(
    *,
    n_layers,
    kv_seq_len,
    hidden_dim,
    intermediate_dim,
    vocab_size,
    mode=FLOPMode.FORWARD,
    **kwargs,
):
    flop_precise = _flops_per_token_precise(
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        kv_seq_len=kv_seq_len,
        intermediate_dim=intermediate_dim,
        vocab_size=vocab_size,
        mode=mode,
        ffn_calc_type="gpt3",
    )
    flop_check = (
        24 * n_layers * hidden_dim * hidden_dim
        + 4 * n_layers * kv_seq_len * hidden_dim
        + 2 * hidden_dim * vocab_size
    )
    flop_rough = flops_per_token(
        num_params=7e9,
        n_layers=n_layers,
        kv_seq_len=kv_seq_len,
        hidden_dim=hidden_dim,
        mode=mode,
    )
    assert (
        round(flop_precise / 1e9, 1) == round(flop_check / 1e9, 1)
    ), f"({flop_precise / 1e9} per token) != ({flop_check / 1e9} check) ({flop_rough / 1e9} rough)"

_LLAMA2_CONFIG = {
    "hidden_dim": 4096,
    "intermediate_dim": 11008,
    "kv_seq_len": 4096,
    "num_attention_heads": 32,
    "n_layers": 32,
    "vocab_size": 32000,
}

if __name__ == "__main__":
    for m in FLOPMode:
        _test_flop_per_token(**_LLAMA2_CONFIG, mode=m.value)
        _test_flop_precise(**_LLAMA2_CONFIG, mode=m.value)
