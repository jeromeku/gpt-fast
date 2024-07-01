import pytest
from contextlib import ExitStack, contextmanager
from unittest.mock import patch
import itertools
import torch
from torch.utils.flop_counter import FlopCounterMode
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM
from triton.testing import do_bench

from device_specs import AVAILABLE_GPU_SPECS, CUDADeviceSpec
from profiling_utils import (
    FLOPMode,
    TransformerConfig,
    SpeedOfLightStats,
    total_model_params,
    flops_per_token,
    _flops_per_token_precise
)

DEVICE_NAMES = ["h100 sxm"]
DTYPES = [torch.float32, torch.bfloat16]
USE_TENSORCORES = [True, False]
DEVICE_CONFIGS = itertools.product(DEVICE_NAMES, DTYPES, USE_TENSORCORES)

@contextmanager
def patch_device(device_name):
    with patch("torch.cuda.get_device_name", return_value=device_name):
        yield
@pytest.mark.parametrize("device_name, dtype, use_tensorcores", DEVICE_CONFIGS, ids=lambda x: str(x))
def test_device_spec(device_name, dtype, use_tensorcores):
    with patch_device(device_name):
        device_spec = CUDADeviceSpec(dtype=dtype, use_tensorcores=use_tensorcores)
        if dtype == torch.float32 and use_tensorcores:
            dtype = "tfloat32"
        expected_flops = AVAILABLE_GPU_SPECS[device_name][dtype]
        assert device_spec.flops == expected_flops


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
    seq_len = 100
    model_config = LlamaConfig()
    model_config.num_hidden_layers = num_hidden_layers
    model_config.num_attention_heads = num_attention_heads
    model_config.num_key_value_heads = num_key_value_heads
    model_config.intermediate_size = intermediate_size
    model_config.hidden_size = hidden_size
    model_config.vocab_size = vocab_size
    model_config.torch_dtype = dtype
    print(model_config)
    with torch.device("meta"):
        model = LlamaForCausalLM(config=model_config)

    # input_ids = torch.randint(
    #     0, model_config.vocab_size, (batch_size, seq_len), dtype=torch.int64
    # ).cuda()
    # with FlopCounterMode(display=False) as m:
    #     _ = model(input_ids, labels=input_ids)
    # ref_flops = m.get_total_flops()
    # print(f"ref_flops = {ref_flops}")

    with patch("torch.cuda.get_device_name", return_value="A100"):
        device_spec = CUDADeviceSpec(device=0, bandwidth=1.555e3, vram=40e9)
        print(f"device bandwidth: {device_spec.bandwidth}")
        print(f"device flops: {device_spec.flops}")
        transformer_config = TransformerConfig(
            name="Llama",
            num_params=total_model_params(model, exclude_embedding=False),
            num_active_params=total_model_params(model, exclude_embedding=True),
            num_hidden_layers=model_config.num_hidden_layers,
            hidden_size=model_config.hidden_size,
            intermediate_size=model_config.intermediate_size,
            num_attention_heads=model_config.num_attention_heads,
            num_key_value_heads=model_config.num_key_value_heads,
            vocab_size=model_config.vocab_size,
            model_dtype=dtype,
            kv_cache_dtype=dtype,
        )

        flops_per_token = transformer_config.flops_per_token(
            context_len=seq_len, mode=FLOPMode.FORWARD
        )
        num_tokens = batch_size * seq_len
        test_flops = flops_per_token * num_tokens
        
        print(f"flops = {test_flops}")
        print(f"time per token = {test_flops / device_spec.flops}")
        print(f"time to load model = {transformer_config.model_size / device_spec.bandwidth}")
        # print(f"diff = {test_flops - ref_flops}")
        sol = SpeedOfLightStats(device_spec, transformer_config)
        stack = ExitStack()
        stack.enter_context(patch.object(sol, "memory_latency", return_value=3.0))
        stack.enter_context(patch.object(sol, "compute_latency", return_value=1.5))
        with stack:
            print(f"Patched memory latency: {sol.memory_latency()}")
            print(f"Patched compute latency: {sol.compute_latency(context_len=seq_len, num_tokens=num_tokens)}")
            print(f"Patched token breakeven: {sol.breakeven_tokens(context_len=seq_len)}")
        unit = "us"
        mem_lat = sol.memory_latency(unit=unit)
        compute_lat = sol.compute_latency(num_tokens=num_tokens, context_len=seq_len, unit=unit)
        # model_bytes = transformer_config.model_size
        # roofline_token_point = sol.roofline_breakeven_point(context_len=seq_len)
        tokens_at_roofline = sol.breakeven_tokens(context_len=seq_len)
        print("Model size: {model_bytes}B".format(model_bytes=transformer_config.model_size))
        print(f"Tokens at roofline: {tokens_at_roofline}")
        print(f"Num tokens: {num_tokens}")
        print(f"mem_lat = {round(mem_lat, 4)}{unit}")
        print(f"compute_lat = {round(compute_lat,4)}{unit}")
        print(f"Ratio at {num_tokens} = {round(compute_lat / mem_lat, 2)}")

# TEST_CONFIG = [
#     TransformerConfig(
#         "llama",
#         num_params=None,
#         num_active_params=None,
#         num_hidden_layers=2,
#         hidden_size=128,
#         intermediate_size=int(2.5 * 128),
#         num_attention_heads=4,
#         num_key_value_heads=4,
#         model_dtype=torch.float16,
#         kv_cache_dtype=torch.float16,
#     ),
# ]
# test_flops(
#     num_hidden_layers=32,
#     num_attention_heads=32,
#     num_key_value_heads=32,
#     hidden_size=4096,
#     intermediate_size=11008,
#     vocab_size=32000,
#     dtype=torch.float16,
# )

# def _test_flop_per_token(
#     *, n_layers, kv_seq_len, hidden_dim, num_params=7e9, mode=FLOPMode.FORWARD, **kwargs
# ):
#     flop_per_token = flops_per_token(
#         num_params=num_params,
#         n_layers=n_layers,
#         kv_seq_len=kv_seq_len,
#         hidden_dim=hidden_dim,
#         mode=mode,
#     )
#     flop_check = (6 * num_params + 12 * n_layers * kv_seq_len * hidden_dim) / 3

#     if mode == FLOPMode.FORWARD_BACKWARD:
#         flop_check *= 3
#     elif mode == FLOPMode.ACTIVATION_CHECKPOINT:
#         flop_check *= 4

#     assert (
#         flop_per_token == flop_check
#     ), f"({flop_per_token / 1e9} per token) != ({flop_check / 1e9} check)"


# def _test_flop_precise(
#     *,
#     n_layers,
#     kv_seq_len,
#     hidden_dim,
#     intermediate_dim,
#     vocab_size,
#     mode=FLOPMode.FORWARD,
#     **kwargs,
# ):
#     flop_precise = _flops_per_token_precise(
#         n_layers=n_layers,
#         hidden_dim=hidden_dim,
#         kv_seq_len=kv_seq_len,
#         intermediate_dim=intermediate_dim,
#         vocab_size=vocab_size,
#         mode=mode,
#         ffn_calc_type="gpt3",
#     )
#     flop_check = (
#         24 * n_layers * hidden_dim * hidden_dim
#         + 4 * n_layers * kv_seq_len * hidden_dim
#         + 2 * hidden_dim * vocab_size
#     )
#     flop_rough = flops_per_token(
#         num_params=7e9,
#         n_layers=n_layers,
#         kv_seq_len=kv_seq_len,
#         hidden_dim=hidden_dim,
#         mode=mode,
#     )
#     assert (
#         round(flop_precise / 1e9, 1) == round(flop_check / 1e9, 1)
#     ), f"({flop_precise / 1e9} per token) != ({flop_check / 1e9} check) ({flop_rough / 1e9} rough)"

# _LLAMA2_CONFIG = {
#     "hidden_dim": 4096,
#     "intermediate_dim": 11008,
#     "kv_seq_len": 4096,
#     "num_attention_heads": 32,
#     "n_layers": 32,
#     "vocab_size": 32000,
# }

# if __name__ == "__main__":
#     for m in FLOPMode:
#         _test_flop_per_token(**_LLAMA2_CONFIG, mode=m.value)
#         _test_flop_precise(**_LLAMA2_CONFIG, mode=m.value)
