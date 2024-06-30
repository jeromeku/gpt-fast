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
)


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.end = time.perf_counter()

    @property
    def elapsed(self):
        return self.end - self.start


class CudaTimer:
    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        return self

    def __exit__(self, type, value, traceback):
        self.end.record()
        torch.cuda.synchronize()
        self.elapsed = self.start.elapsed_time(self.end)

    @property
    def elapsed(self):
        return self.elapsed


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
test_flops(
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    hidden_size=128,
    intermediate_size=int(2.5 * 128),
    vocab_size=3200,
    dtype=torch.float16,
)
