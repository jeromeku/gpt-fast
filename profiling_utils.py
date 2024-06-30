from dataclasses import dataclass
from enum import Enum
from typing import Optional
from contextlib import ExitStack
import torch
from torch.utils.flop_counter import FlopCounterMode

from transformers.models.llama.modeling_llama import LlamaForCausalLM
import time
from device_specs import (
    CUDADeviceSpec,
    DeviceSpec,
)


class FLOPMode(Enum):
    FORWARD = 1
    FORWARD_BACKWARD = 2
    ACTIVATION_CHECKPOINT = 3


# Exclude embeddings when calculating FLOP since they don't contribute to FLOP count
def total_model_params(
    model: torch.nn.Module,
    exclude_embedding: bool = True,
    embedding_key: str = "tok_embeddings",
) -> int:
    num_params = sum(p.numel() for p in model.parameters())
    if exclude_embedding:
        if isinstance(model, LlamaForCausalLM):
            num_params -= model.model.embed_tokens.weight.numel()
        else:
            num_params -= getattr(model, embedding_key).weight.numel()
    return num_params


@dataclass(frozen=True)
class TransformerConfig:
    """
    Minimal decoder transformer model config per HuggingFace
    """

    name: str
    num_params: int
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    vocab_size: int
    model_dtype: torch.dtype
    num_key_value_heads: Optional[int] = None
    kv_cache_dtype: Optional[torch.dtype] = None
    num_active_params: Optional[int] = None
    
    def __post_init__(self):
        if self.num_active_params is None:
            self.num_active_params = self.num_params
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.kv_cache_dtype is None:
            self.kv_cache_dtype = self.model_dtype
    def flops_per_token(
        self, context_len: int, mode: FLOPMode = FLOPMode.FORWARD
    ) -> float:
        return flops_per_token(
            num_active_params=self.num_active_params,
            num_hidden_layers=self.num_hidden_layers,
            context_len=context_len,
            hidden_size=self.hidden_size,
            mode=mode,
        )


@dataclass
class SOLStats:
    """
    Speed of light stats for a given device and model config

    memory: memory latency in seconds
    compute: compute latency in seconds
    device: DeviceSpec
    model_config: TransformerConfig
    """

    memory: float
    compute: float
    device: DeviceSpec
    model_config: TransformerConfig


# @dataclass(frozen=True)
# class SOL_Latency:
#     """
#     Speed of light latency numbers
#     """


#     name: str
#     memory: float
#     compute: float
#     device: CUDADevice
#     model_config: dict


def model_latency(
    num_params: int,
    transformer_config: TransformerConfig,
    model_dtype: torch.dtype,
    num_tokens: int,
    kv_seq_len: int,
    num_active_params: Optional[int] = None,
    device: int = 0,
    mode: FLOPMode = FLOPMode.FORWARD,
):
    # Calculate latencies for the model
    if torch.cuda.is_available():
        device_spec = CUDADeviceSpec(device, dtype=model_dtype)
    else:
        print("Only CUDA devices are supported")
        return None

    model_size = int(num_params * model_dtype.itemsize)
    memory_latency = model_size / device_spec.bandwidth

    # flops = floating point operations NOT floating point operations per second
    model_flops_per_token = flops_per_token(
        num_params=num_active_params,
        num_layers=transformer_config.num_hidden_layers,
        kv_seq_len=kv_seq_len,
        hidden_size=transformer_config.hidden_size,
        mode=mode,
    )
    total_model_flops = num_tokens * model_flops_per_token

    compute_latency = total_model_flops / device_spec.flops

    return SOLStats(
        memory=memory_latency,
        compute=compute_latency,
        device=device_spec,
        model_config=transformer_config,
    )
    # # Calculate latencies for kv cache
    # kv_numel = kvcache(
    #     num_layers=model.num_hidden_layers,
    #     hidden_size=model.hidden_size,
    #     num_attention_heads=model.num_attention_heads,
    #     num_key_value_heads=model.num_key_value_heads,
    #     seq_len=kv_seq_len,
    #     batch_size=batch_size,
    # )

    # kv_bytes = kv_numel * kv_cache_dtype.itemsize
    # kv_ops = (
    #     kv_numel * (model.num_attention_heads // model.num_key_value_heads) * 2
    # )  # assume FMA per parameter
    # kv_memory_latency = kv_bytes / gpu.bandwidth
    # kv_compute_latency = kv_ops / gpu.flops


def kvcache(
    num_layers,
    hidden_size,
    num_attention_heads,
    num_key_value_heads,
    seq_len,
    batch_size,
):
    """
    Returns number of elements in kvcache

    """
    kvdim = hidden_size // num_attention_heads * num_key_value_heads
    return batch_size * num_layers * seq_len * kvdim * 2


def _flops_per_token_precise(
    *,
    n_layers,
    hidden_dim,
    kv_seq_len,
    intermediate_dim,
    vocab_size,
    mode: FLOPMode = FLOPMode.FORWARD,
    ffn_calc_type=None,
):
    """

    flops_per_token calculation per https://arxiv.org/abs/2205.05198

    Need to scale by batch_size * q_seq_len to get to FLOP_per_batch, then divide by seconds per batch to get to FLOPs
    MFU = FLOPs / GPU_FLOPs

    Alternatively,
        token_per_second = batch_size * q_seq_len / seconds_per_batch
        theoretical_peak_throughput = theoretical_peak_matmul_throughput / model_flops
        MFU = tokens_per_second / theoretical_peak_throughput
    """
    # Attention: qkv projection + attention scores + attention over value + out projection
    # FFN: down_proj(gate_proj * up_proj)
    # Logits: MLP

    # Attention
    qkv_proj_flops = 3 * (2 * n_layers * hidden_dim * hidden_dim)
    attention_score_flops = 2 * n_layers * kv_seq_len * hidden_dim
    attention_over_value_flops = attention_score_flops
    out_proj_flops = 2 * n_layers * hidden_dim * hidden_dim

    attention_flops = (
        qkv_proj_flops
        + attention_score_flops
        + attention_over_value_flops
        + out_proj_flops
    )

    # FFN
    if ffn_calc_type is None:
        up_proj_flops = 2 * n_layers * hidden_dim * intermediate_dim
        gate_proj_flops = up_proj_flops
        down_proj_flops = up_proj_flops
        ffn_flops = gate_proj_flops + up_proj_flops + down_proj_flops
    else:
        ffn_flops = 2 * 2 * n_layers * hidden_dim * 4 * hidden_dim

    # Logits
    logits_flops = 2 * hidden_dim * vocab_size

    total_flops = attention_flops + ffn_flops + logits_flops

    # Same logic as flops_per_token
    if mode == FLOPMode.FORWARD_BACKWARD:
        total_flops *= 3
    elif mode == FLOPMode.ACTIVATION_CHECKPOINT:
        total_flops *= 4

    return total_flops


def flops_per_token(
    *,
    num_active_params: int,
    num_hidden_layers: int,
    context_len: int,
    num_attention_heads: Optional[int] = None,
    dim_per_head: Optional[int] = None,
    hidden_size: Optional[int] = None,
    mode: FLOPMode = FLOPMode.FORWARD,
) -> int:
    """
    Calculate FLOP per token

    Calculates FLOP per token based on PaLM MFU formula
    https://arxiv.org/abs/2204.02311

    Need to scale by `batch_size * q_seq_len` to get to FLOP_per_batch, then divide by seconds per iteration (or batch) to get to FLOPs
    ```
        num_tokens = batch_size * q_seq_len
        MFU = (num_tokens * flops_per_token) / GPU_FLOPs
    ```

    Alternatively,
    ```
        num_tokens_per_batch = batch_size * q_seq_len
        token_per_second = num_tokens_per_batch / seconds_per_batch
        theoretical_peak_throughput = GPU_FLOPs / model_flops
        MFU = tokens_per_second / theoretical_peak_throughput
    ```
    where `GPU_FLOPs` is the theoretical peak GPU (tensor-core) FLOPs for the chip

    Notes:
    - I separated seq_len into q_seq_len and kv_seq_len, since these differ depending on the regime:
        - During training, these are equal
        - During inference we have 2 phases (disregarding speculative decoding and other parallel decoding schemes for now):
            - pre-fill: q_seq_len and kv_seq_len are equal
            - decode: q_seq_len = 1 and kv_seq_len = context length
    - Further refinements
        - Account for logits calculation
        - Account for kv cache during inference
    """
    assert hidden_size is not None or (
        dim_per_head is not None and num_attention_heads is not None
    ), "hidden_dim or (dim_per_head and n_heads) must be provided"

    if hidden_size is None:
        hidden_size = dim_per_head * num_attention_heads

    num_params_flop_per_token = 2 * num_active_params

    # First factor of 2: attention scores + attention over values
    # 2nd factor of 2: multiply + add
    # n_layers  * (kv_seq_len * hidden_dim) = GEMM dims (excluding num_tokens -> M)
    attention_flop_per_token = 2 * 2 * num_hidden_layers * context_len * hidden_size

    flop_per_token_per_pass = num_params_flop_per_token + attention_flop_per_token

    # Backwards is 2 passes (need to account for activation and weight gradient FLOP)
    # 3 = 1 (forward) + 2 (backward)
    if mode == FLOPMode.FORWARD_BACKWARD:
        flop_per_token_per_pass *= 3
    # Checkpointing implicitly includes backwards then includes another pass for recomputation
    # 4 = 1 (forward) + 2 (backward) + 1 (recomputation)
    elif mode == FLOPMode.ACTIVATION_CHECKPOINT:
        flop_per_token_per_pass *= 4

    return flop_per_token_per_pass


def roofline_breakeven_point(device: DeviceSpec):
    """ "
    Arithmetic intensity (FLOP / byte) transition point from
    memory-bound to compute-bound
    """
    return device.flops / device.bandwidth


def memory_latency(device: DeviceSpec, model_config: TransformerConfig) -> float:
    """
    Memory latency: theoretical peak memory access latency
    in seconds for a given number of bytes
    """
    num_bytes = model_config.num_params * model_config.model_dtype.itemsize
    # device bandwidth is in GB/s
    bps = device.bandwidth * 1e9
    return num_bytes / bps


def compute_latency(
    device: DeviceSpec,
    model_config: TransformerConfig,
    num_tokens: int,
    context_len: int,
    mode: FLOPMode = FLOPMode.FORWARD,
) -> float:
    """
    Compute latency: theoretical peak compute latency in seconds for a given number of FLOPS
    """
    flops = flops_per_token(
        num_active_params=model_config.num_active_params,
        num_hidden_layers=model_config.num_hidden_layers,
        num_attention_heads=model_config.num_attention_heads,
        context_len=context_len,
        hidden_size=model_config.hidden_size,
        mode=mode,
    )
    total_flops = flops * num_tokens
    return total_flops / device.flops




class FlopsTimer:
    def __init__(self, name, depth=10):
        self.name = name
        self.flop_counter = FlopCounterMode(display=False, depth=depth)
        
    def __enter__(self):
        self.start = time.perf_counter()
        self.flop_counter.__enter__()
        return self

    def _print_exit_msg(self):
        gflops = round(self.total_flops / 1e9, 2)
        ms = round(self.elapsed, 2)
        print(f"{self.name.upper()}:  Elapsed = {ms}ms, FLOPS = {gflops}GFLOPS")

    def __exit__(self, type, value, traceback):
        self.end = time.perf_counter()
        # Convert to ms
        self.elapsed = (self.end - self.start) * 1000
        self.flop_counter.__exit__(type, value, traceback)
        self._print_exit_msg()        

    @property
    def total_flops(self):
        return self.flop_counter.get_total_flops()
    
    @property
    def flops_table(self):
        return self.flop_counter.get_table()
    
class CudaFlopsTimer(FlopsTimer):
        
    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        self.flop_counter = FlopCounterMode()
        return self

    def __exit__(self, type, value, traceback):
        self.end.record()
        torch.cuda.synchronize()
        self.elapsed = self.start.elapsed_time(self.end)
        self._print_exit_msg()        

