from dataclasses import dataclass
from enum import Enum
import inspect
import math
from typing import Optional
from contextlib import ExitStack, contextmanager
import torch
from torch.utils.flop_counter import FlopCounterMode

import time
from device_specs import (
    DeviceSpec,
)


class FLOPMode(Enum):
    FORWARD = 1
    FORWARD_BACKWARD = 2
    ACTIVATION_CHECKPOINT = 3


_HUGGINGFACE_CAUSAL_LM_BASE_CLASSES = [
    "causallm",
    "pretrainedmodel",
    "generationmixin",
]

def convert_to_nearest_power(n: float, base=1e9, num_decimals=2):
    nearest_power = int(math.floor(math.log(n, base)))
    p = math.pow(base, nearest_power)
    return round(n / p, num_decimals)

def get_all_base_classes(object):
    return [cls.__name__.lower() for cls in inspect.getmro(object.__class__)]
# Exclude embeddings when calculating FLOP since they don't contribute to FLOP count
def total_model_params(
    model: torch.nn.Module,
    exclude_embeddings: bool = True,
    embedding_key: str = "tok_embeddings",
) -> int:
    num_params = sum(p.numel() for p in model.parameters())
    if exclude_embeddings:
        # Not the cleanest, but check if any base class of the model is in _HUGGINGFACE_CAUSAL_LM_BASE_CLASSES
        if len(set(get_all_base_classes(model)).intersection(_HUGGINGFACE_CAUSAL_LM_BASE_CLASSES)) > 0:
            num_params -= model.model.embed_tokens.weight.numel()
        elif hasattr(model, embedding_key):
            num_params -= getattr(model, embedding_key).weight.numel()
        else:
            raise ValueError(f"Could not find embedding in model {model.__class__.__name__}, please specify embedding attribute key")
    return num_params
        
@dataclass
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
    @property
    def model_size(self):
        """
        Model size in bytes
        """
        return self.num_params * self.model_dtype.itemsize

    @classmethod
    def from_model(cls, model):
        """
        Initialize config object from a HuggingFace model 
        """
        return cls(
            name=model.__class__.__name__,
            num_params=model.num_parameters(exclude_embeddings=False),
            num_active_params=model.num_parameters(exclude_embeddings=True),
            num_hidden_layers=model.config.num_hidden_layers,
            hidden_size=model.config.hidden_size,
            intermediate_size=model.config.intermediate_size,
            num_attention_heads=model.config.num_attention_heads,
            num_key_value_heads=model.config.num_key_value_heads,
            vocab_size=model.config.vocab_size,
            model_dtype=model.config.torch_dtype,
        )

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


def _roofline_breakeven_point(device: DeviceSpec):
    """
    Arithmetic intensity (FLOP / byte) transition point from
    memory-bound to compute-bound
    """
    return device.flops / device.bandwidth


def _memory_latency(device: DeviceSpec, model_config: TransformerConfig) -> float:
    """
    Memory latency: theoretical peak memory access latency
    in seconds for a given number of bytes
    """
    num_bytes = model_config.num_params * model_config.model_dtype.itemsize
    # device bandwidth is in GB/s
    bps = device.bandwidth * 1e9
    return num_bytes / bps
def _compute_latency(
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

STR_TO_UNIT = {"s": 1, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
@dataclass
class SpeedOfLightStats:
    device_spec: DeviceSpec
    model_config: TransformerConfig
    def memory_latency(self, unit="s"):
        assert unit in STR_TO_UNIT
        num_bytes = self.model_config.model_size
        # device bandwidth is in GB/s
        bytes_per_s = self.device_spec.bandwidth
        latency_in_s = num_bytes / bytes_per_s
        return latency_in_s / STR_TO_UNIT[unit]

    def compute_latency(self, context_len: int, num_tokens: int, mode: FLOPMode = FLOPMode.FORWARD, unit="s"):
        assert unit in STR_TO_UNIT
        flops_per_token = self.model_config.flops_per_token(context_len=context_len, mode=mode)
        total_FLOP = flops_per_token * num_tokens
        latency_in_s = total_FLOP / self.device_spec.flops
        return latency_in_s / STR_TO_UNIT[unit]
    
    def breakeven_tokens(self, context_len:int):
        """
        Transition point from memory-bound to compute-bound in terms of number of tokens
        
        Computed as follows:
        1) Memory latency (ms): time to load the model 
        2) Compute latency (ms / token): time to generate 1 token for a given context length
        3) Transition point (tokens): memory latency / compute latency
        """
        #Transition point in terms of arithmetic intensity (FLOP / byte)
        memory_lat = self.memory_latency()
        compute_lat = self.compute_latency(context_len=context_len, num_tokens=1, mode=FLOPMode.FORWARD)
        breakeven_tokens = memory_lat / compute_lat
        return breakeven_tokens 
        # flops_per_byte = self.device_spec.roofline_breakeven_point
        
        # # Convert to tokens / byte
        # flops_per_token = self.model_config.flops_per_token(context_len, mode=FLOPMode.FORWARD)   
        # tokens_per_byte = flops_per_byte / flops_per_token
        # return tokens_per_byte
    def __str__(self):
        return f"{self.device_spec} {self.model_config}"

class FlopsTimer:
    def __init__(self, name, depth=10, precision=1):
        self.name = name
        self.flop_counter = FlopCounterMode(display=False, depth=depth)
        self.precision = precision
    def __enter__(self):
        self.start = time.perf_counter()
        self.flop_counter.__enter__()
        return self

    def _print_exit_msg(self):
        gflops = round(self.total_flops / 1e9, self.precision)
        ms = round(self.elapsed, self.precision)
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

class FlopCounterManager(ExitStack):
    def __init__(self):
        super().__init__()
        self.counts = {}
        
    @contextmanager
    def with_label(self, label):
        flop_counter = FlopCounterMode(display=False, depth=10)
        self.enter_context(flop_counter)
        try:
            yield self
        finally:
            self.counts[label] = flop_counter.flop_counts()
    
    def get_counts(self):
        return self.counts            
