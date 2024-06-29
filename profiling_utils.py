from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch

from gpu_specs import get_bandwidth, get_chip_name, get_flops_by_dtype, get_vram


class FLOPMode(Enum):
    FORWARD = 1
    FORWARD_BACKWARD = 2
    ACTIVATION_CHECKPOINT = 3


# Exclude embeddings when calculating FLOP since they don't contribute to FLOP count
def total_model_params(model: torch.nn.Module, exclude_embedding: bool = True) -> int:
    num_params = sum(p.numel() for p in model.parameters())
    if exclude_embedding:
        num_params -= model.tok_embeddings.weight.numel()
    return num_params


@dataclass
class DeviceSpec:
    """
    Abstract device specs for theoretical peak performance calculations.

    Fields will be auto-populated in __post_init__ if not already specified
    and if data is available
    - bandwidth (GB/s)
    - flops (FLOP / s)
    - vram (bytes)
    - dtype (torch.dtype) dtype used for theoretical peak performance
    """

    name: Optional[str] = None
    bandwidth: Optional[int] = None
    flops: Optional[int] = None
    vram: Optional[int] = None
    dtype: Optional[torch.dtype] = torch.float32

    def _post_init_check(self):
        if self.bandwidth is None:
            print(
                "GPU bandwidth is None - please specify the bandwidth in GB/s in order to enable SOL calculations"
            )

        if self.flops is None:
            print(
                "GPU flops is None - please specify the flops in FLOP/s in order to enable SOL calculations"
            )

        if self.vram is None:
            print("GPU vram is None - please specify the vram in bytes")

    def __str__(self):
        bw = round(self.bandwidth, 1)
        tflops = round(self.flops / 1e12, 2)
        vram_GB = round(self.vram / 1e9, 1)
        return f"DeviceSpec(name={self.name}, dtype={self.dtype}, bandwidth={bw}GB/s, flops={tflops}TFLOPs, vram={vram_GB}GB)"


@dataclass
class CUDADevice(DeviceSpec):
    """
    CUDA specs for theoretical peak performance

    Fields will be auto-populated in __post_init__ if not already specified
    and if data is available

    See AVAILABLE_GPU_SPECS for a list of available chips
    """

    # Device index
    device: Optional[int] = 0
    # Use tfloat32 FLOPs for dtype == torch.float32
    use_tensorcores: bool = True

    def __post_init__(self):
        # Populate fields if not already populated
        self.name = torch.cuda.get_device_name(self.device)

        # Memory bandwidth
        if self.bandwidth is None:
            self.bandwidth = get_bandwidth()

        # FLOPs
        if self.flops is None:
            chip_name = get_chip_name(self.device)
            if chip_name is None:
                print(f"No FLOPs data available for device name {self.name}")
            else:
                flops_by_dtype = get_flops_by_dtype(chip_name)
                # Populate flops if not already populated
                if self.dtype in flops_by_dtype:
                    if self.dtype == torch.float32:
                        should_use_tf32 = (
                            self.dtype == torch.float32
                            and "tfloat32" in flops_by_dtype
                            and torch.get_float32_matmul_precision() != "highest"
                            and self.use_tensorcores
                        )
                        if should_use_tf32:
                            self.dtype = "tfloat32"
                    self.flops = flops_by_dtype[self.dtype]
                else:
                    print(
                        f"Could not find FLOPs for dtype {self.dtype} for device {self.name}"
                    )
        # Vram
        if self.vram is None:
            self.vram = get_vram()

        # Issue post check warnings
        self._post_init_check()

    def roofline_breakeven_point(self, dtype=torch.float16, tensorcore=True):
        """ "
        Arithmetic intensity (FLOP / byte) transition point from
        memory-bound to compute-bound
        """
        if tensorcore:
            return (
                self.tensorops_fp32 / self.bandwidth
                if dtype.itemsize == 4
                else self.tensorops_fp16 / self.bandwidth
            )
        return self.flops / self.bandwidth

    def memory_latency(self, num_bytes: int) -> float:
        """
        Memory latency: theoretical peak memory access latency
        in seconds for a given number of bytes
        """
        return num_bytes / self.bandwidth

    def compute_latency(self, FLOPS: int, dtype: torch.dtype, tensorcore=True) -> float:
        """
        Compute latency: theoretical peak compute latency in seconds for a given number of FLOPS
        """
        if tensorcore:
            return (
                FLOPS / self.tensorops_fp32
                if dtype.itemsize == 4
                else FLOPS / self.tensorops_fp16
            )
        return FLOPS / self.flops


@dataclass(frozen=True)
class ModelConfig:
    """
    Minimal decoder transformer model config per HuggingFace
    """

    name: str
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    model_dtype: torch.dtype
    kv_cache_dtype: Optional[torch.dtype]


@dataclass(frozen=True)
class SOL_Latency:
    """
    Speed of light latency numbers
    """

    name: str
    memory: float
    compute: float
    device: CUDADevice
    model_config: dict


def SOL_latency(
    gpu: CUDADevice,
    model: torch.nn.Module,
    q_seq_len,
    kv_seq_len: int,
    batch_size: int,
    model_dtype: torch.dtype,
    kv_cache_dtype: torch.dtype,
    tensorcores=True,
):
    # Calculate latencies for the model
    model_size = int(
        total_model_params(model, exclude_embedding=False) * model_dtype.itemsize
    )
    memory_latency = model_size / gpu.bandwidth

    active_params = total_model_params(model, exclude_embedding=True)

    # flops = floating point operations NOT floating point operations per second
    model_flops_per_token = FLOP_per_token(
        num_params=active_params,
        n_layers=model.num_hidden_layers,
        kv_seq_len=kv_seq_len,
        hidden_dim=model.hidden_size,
        mode=FLOPMode.FORWARD,
    )
    total_model_flops = batch_size * q_seq_len * model_flops_per_token

    if tensorcores:
        theoretical_FLOPs = (
            gpu.tensorops_fp32 if model_dtype.itemsize == 4 else gpu.tensorops_fp16
        )
    else:
        # FLOPS = floating point operations per second
        theoretical_FLOPs = gpu.flops

    compute_latency = total_model_flops / theoretical_FLOPs

    # Calculate latencies for kv cache
    kv_numel = kvcache(
        num_layers=model.num_hidden_layers,
        hidden_size=model.hidden_size,
        num_attention_heads=model.num_attention_heads,
        num_key_value_heads=model.num_key_value_heads,
        seq_len=kv_seq_len,
        batch_size=batch_size,
    )

    kv_bytes = kv_numel * kv_cache_dtype.itemsize
    kv_ops = (
        kv_numel * (model.num_attention_heads // model.num_key_value_heads) * 2
    )  # assume FMA per parameter
    kv_memory_latency = kv_bytes / gpu.bandwidth
    kv_compute_latency = kv_ops / gpu.flops

    return {
        "model": {"io": io_latency, "compute": compute_latency},
        "kvcache": {"io": kv_io_latency, "compute": kv_compute_latency},
    }


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


def FLOP_per_token_precise(
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

    FLOP_per_token calculation per https://arxiv.org/abs/2205.05198

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

    # Same logic as FLOP_per_token
    if mode == FLOPMode.FORWARD_BACKWARD:
        total_flops *= 3
    elif mode == FLOPMode.ACTIVATION_CHECKPOINT:
        total_flops *= 4

    return total_flops


def FLOP_per_token(
    *,
    num_params: int,
    n_layers: int,
    kv_seq_len: int,
    n_heads: Optional[int] = None,
    dim_per_head: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    mode: FLOPMode = FLOPMode.FORWARD,
) -> int:
    """
    Calculate FLOP per token

    Calculates FLOP per token based on PaLM MFU formula
    https://arxiv.org/abs/2204.02311

    Need to scale by `batch_size * q_seq_len` to get to FLOP_per_batch, then divide by seconds per iteration (or batch) to get to FLOPs
    ```
        num_tokens = batch_size * q_seq_len
        MFU = (num_tokens * FLOP_per_token) / GPU_FLOPs
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
    assert hidden_dim is not None or (
        dim_per_head is not None and n_heads is not None
    ), "hidden_dim or (dim_per_head and n_heads) must be provided"

    if hidden_dim is None:
        hidden_dim = dim_per_head * n_heads

    num_params_flop_per_token = 2 * num_params

    # First factor of 2: attention scores + attention over values
    # 2nd factor of 2: multiply + add
    # n_layers  * (kv_seq_len * hidden_dim) = GEMM dims (excluding num_tokens -> M)
    attention_flop_per_token = 2 * 2 * n_layers * kv_seq_len * hidden_dim

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


_LLAMA2_CONFIG = {
    "hidden_dim": 4096,
    "intermediate_dim": 11008,
    "kv_seq_len": 4096,
    "num_attention_heads": 32,
    "n_layers": 32,
    "vocab_size": 32000,
}


def _test_flop_per_token(
    *, n_layers, kv_seq_len, hidden_dim, num_params=7e9, mode=FLOPMode.FORWARD, **kwargs
):
    flop_per_token = FLOP_per_token(
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
    flop_precise = FLOP_per_token_precise(
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
    flop_rough = FLOP_per_token(
        num_params=7e9,
        n_layers=n_layers,
        kv_seq_len=kv_seq_len,
        hidden_dim=hidden_dim,
        mode=mode,
    )
    assert (
        round(flop_precise / 1e9, 1) == round(flop_check / 1e9, 1)
    ), f"({flop_precise / 1e9} per token) != ({flop_check / 1e9} check) ({flop_rough / 1e9} rough)"


if __name__ == "__main__":
    for m in FLOPMode:
        _test_flop_per_token(**_LLAMA2_CONFIG, mode=m.value)
        _test_flop_precise(**_LLAMA2_CONFIG, mode=m.value)
