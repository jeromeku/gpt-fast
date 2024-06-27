import torch
from enum import Enum
from typing import Optional

class FLOPMode(Enum):
    FORWARD = 1
    FORWARD_BACKWARD = 2
    ACTIVATION_CHECKPOINT = 3
    
#Exclude embeddings when calculating FLOP using PaLM MFU formula
def total_model_params(model: torch.nn.Module, exclude_embedding: bool = True) -> int:
    num_params = sum(p.numel() for p in model.parameters())
    if exclude_embedding:
        num_params -= model.tok_embeddings.weight.numel()
    return num_params




def FLOP_per_token(
    num_params: int, 
    n_layers: int, 
    kv_seq_len: int, 
    n_heads: Optional[int] = None, 
    dim_per_head: Optional[int] = None, 
    hidden_dim: Optional[int] = None,
    mode: FLOPMode = FLOPMode.FORWARD
) -> int:
    """
    Calculate FLOP per token
    
    Calculates FLOP per token based on PaLM MFU formula
    https://arxiv.org/abs/2204.02311
    
    Notes: 
    - To scale to batch, need to multiply by `num_tokens_per_batch`, where
    `num_tokens_per_batch = batch_size * q_seq_len`
    - We separate seq_len into q_seq_len and kv_seq_len, since these differ depending on the mode
        - During training, these are equal
        - During inference we have 2-phases (excluding speculative decoding and other parallel decoding schemes):
            - pre-fill: q_seq_len and kv_seq_len are equal
            - decode: q_seq_len = 1 and kv_seq_len = context length 
    - Further refinements
        - Account for logits calculation
        - Account for kv cache during inference
    """
    assert hidden_dim is not None or (dim_per_head is not None and n_heads is not None), "hidden_dim or (dim_per_head and n_heads) must be provided"

    if hidden_dim is None:
        hidden_dim = dim_per_head * n_heads
    
    num_params_flop_per_token = 2 * num_params
    
    # First factor of 2: attention scores + attention over values
    # 2nd factor of 2: multiply + add
    # n_layers  * (kv_seq_len * hidden_dim) = GEMM dims (excluding num_tokens -> M)
    attention_flop_per_token = 2 * 2 * n_layers * kv_seq_len * hidden_dim
   
    flop_per_token_per_pass = num_params_flop_per_token + attention_flop_per_token  
    _flop_per_token_per_pass_check = (6 * num_params + 12 * n_layers * kv_seq_len * hidden_dim) / 3

    # Backwards is 2 passes (need to account for activation and weight gradient FLOP)
    # 3 = 1 (forward) + 2 (backward)
    if mode == FLOPMode.FORWARD_BACKWARD:
        flop_per_token_per_pass *= 3
    # Checkpointing implicitly includes backwards then includes another pass for recomputation
    # 4 = 1 (forward) + 2 (backward) + 1 (recomputation)
    elif mode == FLOPMode.ACTIVATION_CHECKPOINT:
        flop_per_token_per_pass *= 4
    
    return flop_per_token_per_pass