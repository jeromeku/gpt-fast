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


def FLOP_per_token_precise(*, n_layers, hidden_dim, kv_seq_len, intermediate_dim, vocab_size, mode: FLOPMode = FLOPMode.FORWARD, ffn_calc_type=None):
    """
    
    FLOP_per_token precise per https://arxiv.org/abs/2205.05198
    
    Need to scale by batch_size * q_seq_len to get to FLOP_per_batch
    """
    # Attention: qkv projection + attention scores + attention over value + out projection
    # FFN: down_proj(gate_proj * up_proj)
    # Logits: MLP
    
    #Attention
    qkv_proj_flops = 3 * (2 * n_layers * hidden_dim * hidden_dim)
    attention_score_flops = 2 * n_layers * kv_seq_len * hidden_dim
    attention_over_value_flops = attention_score_flops
    out_proj_flops = 2 * n_layers * hidden_dim * hidden_dim
    
    attention_flops = qkv_proj_flops + attention_score_flops + attention_over_value_flops + out_proj_flops
    
    #FFN
    if ffn_calc_type is None:
        up_proj_flops = 2 * n_layers * hidden_dim * intermediate_dim
        gate_proj_flops = up_proj_flops
        down_proj_flops = up_proj_flops
        ffn_flops = gate_proj_flops + up_proj_flops + down_proj_flops
    else:
        ffn_flops = 2 * 2 * n_layers * hidden_dim * 4 * hidden_dim
    
    #Logits
    logits_flops = 2 * hidden_dim * vocab_size

    total_flops = attention_flops + ffn_flops + logits_flops
    
    # Same logic as FLOP_per_token
    if mode == FLOPMode.FORWARD_BACKWARD:
        total_flops *= 3
    elif mode == FLOPMode.ACTIVATION_CHECKPOINT:
        total_flops *= 4
    
    return total_flops

def FLOP_per_token(*, 
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

    # Backwards is 2 passes (need to account for activation and weight gradient FLOP)
    # 3 = 1 (forward) + 2 (backward)
    if mode == FLOPMode.FORWARD_BACKWARD:
        flop_per_token_per_pass *= 3
    # Checkpointing implicitly includes backwards then includes another pass for recomputation
    # 4 = 1 (forward) + 2 (backward) + 1 (recomputation)
    elif mode == FLOPMode.ACTIVATION_CHECKPOINT:
        flop_per_token_per_pass *= 4
    
    return flop_per_token_per_pass



LLAMA2_CONFIG = {
  "hidden_dim": 4096,
  "intermediate_dim": 11008,
  "kv_seq_len": 4096,
  "num_attention_heads": 32,
  "n_layers": 32,
  "vocab_size": 32000
}

def test_flop_per_token(*, n_layers, kv_seq_len, hidden_dim, num_params=7e9, mode=FLOPMode.FORWARD, **kwargs):
    flop_per_token = FLOP_per_token(num_params=num_params, 
                                    n_layers=n_layers, 
                                    kv_seq_len=kv_seq_len, 
                                    hidden_dim=hidden_dim, 
                                    mode=mode)
    flop_check = (6 * num_params + 12 * n_layers * kv_seq_len * hidden_dim) / 3
    
    if mode == FLOPMode.FORWARD_BACKWARD:
        flop_check *= 3
    elif mode == FLOPMode.ACTIVATION_CHECKPOINT:
        flop_check *= 4
        
    assert flop_per_token == flop_check, f"({flop_per_token / 1e9} per token) != ({flop_check / 1e9} check)"

def test_flop_precise(*, n_layers, kv_seq_len, hidden_dim, intermediate_dim, vocab_size, mode=FLOPMode.FORWARD, **kwargs):
    flop_precise = FLOP_per_token_precise(n_layers=n_layers, 
                                          hidden_dim=hidden_dim, 
                                          kv_seq_len=kv_seq_len, 
                                          intermediate_dim=intermediate_dim, 
                                          vocab_size=vocab_size, 
                                          mode=mode, ffn_calc_type="gpt3")
    flop_check = 24 * n_layers * hidden_dim * hidden_dim + 4 * n_layers * kv_seq_len * hidden_dim + 2 * hidden_dim * vocab_size
    flop_rough = FLOP_per_token(num_params=7e9, n_layers=n_layers, kv_seq_len=kv_seq_len, hidden_dim=hidden_dim, mode=mode)
    assert round(flop_precise / 1e9, 1) == round(flop_check / 1e9, 1), f"({flop_precise / 1e9} per token) != ({flop_check / 1e9} check) ({flop_rough / 1e9} rough)"
    
if __name__ == "__main__":
    test_flop_precise(**LLAMA2_CONFIG)
    # for m in FLOPMode:
    #     test_flop_per_token(**LLAMA2_CONFIG, mode=m.value)
    