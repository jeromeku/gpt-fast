import pytest
# Skip if transformers is not installed
transformers = pytest.importorskip("transformers")
LlamaConfig = transformers.models.llama.modeling_llama.LlamaConfig
LlamaForCausalLM = transformers.models.llama.modeling_llama.LlamaForCausalLM

from parameterized import parameterized
from parameterized import parameterized_class
import unittest
from contextlib import ExitStack, contextmanager
from unittest.mock import patch
from copy import deepcopy
import itertools
import torch
from torch.utils.flop_counter import FlopCounterMode

from device_specs import AVAILABLE_GPU_SPECS, CUDADeviceSpec, get_chip_name
from profiling_utils import (
    FLOPMode,
    TransformerConfig,
    SpeedOfLightStats,
    total_model_params,
    convert_to_nearest_power,
    flops_per_token,
    _flops_per_token_precise
)

DEVICE_NAMES = ["h100 sxm", "a100", "nvidia geforce rtx 4090"]
DTYPES = [torch.float32, torch.bfloat16, torch.float16]
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
        chip_name = get_chip_name(device_name)
        expected_flops = AVAILABLE_GPU_SPECS[chip_name][dtype]
        assert device_spec.flops == expected_flops

MODEL_CONFIG_KEYS = ["name", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "hidden_size", "intermediate_size", "vocab_size", "dtype"]
MODEL_CONFIGS = [("llama-7b", 32, 32, 32, 4096, 11008, 32000, torch.float16), ]

@parameterized_class([dict(zip(MODEL_CONFIG_KEYS, config)) for config in MODEL_CONFIGS])
class TestTransformerConfig(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_config = LlamaConfig(
            num_hidden_layers=cls.num_hidden_layers,
            num_attention_heads=cls.num_attention_heads,
            num_key_value_heads=cls.num_key_value_heads,
            hidden_size=cls.hidden_size,
            intermediate_size=cls.intermediate_size,
            vocab_size=cls.vocab_size,
            torch_dtype=cls.dtype
        )
        with torch.device("meta"):
            cls.model = LlamaForCausalLM(config=cls.model_config)

    def test_params_count(self):
        model = self.model
        
        # Model params check
        for should_exclude in [False, True]:
            num_params_ref = model.num_parameters(exclude_embeddings=should_exclude)
            num_params_test = total_model_params(model, exclude_embeddings=should_exclude)
            self.assertEqual(num_params_ref, num_params_test)
        
        test_config = TransformerConfig.from_model(model)
        self.assertEqual(test_config.num_params, model.num_parameters(exclude_embeddings=False))
        self.assertEqual(test_config.num_active_params, model.num_parameters(exclude_embeddings=True))
    
    def test_model_size(self):
        model = self.model
        config = TransformerConfig(
            "test",
            num_params=total_model_params(model, exclude_embeddings=False),
            num_active_params=total_model_params(model, exclude_embeddings=True),
            num_hidden_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            model_dtype=self.dtype,
            kv_cache_dtype=self.dtype,
            vocab_size=self.vocab_size
        )
        # Model size check
        model_bytes = config.model_size
        self.assertEqual(model_bytes, config.num_params * config.model_dtype.itemsize)
        
        # Test factory method
        test_config = TransformerConfig.from_model(model)
        self.assertEqual(test_config.model_size, model_bytes)
        self.assertEqual(test_config.num_params, model.num_parameters(exclude_embeddings=False))
        self.assertEqual(test_config.num_active_params, model.num_parameters(exclude_embeddings=True))
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_flops_count(self):
        model_config = deepcopy(self.model_config)
        model_config.num_hidden_layers = 2
        model = LlamaForCausalLM(model_config).to("cuda")

        batch_size = 1
        seq_len = 128
        input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_len), device="cuda")

        # FLOPs check
        model.eval()
        with FlopCounterMode(display=False) as flop_counter:
            _ = model(input_ids)
        flops_ref = flop_counter.get_total_flops()
        
        transformer_config = TransformerConfig.from_model(model)
        
        flops_per_token = transformer_config.flops_per_token(
            context_len=seq_len, mode=FLOPMode.FORWARD
        )
        num_tokens = batch_size * seq_len
        flops = flops_per_token * num_tokens
        #Round to nearest GFlops with 3 significant digits
        assert convert_to_nearest_power(flops, base=1e9, num_decimals=2) == convert_to_nearest_power(flops_ref, base=1e9, num_decimals=2)
    @parameterized.expand([("A100", 1.555e12, 40e9, 1, 128, torch.float16), ])
    def test_speed_of_light(self, device_name, bandwidth, vram, batch_size, seq_len, dtype):
        model = self.model
        
        with patch_device(device_name):
            transformer_config = TransformerConfig.from_model(model)
            device_spec = CUDADeviceSpec(device=0, bandwidth=bandwidth, vram=vram)

            flops_per_token = transformer_config.flops_per_token(
                context_len=seq_len, mode=FLOPMode.FORWARD
            )
            num_tokens = batch_size * seq_len
            test_flops = flops_per_token * num_tokens

            sol = SpeedOfLightStats(device_spec, transformer_config)
            stack = ExitStack()
            stack.enter_context(patch.object(sol, "memory_latency", return_value=3.0))
            stack.enter_context(patch.object(sol, "compute_latency", return_value=1.5))
            with stack:
                self.assertEqual(sol.memory_latency(), 3.0)
                self.assertEqual(sol.compute_latency(context_len=seq_len, num_tokens=num_tokens), 1.5)
                self.assertEqual(sol.breakeven_tokens(context_len=seq_len), sol.breakeven_tokens(context_len=seq_len))

# @pytest.mark.parametrize("num_hidden_layers, num_attention_heads, num_key_value_heads, hidden_size, intermediate_size, vocab_size, dtype", MODEL_CONFIGS, ids=lambda x: str(x))
# def test_transformer_config(num_hidden_layers, num_attention_heads, num_key_value_heads, hidden_size, intermediate_size, vocab_size, dtype):
#     model_config = LlamaConfig(num_hidden_layers=num_hidden_layers,
#                                num_attention_heads=num_attention_heads,
#                                num_key_value_heads=num_key_value_heads, 
#                                hidden_size=hidden_size, 
#                                intermediate_size=intermediate_size, 
#                                vocab_size=vocab_size,
#                                torch_dtype=dtype)

#     with torch.device("meta"):
#         model = LlamaForCausalLM(config=model_config)
        
#     # Model params check
#     for should_exclude in [False, True]:
#         num_params_ref = model.num_parameters(exclude_embeddings=should_exclude)
#         num_params_test = total_model_params(model, exclude_embeddings=should_exclude)
#         assert num_params_ref == num_params_test
    
#     test_config = TransformerConfig.from_model(model)
#     assert test_config.num_params == model.num_parameters(exclude_embeddings=False)
#     assert test_config.num_active_params == model.num_parameters(exclude_embeddings=True)
    
#     config = TransformerConfig(
#         "test",
#         num_params=total_model_params(model, exclude_embeddings=False),
#         num_active_params=total_model_params(model, exclude_embeddings=True),
#         num_hidden_layers=num_hidden_layers,
#         hidden_size=hidden_size,
#         intermediate_size=intermediate_size,
#         num_attention_heads=num_attention_heads,
#         num_key_value_heads=num_key_value_heads,
#         model_dtype=dtype,
#         kv_cache_dtype=dtype,
#         vocab_size=vocab_size
#     )
#     # Model size check
#     model_bytes = config.model_size
#     assert model_bytes == config.num_params * config.model_dtype.itemsize

# def test_flops(
#     num_hidden_layers,
#     num_attention_heads,
#     num_key_value_heads,
#     intermediate_size,
#     hidden_size,
#     vocab_size,
#     dtype,
# ):
#     batch_size = 1
#     seq_len = 100
#     model_config = LlamaConfig()
#     model_config.num_hidden_layers = num_hidden_layers
#     model_config.num_attention_heads = num_attention_heads
#     model_config.num_key_value_heads = num_key_value_heads
#     model_config.intermediate_size = intermediate_size
#     model_config.hidden_size = hidden_size
#     model_config.vocab_size = vocab_size
#     model_config.torch_dtype = dtype
#     print(model_config)
#     with torch.device("meta"):
#         model = LlamaForCausalLM(config=model_config)

#     # input_ids = torch.randint(
#     #     0, model_config.vocab_size, (batch_size, seq_len), dtype=torch.int64
#     # ).cuda()
#     # with FlopCounterMode(display=False) as m:
#     #     _ = model(input_ids, labels=input_ids)
#     # ref_flops = m.get_total_flops()
#     # print(f"ref_flops = {ref_flops}")

#     with patch("torch.cuda.get_device_name", return_value="A100"):
#         device_spec = CUDADeviceSpec(device=0, bandwidth=1.555e3, vram=40e9)
#         print(f"device bandwidth: {device_spec.bandwidth}")
#         print(f"device flops: {device_spec.flops}")
#         transformer_config = TransformerConfig(
#             name="Llama",
#             num_params=total_model_params(model, exclude_embedding=False),
#             num_active_params=total_model_params(model, exclude_embedding=True),
#             num_hidden_layers=model_config.num_hidden_layers,
#             hidden_size=model_config.hidden_size,
#             intermediate_size=model_config.intermediate_size,
#             num_attention_heads=model_config.num_attention_heads,
#             num_key_value_heads=model_config.num_key_value_heads,
#             vocab_size=model_config.vocab_size,
#             model_dtype=dtype,
#             kv_cache_dtype=dtype,
#         )

#         flops_per_token = transformer_config.flops_per_token(
#             context_len=seq_len, mode=FLOPMode.FORWARD
#         )
#         num_tokens = batch_size * seq_len
#         test_flops = flops_per_token * num_tokens
        
#         print(f"flops = {test_flops}")
#         print(f"time per token = {test_flops / device_spec.flops}")
#         print(f"time to load model = {transformer_config.model_size / device_spec.bandwidth}")
#         # print(f"diff = {test_flops - ref_flops}")
#         sol = SpeedOfLightStats(device_spec, transformer_config)
#         stack = ExitStack()
#         stack.enter_context(patch.object(sol, "memory_latency", return_value=3.0))
#         stack.enter_context(patch.object(sol, "compute_latency", return_value=1.5))
#         with stack:
#             print(f"Patched memory latency: {sol.memory_latency()}")
#             print(f"Patched compute latency: {sol.compute_latency(context_len=seq_len, num_tokens=num_tokens)}")
#             print(f"Patched token breakeven: {sol.breakeven_tokens(context_len=seq_len)}")
#         unit = "us"
#         mem_lat = sol.memory_latency(unit=unit)
#         compute_lat = sol.compute_latency(num_tokens=num_tokens, context_len=seq_len, unit=unit)
#         # model_bytes = transformer_config.model_size
#         # roofline_token_point = sol.roofline_breakeven_point(context_len=seq_len)
#         tokens_at_roofline = sol.breakeven_tokens(context_len=seq_len)
#         print("Model size: {model_bytes}B".format(model_bytes=transformer_config.model_size))
#         print(f"Tokens at roofline: {tokens_at_roofline}")
#         print(f"Num tokens: {num_tokens}")
#         print(f"mem_lat = {round(mem_lat, 4)}{unit}")
#         print(f"compute_lat = {round(compute_lat,4)}{unit}")
#         print(f"Ratio at {num_tokens} = {round(compute_lat / mem_lat, 2)}")

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
