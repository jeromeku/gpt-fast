import pytest

# Skip if transformers is not installed
transformers = pytest.importorskip("transformers")
LlamaConfig = transformers.models.llama.modeling_llama.LlamaConfig
LlamaForCausalLM = transformers.models.llama.modeling_llama.LlamaForCausalLM

import itertools
import math
import time
import unittest
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from unittest.mock import PropertyMock, patch

import torch
from parameterized import parameterized, parameterized_class
from torch.utils.flop_counter import FlopCounterMode

from device_specs import AVAILABLE_GPU_SPECS, CUDADeviceSpec, get_chip_name
from performance_counter import PerformanceCounterMode
from profiling_utils import (
    CUDAPerformanceTimer,
    FLOPMode,
    PerformanceCounterManager,
    PerformanceTimer,
    SpeedOfLightStats,
    TransformerConfig,
    convert_to_nearest_power,
    total_model_params,
)

# -------------------- Device Spec Tests ------------------- #
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
        assert device_spec.flop_per_s == expected_flops
        assert device_spec.flops_by_dtype == AVAILABLE_GPU_SPECS[chip_name]
        assert device_spec.flops_by_dtype[dtype] == expected_flops
        assert device_spec.roofline_balancepoint == expected_flops / device_spec.bandwidth
        
        with pytest.raises(AssertionError):
            device_spec.flop_per_s = None
            print(device_spec.roofline_balancepoint)
        # Prevent setting attributes not in named fields to guard against user error
        with pytest.raises(AttributeError):
            device_spec.FLOPs = None

def test_empty_device_spec():
    device_name = "fake device"
    with patch_device(device_name):
        with pytest.raises(AssertionError):
            device_spec = CUDADeviceSpec()
        
        # Ok to instantiate as long as fields are filled
        device_spec = CUDADeviceSpec(name=device_name, 
                                     flop_per_s=1.0, 
                                     bandwidth=1.0, 
                                     dtype=torch.float32, 
                                     use_tensorcores=True)
    device_name = DEVICE_NAMES[0]
    
    with patch_device(device_name):
        # All critical fields will be auto-filled except for dtype (and vram, but vram is not used for downstream calcs atm)
        device_spec = CUDADeviceSpec(dtype=torch.float32)
        
        # No dtype specified
        with pytest.raises(AssertionError):
            device_spec = CUDADeviceSpec()
        
        
# -------------------- Transformer Config Tests ------------------- #
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

# -------------------- Speed of Light Tests ------------------- #
@parameterized_class([dict(zip(MODEL_CONFIG_KEYS, config)) for config in MODEL_CONFIGS])
class TestSpeedOfLight(unittest.TestCase):
    
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
            
    @parameterized.expand([("A100", 1.555e12, 40e9, 1, 128, torch.float16), ])
    def test_balancepoint(self, device_name, bandwidth, vram, batch_size, seq_len, dtype):
        model = self.model
        
        with patch_device(device_name):
            transformer_config = TransformerConfig.from_model(model)
            device_spec = CUDADeviceSpec(device=0, bandwidth=bandwidth, vram=vram, dtype=dtype)

            flops_per_token = transformer_config.flops_per_token(
                context_len=seq_len, mode=FLOPMode.FORWARD
            )
            num_tokens = batch_size * seq_len
            flop_count = flops_per_token * num_tokens

            sol = SpeedOfLightStats(device_spec, transformer_config)
            
            # Test latencies
            m_lat_ref = transformer_config.model_size / device_spec.bandwidth
            m_lat_test = sol.memory_latency(unit="s")
            self.assertEqual(m_lat_ref, m_lat_test)
            
            c_lat_ref = flop_count / device_spec.flop_per_s 
            c_lat_test = sol.compute_latency(context_len=seq_len, num_tokens=num_tokens, unit="s")
            self.assertEqual(c_lat_ref, c_lat_test)
            
            # Test balancepoint
            # Balancepoint is the transition point from memory bound to compute bound 
            # First calculate in terms of arithmetic intensity (FLOPS / byte)
            arith_intensity = device_spec.flop_per_s / device_spec.bandwidth
            # Convert to tokens / byte
            token_intensity = arith_intensity / flops_per_token
            balancepoint_tokens_check = int(math.ceil(token_intensity * transformer_config.model_size))
            balancepoint_tokens = sol.token_balancepoint(context_len=seq_len)
            self.assertAlmostEqual(balancepoint_tokens, balancepoint_tokens_check)
            
            # Another breakeven sanity check
            # memory latency is the time it takes to load the model
            # compute latency is the time it takes to generate 1 token for a given context length
            # breakeven point is therefore the ratio of memory latency to compute latency
            stack = ExitStack()
            stack.enter_context(patch.object(sol, "memory_latency", return_value=3.0))
            stack.enter_context(patch.object(sol, "compute_latency", return_value=1.5))
            with stack:
                self.assertEqual(sol.memory_latency(), 3.0)
                self.assertEqual(sol.compute_latency(context_len=seq_len, num_tokens=num_tokens), 1.5)
                self.assertEqual(sol.token_balancepoint(context_len=seq_len), 2)
    
            # Check that when num_tokens is below token_balancepoint, memory latency > compute latency
            num_tokens = balancepoint_tokens_check - 10
            self.assertGreater(sol.memory_latency(), sol.compute_latency(context_len=seq_len, num_tokens=num_tokens))
            
            # Check that when num_tokens is above token_balancepoint, compute latency > memory latency
            num_tokens = balancepoint_tokens_check + 10
            self.assertGreater(sol.compute_latency(context_len=seq_len, num_tokens=num_tokens), sol.memory_latency())
            
            #Check that when num_tokens is equal to token_balancepoint, memory latency = compute latency
            num_tokens = balancepoint_tokens_check
            self.assertAlmostEqual(sol.memory_latency(), sol.compute_latency(context_len=seq_len, num_tokens=num_tokens), places=4)
             
    @parameterized.expand([("A100", 1.555e12, 40e9, 1, 128, torch.float16), ])
    def test_arithmetic_intensity(self, device_name, bandwidth, vram, batch_size, seq_len, dtype):
        model = self.model
        
        device_spec = CUDADeviceSpec(device=0, bandwidth=bandwidth, vram=vram, dtype=dtype)
        transformer_config = TransformerConfig.from_model(model)
        sol = SpeedOfLightStats(device_spec, transformer_config)

        # Arithmetic intensity is FLOPs / byte
        # Should be equal to the total flops (flops_per_token * num_tokens) divided by the model size
        # `calculate_flops` returns the total_flops, `model_size` returns the model size in bytes
        # We mock these values as a sanity check
        stack = ExitStack()
        stack.enter_context(patch.object(sol, "calculate_flops", return_value=1000))
        stack.enter_context(patch.object(TransformerConfig, "model_size", new_callable=PropertyMock, return_value=500))

        with stack:
            self.assertEqual(sol.arithmetic_intensity(num_tokens=seq_len, context_len=seq_len), 1000/500)
    
    @parameterized.expand([("A100", 1.555e12, 40e9, 1, 2048, torch.float16), ])
    def test_model_bandwidth_utilization(self, device_name, bandwidth, vram, batch_size, seq_len, dtype):
        model = self.model
        # Assume decode phase
        num_tokens = batch_size

        tokens_generated = 5
        total_runtime = 1
        # tokens / s
        token_throughput = tokens_generated / total_runtime
        with patch_device(device_name):
            device_spec = CUDADeviceSpec(device=0, bandwidth=bandwidth, vram=vram, dtype=dtype)
            transformer_config = TransformerConfig.from_model(model)
            sol = SpeedOfLightStats(device_spec, transformer_config)
          
            # Check that we are in the memory-bound region
            assert num_tokens < sol.token_balancepoint(context_len=seq_len)
    
            # MBU = token_throughput * model_size / device_bandwidth
            MBU_ref = token_throughput * transformer_config.model_size / device_spec.bandwidth
            MBU_test = sol.model_bandwidth_utilization(token_throughput=token_throughput, context_len=seq_len)
            self.assertEqual(MBU_ref, MBU_test)
            
    @parameterized.expand([("A100", 1.555e12, 40e9, 128, 2048, torch.float16), ])
    def test_model_flops_utilization(self, device_name, bandwidth, vram, batch_size, seq_len, dtype):
        model = self.model
        # Assume compute-bound
        num_tokens = batch_size

        tokens_generated = 10
        total_runtime = 2
        # tokens / s
        token_throughput = tokens_generated / total_runtime
        
        with patch_device(device_name):
            device_spec = CUDADeviceSpec(device=0, bandwidth=bandwidth, vram=vram, dtype=dtype)
            transformer_config = TransformerConfig.from_model(model)
            sol = SpeedOfLightStats(device_spec, transformer_config)
          
            # Check that we are in the compute-bound region
            assert num_tokens > sol.token_balancepoint(context_len=seq_len)
            total_flops = sol.calculate_flops(num_tokens=num_tokens, context_len=seq_len)
            flop_per_second_achieved = total_flops / total_runtime
            MFU_ref = flop_per_second_achieved / device_spec.flop_per_s
            MFU_test = sol.model_flops_utilization(token_throughput=token_throughput, context_len=seq_len)
            self.assertEqual(MFU_ref, MFU_test)
   
# -------------------- Flop Counter Tests ------------------- #
@pytest.mark.parametrize("shape", [(1, 1024, 4096, 4096), (128, 1, 1024, 4096)], ids=lambda p: ",".join(map(str, p)))
@pytest.mark.parametrize("timer_cls", [PerformanceTimer, CUDAPerformanceTimer], ids=lambda p: p.__name__)
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=str)
def test_performance_counter_manager(shape, timer_cls, dtype):
    
    batch_size, query_len, in_features, out_features = shape
    num_tokens = batch_size * query_len
    element_size = dtype.itemsize
    a = torch.randn(num_tokens, in_features, dtype=dtype, device="cuda")
    b = torch.randn(in_features, out_features, dtype=dtype, device="cuda")
    
    cm = PerformanceCounterManager(timer_cls=timer_cls)
    start = time.perf_counter()
    with cm.count("a", num_tokens=num_tokens):
        _ = torch.matmul(a, b)
    end = time.perf_counter()
    
    elapsed = (end - start)
    expected_flops = 2 * num_tokens * in_features * out_features
    expected_io = (num_tokens * in_features + in_features * out_features + num_tokens * out_features) * element_size 
    assert cm.total_flops == expected_flops
    counts = cm.get_counts()
    assert "a" in counts
    assert abs(counts['a']['elapsed'] - elapsed) < 1e-1 # +/- 100ms
    assert counts['a']['total_flops'] == expected_flops
    assert counts['a']['total_io'] == expected_io
    assert counts['a']['token_throughput'] == counts['a']['num_tokens'] / counts['a']['elapsed']
    assert counts['a']['flops_throughput'] == counts['a']['total_flops'] / counts['a']['elapsed']
    assert counts['a']['io_throughput'] == counts['a']['total_io'] / counts['a']['elapsed']
    
    start = time.perf_counter()
    with cm.count("b", num_tokens=num_tokens):
        _ = torch.matmul(a, b)
    end = time.perf_counter()
    elapsed = end - start 
    cm.print_summary(labels=["a", "b"], verbose=True)
    assert "a" in cm.counts
    assert "b" in cm.counts
    counts = cm.counts
    assert abs(counts['b']['elapsed'] - elapsed) < 1e-1 # +/- 100ms
    assert counts['b']['total_flops'] == expected_flops
    assert counts['b']['total_io'] == expected_io
    assert cm.total_flops == 2 * expected_flops
    assert cm.total_io == 2 * expected_io
    
    summary = cm.get_summary()
    expected_tokens = 2 * num_tokens
    expected_total_flops = 2 * expected_flops
    expected_total_time = cm.total_time
    expected_token_throughput = expected_tokens / expected_total_time
    expected_flops_throughput = expected_total_flops / expected_total_time
    assert summary['total_tokens'] == expected_tokens
    assert summary['total_flops'] == expected_total_flops
    assert summary['total_time'] == expected_total_time
    assert abs(summary['token_throughput'] - expected_token_throughput) < 1e-1
    assert abs(summary['flop_throughput'] - expected_flops_throughput) < 1e-1
    cm.print_summary()
# -------------------- Performance Counter Tests ------------------- #

def get_leaf_nodes(count_keys, module_name):
    return [k for k in count_keys if k.endswith(module_name)]

def attn_proj_io_check(model_config, batch_size, seqlen, element_size):
    input_size = batch_size * seqlen * model_config.hidden_size * element_size 
    weight_size = model_config.hidden_size * model_config.hidden_size * element_size 
    output_size = batch_size * seqlen * model_config.hidden_size * element_size
    return input_size + weight_size + output_size
def attn_io_check(model_config, batch_size, seqlen, element_size):
    # queries, keys, values -> factor of 3
    input_size = (batch_size * seqlen * model_config.hidden_size * 3) * element_size
    output_size = (batch_size * seqlen * model_config.hidden_size) * element_size
    return input_size + output_size 
    
def ffn_io_check(model_config, batch_size, seqlen, element_size, module_name):
    assert module_name in ["up_proj", "gate_proj", "down_proj"]

    if module_name == "down_proj":
        input_size = batch_size * seqlen * model_config.intermediate_size * element_size
    else:
        input_size = batch_size * seqlen * model_config.hidden_size * element_size
    weight_size = model_config.hidden_size * model_config.intermediate_size * element_size 
    if module_name == "down_proj":
        output_size = batch_size * seqlen * model_config.hidden_size * element_size
    else:
        output_size = batch_size * seqlen * model_config.intermediate_size * element_size 

    return input_size + weight_size + output_size


SMALL_MODEL_CONFIG = (1, 128, 352, 4, 32000)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("num_hidden_layers, hidden_size, intermediate_size, num_attention_heads, vocab_size", [SMALL_MODEL_CONFIG])
@pytest.mark.parametrize("batch_size, seqlen", [(1, 128),])
@pytest.mark.parametrize("dtype", [torch.float16], ids=lambda p: str(p))
def test_performance_counter(num_hidden_layers, hidden_size, intermediate_size, num_attention_heads, vocab_size, batch_size, seqlen, dtype):

    cfg = LlamaConfig(num_hidden_layers=num_hidden_layers, 
                      hidden_size=hidden_size,
                      intermediate_size=intermediate_size, 
                      num_attention_heads=num_attention_heads, 
                      vocab_size=vocab_size)

    # Note we set some options manually since the model doesn't seem to be initialized correctly
    # when these options are set in LlamaConfig
    cfg._attn_implementation = "sdpa"
    model = LlamaForCausalLM(cfg).to(dtype).to("cuda")
    model_config = model.config
    element_size = dtype.itemsize
    
    input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seqlen), device="cuda")
    with torch.no_grad():
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION):
            with PerformanceCounterMode() as perf_counter:
                _ = model(input_ids)
    summary_flops = perf_counter.get_summary_flop_counts()
    summary_io = perf_counter.get_summary_io_counts()
    flops_by_op = perf_counter.get_flop_counts()
    io_by_op = perf_counter.get_io_counts()
    assert len(summary_flops) == len(summary_io)
    assert summary_flops.keys() == summary_io.keys()

    # Attn Projections
    for k in ["q_proj", "k_proj", "v_proj"]:
        # Flops check
        proj_keys = get_leaf_nodes(summary_flops.keys(), k)
        assert len(proj_keys) == model.config.num_hidden_layers
        expected_flops = 2 * batch_size * seqlen * model_config.hidden_size * model_config.hidden_size
        assert expected_flops == summary_flops[proj_keys[0]]
        
        # io movement check
        expected_size = attn_proj_io_check(model_config, batch_size, seqlen, element_size)
        assert expected_size == summary_io[proj_keys[0]]

    # Attention
    attention_keys = get_leaf_nodes(summary_flops.keys(), "self_attn")
    for k in attention_keys:
        flops = flops_by_op[k]
        io_movement = io_by_op[k]
        for op, count in flops.items():
            if "attention" in op.__name__: 
                expected_flops = 2 * 2 * batch_size * seqlen * seqlen * model_config.hidden_size
                assert expected_flops == count
        for op, count in io_movement.items():
            if "attention" in op.__name__:
                # Check approx equal due to other small artifacts returned by sdpa.mem_efficient_attention
                # See #https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/transformers/cuda/attention.cu#L867
                # Check within 100 bytes
                expected_size = attn_io_check(model_config, batch_size, seqlen, element_size)
                assert abs(expected_size - count) < 100
    # FFN
    for k in ["up_proj", "gate_proj", "down_proj"]:
        proj_keys = get_leaf_nodes(summary_flops.keys(), k)
        assert len(proj_keys) == model.config.num_hidden_layers
        expected_flops = 2 * batch_size * seqlen * model_config.hidden_size * model_config.intermediate_size
        assert expected_flops == summary_flops[proj_keys[0]]
        
        # io movement check
        expected_size = ffn_io_check(model_config, batch_size, seqlen, element_size, k)
        assert expected_size == summary_io[proj_keys[0]]
