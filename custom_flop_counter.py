# mypy: allow-untyped-defs
from collections import defaultdict

import torch
from torch.utils._pytree import tree_map
from torch.utils.flop_counter import FlopCounterMode

aten = torch.ops.aten

class PerformanceCounterMode(FlopCounterMode):
    def __init__(self, display=False, depth=10, debug=False):
        self.debug = debug
        self.data_counts = defaultdict(lambda: defaultdict(int))
        super().__init__(display=display, depth=depth)
    
    def get_data_counts(self):
        return {k: dict(v) for k,v in self.data_counts.items()}
    
    def _get_data_sizes(self, args):
        sizes = tree_map(lambda x: x.numel() * x.element_size() if isinstance(x, torch.Tensor) else 0, args)
        if not hasattr(sizes, "__len__"):
            sizes = [sizes]
        return sizes
    
    def get_summary_flop_counts(self):
        flop_counts = self.get_flop_counts()
        return {k: sum(v.values()) for k,v in flop_counts.items()}
    
    def get_summary_data_counts(self):
        data_counts = self.get_data_counts()
        return {k: sum(v.values()) for k,v in data_counts.items()}
    
    def _count_data_movement(self, func_packet, out, args, kwargs):
        arg_sizes = self._get_data_sizes(args)
        kwargs_sizes = self._get_data_sizes(kwargs.values())
        out_sizes = self._get_data_sizes(out)
        arg_size, kwargs_size, out_size = sum(arg_sizes), sum(kwargs_sizes), sum(out_sizes)
        return arg_size, kwargs_size, out_size
    
    def _count_flops(self, func_packet, out, args, kwargs):
        if func_packet in self.flop_registry:
            flop_count_func = self.flop_registry[func_packet]
            flop_count = flop_count_func(*args, **kwargs, out_val=out)  # type: ignore[operator]
            arg_size, kwarg_size, out_size = self._count_data_movement(func_packet, out, args, kwargs)
            total_size = arg_size + kwarg_size + out_size

            for par in set(self.mod_tracker.parents):
                if self.debug:
                    print(f"Counting flops for {par}, {func_packet}: {flop_count}")
                    print(f"Counting memory counts for {par}, {func_packet}: {sum([arg_size, kwarg_size, out_size])} = {arg_size} + {kwarg_size} + {out_size}")
                self.flop_counts[par][func_packet] += flop_count
                self.data_counts[par][func_packet] += total_size
        
        return out

if __name__ == "__main__":
    import torch
    from transformers import AutoModelForCausalLM, LlamaForCausalLM
    
    model_id = "/home/ubuntu/gpt-fast-dev/checkpoints/7B"
    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 16), dtype=torch.int64, device="cuda")

    with PerformanceCounterMode(display=False, depth=10) as perf_counter:
        _ = model(input_ids)