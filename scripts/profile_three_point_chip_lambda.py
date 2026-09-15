"""Production-size Transformer PPO probe, including non-PyTorch GPU memory."""
import json
import math
import os
from pathlib import Path
import runpy
import time

import torch
from mimic_lite_learning.ppo import PPOPolicy

original = PPOPolicy.train_op
iteration = 0
last_update_end = None


def measured(self, *args, **kwargs):
    global iteration, last_update_end
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = original(self, *args, **kwargs)
    torch.cuda.synchronize()
    end = time.perf_counter()
    free, total = torch.cuda.mem_get_info()
    reserved = torch.cuda.memory_reserved()
    external = max(0, total - free - reserved)
    record = {
        "iteration": iteration,
        "device": torch.cuda.get_device_name(),
        "total_mib": total / 2**20,
        "used_mib": (total - free) / 2**20,
        "external_mib": external / 2**20,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "estimated_peak_total_mib": (external + torch.cuda.max_memory_reserved()) / 2**20,
        "training_seconds": end - start,
        "iteration_seconds": None if last_update_end is None else end - last_update_end,
    }
    for key, value in result.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            value = value.detach().item()
        if isinstance(value, (int, float)):
            if not math.isfinite(value):
                raise FloatingPointError(f"Nonfinite PPO metric: {key}")
            record[key] = value
    with Path(os.environ['THREE_POINT_PROFILE_JSONL']).open('a') as stream:
        stream.write(json.dumps(record) + '\n')
    print('THREE_POINT_PROFILE', json.dumps(record), flush=True)
    iteration += 1
    last_update_end = end
    torch.cuda.reset_peak_memory_stats()
    return result


PPOPolicy.train_op = measured
runpy.run_path(str(Path(__file__).with_name('train.py')), run_name='__main__')
