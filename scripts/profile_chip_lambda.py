"""Single-GPU memory/throughput probe; does not resume or contact W&B."""
import json
import math
import os
from pathlib import Path
import runpy
import torch
from mimic_lite_learning.ppo import PPOPolicy

original = PPOPolicy.train_op
iteration = 0


def measured(self, *args, **kwargs):
    global iteration
    result = original(self, *args, **kwargs)
    record = {"iteration": iteration,
              "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
              "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
    for k, value in result.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            value = value.detach().item()
        if isinstance(value, (int, float)):
            record[k] = value
    if not all(math.isfinite(v) for v in record.values()):
        raise FloatingPointError("Nonfinite PPO metrics")
    with Path(os.environ["CHIP_PROFILE_JSONL"]).open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print("CHIP_PROFILE", json.dumps(record), flush=True)
    iteration += 1
    torch.cuda.reset_peak_memory_stats()
    return result


PPOPolicy.train_op = measured
runpy.run_path(str(Path(__file__).with_name("train.py")), run_name="__main__")
