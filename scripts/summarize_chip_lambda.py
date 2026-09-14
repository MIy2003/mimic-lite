"""Summarize completed/partial probes without loading torch or contacting W&B."""
import json
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
for path in sorted(root.glob("env-*.jsonl"), key=lambda p: int(p.stem.split("-")[1])):
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    if not rows:
        continue
    metrics = root / path.stem / "metrics.jsonl"
    timing = []
    if metrics.exists():
        for line in metrics.read_text().splitlines():
            try:
                r = json.loads(line)
                if r["iteration"] >= 3:
                    timing.append(r)
            except (KeyError, json.JSONDecodeError):
                pass
    print(json.dumps({"num_envs": int(path.stem.split("-")[1]), "updates": len(rows),
                      "peak_allocated_gib": max(r["peak_allocated_mib"] for r in rows)/1024,
                      "peak_reserved_gib": max(r["peak_reserved_mib"] for r in rows)/1024,
                      "warmed_iter_seconds": statistics.median(r["performance/iter_time"] for r in timing) if timing else None,
                      "warmed_rollout_fps": statistics.median(r["performance/rollout_fps"] for r in timing) if timing else None}))
