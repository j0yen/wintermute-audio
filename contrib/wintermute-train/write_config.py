#!/usr/bin/env python3
"""Emit training_parameters.yaml for micro-wake-word training.

Mirrors basic_training_notebook cell 9. Feature dirs point into $OUT
(produced by gen_features.py). Steps come from $TRAIN_STEPS; $MWW_SMOKE
shrinks batch/eval cadence so a smoke run finishes in minutes.
"""
import os, yaml
from pathlib import Path

OUT = Path(os.environ["OUT"])
SMOKE = os.environ.get("MWW_SMOKE", "0") == "1"
STEPS = os.environ.get("TRAIN_STEPS", "200" if SMOKE else "5000,5000")
steps = [int(x) for x in STEPS.split(",")]
lr = [0.001] * len(steps)

neg = OUT / "negative_datasets"
features = [
    {"features_dir": str(OUT / "generated_augmented_features"),
     "sampling_weight": 2.0, "penalty_weight": 1.0, "truth": True,
     "truncation_strategy": "truncate_start", "type": "mmap"},
]
# Add whichever negative sets actually downloaded (smoke pulls only no_speech).
neg_specs = [
    ("speech", 10.0, 1.0, "random"),
    ("no_speech", 5.0, 1.0, "random"),
    ("dinner_party", 10.0, 1.0, "random"),
]
for name, sw, pw, trunc in neg_specs:
    d = neg / name
    if d.exists():
        features.append({"features_dir": str(d), "sampling_weight": sw,
                         "penalty_weight": pw, "truth": False,
                         "truncation_strategy": trunc, "type": "mmap"})
# ambient (for false-accepts/hour estimate) — use dinner_party_eval if present
amb = neg / "dinner_party_eval"
if amb.exists():
    features.append({"features_dir": str(amb), "sampling_weight": 0.0,
                     "penalty_weight": 1.0, "truth": False,
                     "truncation_strategy": "split", "type": "mmap"})

config = {
    "window_step_ms": 10,
    "train_dir": str(OUT / "trained_models" / "wakeword"),
    "features": features,
    "training_steps": steps,
    "positive_class_weight": [1],
    "negative_class_weight": [20],
    "learning_rates": lr,
    "batch_size": 24 if SMOKE else 128,
    "time_mask_max_size": [0] if SMOKE else [5],
    "time_mask_count": [0] if SMOKE else [2],
    "freq_mask_max_size": [0] if SMOKE else [5],
    "freq_mask_count": [0] if SMOKE else [2],
    "eval_step_interval": 50 if SMOKE else 500,
    # clip_duration_ms must yield a spectrogram_length divisible by the model
    # stride (3) — the int8 representative_dataset_gen asserts
    # `spectrogram.shape[0] % stride == 0`. With mixednet stride=3 + our kernel
    # sizes (slices_dropped=134), 1560ms -> spectrogram_length 186 (186%3==0).
    "clip_duration_ms": 1560,
    "target_minimization": 0.9,
    "minimization_metric": None,
    "maximization_metric": "average_viable_recall",
}

OUT.mkdir(parents=True, exist_ok=True)
p = OUT / "training_parameters.yaml"
with open(p, "w") as f:
    yaml.dump(config, f, sort_keys=False)
print("[write_config] wrote", p, "| steps:", steps, "| features:", len(features))
