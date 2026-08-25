# NavigateKitchen Original ACT Baseline

This run is a clean, single-task ACT baseline for RoboCasa `NavigateKitchen`.

## Fixed protocol

- Random initialization, including the ResNet-18 visual backbone.
- Native 12-dimensional, time-indexed RoboCasa action target.
- ACT chunk size 32; execute the first 8 actions at 20 Hz.
- Observation: three RGB cameras, 16-dimensional robot state, and a 13-way
  one-hot goal identifier. The goal identifier is required because the
  `NavigateKitchen` environment contains 13 distinct target descriptions.
- Four GPUs with per-GPU batch 32: global batch size 128.
- AdamW, learning rate `1e-5`, 50,000 optimizer steps, bfloat16 forward pass.
- Deterministic 450/25/25 train/validation/test episode split from seed
  `20260820`.
- Model-only checkpoints every 5,000 steps and one overwrite-in-place resume
  checkpoint. Outputs use a new run directory and do not overwrite old runs.

This is not LP-ACT: it contains no waypoint label, path parameterization,
duration prediction, rate prediction, or pretrained language/vision model.
