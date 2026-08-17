# LP-GR00T-Base V1

This directory contains the first executable reference implementation of the
project's path-time action representation. It is the exact **base-only** variant
used for the RoboCasa `NavigateKitchen` experiment with a pretrained GR00T N1.6
backbone.

It is deliberately narrower than the repository's final whole-body research
proposal:

- only the mobile-base output is learned;
- the GR00T vision-language backbone and action head are retained;
- arm, end-effector, and gripper commands are not learned by this variant;
- timing is predicted per path segment, but there is no learned online phase
  controller yet.

Consequently, this implementation is a representation and execution baseline,
not evidence that the complete time-invariant whole-body method is finished.

## Representation

For an observation at time `t`, measured world-frame base poses are converted to
one local trajectory with a common origin:

```text
P_k = inverse(T_t) * T_k = [x_rel, y_rel, yaw_rel]
```

The implementation never integrates the recorded normalized `base_motion`
command and calls it ground-truth motion. Labels come from measured base poses.

Adjacent SE(2) increments define a dimensionless path coordinate:

```text
delta_sigma = sqrt(
    (delta_x / 0.05)^2
  + (delta_y / 0.05)^2
  + (delta_yaw / 0.15)^2
)
```

The default label window ends at path extent `16.0`, after four seconds, or at
episode end. A complete window is sampled into 32 equally spaced path anchors.
Each target row is:

```text
[x_rel, y_rel, yaw_rel, log(delta_t)]
```

Incomplete windows retain reached fixed-extent anchors, add one truthful
terminal anchor, repeat the terminal value for tensor padding, and mask the
repeated padding in the training loss. Static samples are capped in the dataset
sampler rather than silently dominating training.

## Execution

At inference time, the 32 predicted path-time anchors are interpolated in SE(2)
at the 20 Hz controller timestamps. Adjacent desired poses are converted to
body-frame twists with `SE(2) Log`, then mapped to normalized RoboCasa commands
with a train-split controller calibration:

```text
observation
  -> GR00T N1.6
  -> 32 x [x_rel, y_rel, yaw_rel, log(delta_t)]
  -> SE(2) interpolation at 20 Hz
  -> calibrated normalized base commands
  -> execute 8 frames (0.4 s)
  -> observe and replan
```

The wrapper sends zero end-effector/gripper commands and selects base control
mode. This is suitable for isolating navigation behavior; it is not a valid
whole-body manipulation policy.

## Layout

```text
configs/        GR00T modality registration
gr00t_patch/    minimal GR00T dataset, mask, optimizer, and execution patches
scripts/        cache, calibration, training, validation, and evaluation tools
src/            SE(2), label generation, reconstruction, and validation code
tests/          deterministic geometry, labels, execution, and v3 dataset tests
supervisor/     recorded Vast.ai service configurations from the experiment
```

## Local tests

From this directory:

```bash
python -m pip install -e .
python -m pytest tests
```

The deterministic gates cover SE(2) conventions, fixed-extent labels, static and
terminal masking, command reconstruction, and compact LeRobot v3 loading.

## Reproducing the historical experiment

The included scripts record the original Vast.ai paths and hyperparameters for
auditability. They are not portable launchers: update dataset, checkpoint, and
workspace paths before using another machine. Do not commit datasets,
checkpoints, tokens, or generated videos.

The historical run started from a RoboCasa-multitask GR00T N1.6
`checkpoint-120000` and fine-tuned for 10,000 steps with global batch size 16,
Adafactor, and two GPUs. Its fixed-seed `NavigateKitchen` evaluation succeeded on
7 of 10 seeds. That 70% is an implementation result, not a demonstrated
advantage over GR00T: a matched command-chunk baseline with identical data,
training budget, seeds, horizon, and success criterion is still required.

## Known research gaps

1. The current path metric contains only chassis motion. A manipulation-capable
   version must construct synchronized whole-body anchors so that base-static
   arm motions such as `A -> B -> A` are not collapsed.
2. Segment durations are predicted, but progress is not corrected online from
   tracking feedback.
3. Inference does not predict a validity mask; very short terminal windows need
   explicit evaluation to ensure the executed prefix never reaches unsupervised
   padding.
4. Time-warp, control-frequency, delay, and matched-baseline experiments remain
   necessary before claiming time-reparameterization robustness.

