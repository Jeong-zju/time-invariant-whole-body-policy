# LP-ACT V1 for BEHAVIOR 2026

Phase-1 scope is deliberately single-task: both standard ACT and LP-ACT V1 use
only `turning_on_radio`. The `bringing_water` and `sweeping_garage` assets remain
installed for simulator coverage, but they are not part of training.

This project is separate from the upstream BEHAVIOR-1K checkout. The first
deliverable is an offline, deterministic validator:

```text
30 Hz R1Pro trajectory
  -> integrate measured robot-frame base qvel in SE(2)
  -> combine local base pose with absolute trunk/arm targets and grippers
  -> resample 32 whole-body path anchors with demonstrated arrival times
  -> reconstruct 30 Hz commands
  -> report geometry, joint, gripper, duration, and velocity errors
```

No robot or simulator is required for these checks.

## Verified data conventions

- `action`: 23D
  - `0:3`: robot-frame holonomic base command velocity `(vx, vy, wz)`
  - `3:7`: absolute trunk targets
  - `7:14`: absolute left-arm targets
  - `14`: left-gripper command
  - `15:22`: absolute right-arm targets
  - `22`: right-gripper command
- `observation.state`: 61D
  - `0:3`: measured robot-frame base qvel
  - `3:10`: measured left-arm qpos
  - `28:35`: measured right-arm qpos
  - `53:57`: measured trunk qpos
- No odometry or localization pose is present in the released demonstration
  rows. LP-ACT labels therefore describe an integrated measured-velocity path,
  not ground-truth global motion.
- The recorder pairs `obs[t]` with the action subsequently applied at `t`.
  Its effect is measured in `obs[t+1]`. Consequently, the physical base motion
  over interval `t -> t+1` uses `state[t+1, 0:3]`, while its command comparison
  uses `action[t, 0:3]`. This one-frame alignment is explicit in code.

LP-ACT V1 target order is 24D:

```text
[relative base pose (3), original action[3:23] targets (20), log segment duration (1)]
```

## Split

The deterministic phase-1 split is by episode index:

- validation: `episode_index % 5 == 0` (40 episodes)
- training: remaining 160 episodes

This prevents frames from one episode appearing in both splits.

## Commands

```bash
python -m pytest
python -m lp_act.validate_oracle \
  --data-root /workspace/behavior-2026/demos \
  --output-dir /workspace/behavior-2026/lp-act-v1/outputs/oracle_fixed_v2 \
  --num-windows 512 \
  --extent-mode fixed
```

The validator fits task-specific metric scales on the training episodes only,
derives `L_sigma` as the median whole-body metric length covered by 32 raw
intervals, evaluates held-out windows, and writes JSON plus diagnostic plots.

## V1 terminal rule

LP-ACT V1 uses a fixed metric extent `L_sigma`. A trajectory that reaches the
extent is interpolated to end exactly at `L_sigma`; the overshooting part of the
raw control interval is not included. If an episode or the four-second time cap
ends first, the final partial path segment is retained as one valid anchor and
all later anchors are padding. Padding is excluded from ACT's L1 loss.

The held-out `oracle_fixed_v2` gate used 512 windows and obtained mean base
translation RMSE `2.95311e-05 m`, mean yaw RMSE `1.55540e-04 rad`, mean joint
target RMSE `4.19964e-04 rad`, and zero duration error. Seventy windows required
padding; the other 442 used all 32 anchors.
