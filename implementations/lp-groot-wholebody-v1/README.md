# LP-GR00T Whole-Body V1

Matched RoboCasa experiment for three GR00T N1.6 fine-tunes:

- **B0 / GR00T-Command** predicts the native 32-command chunk at 20 Hz.
- **B1 / Pose-Time** predicts the 32 measured future poses at their original
  frame times.
- **B2 / Path-Time** predicts 32 measured whole-body path anchors and one
  segment duration per anchor.

All methods start from the same RoboCasa-trained checkpoint, see the same three
tasks, use the same deterministic split and episode-balanced sampler, and use
effective global batch size 16. B0 and B1 train on two GPUs each; B2 starts on
all four GPUs after both complete.

## Frozen data contract

Tasks:

1. `NavigateKitchen`
2. `PickPlaceCounterToStove`
3. `DeliverStraw`

The audited LeRobot v2.1 rows contain a 16-D measured state and a 12-D native
command. State order is base position (3), base quaternion xyzw (4), EEF
position relative (3), EEF quaternion relative xyzw (4), and gripper qpos (2).
Native command order is base motion (4), control mode (1), EEF position command
(3), EEF rotation command (3), and gripper command (1). The modality adapter
exposes these in GR00T's named-key order.

For one B2 sample at frame `t0`, 33 measured states (`t0...t32`) define 32
segments, while the matched B0 sample contains commands `t0...t31`. At 20 Hz
both cover exactly 1.6 seconds. The GR00T checkpoint has a 50-step action head,
so `K=32` does not resize or randomly initialize that head.

## B2 output

The existing 12-D action slots are deliberately reinterpreted by the B2
adapter:

| GR00T key | B2 value | Dim |
|---|---|---:|
| `end_effector_position` | EEF position displacement from `t0` | 3 |
| `end_effector_rotation` | EEF relative rotation vector from `t0` | 3 |
| `gripper_close` | executable gripper event, ZOH | 1 |
| `base_motion` | local base `x,y,yaw` plus `log(segment_dt)` | 4 |
| `control_mode` | executable control-mode event, ZOH | 1 |

Continuous anchors are sampled at equal cumulative whole-body geometric
progress using fixed physical scales: base translation 0.05 m, base yaw 0.15
rad, EEF translation 0.02 m, and EEF rotation 0.15 rad. These constants do not
come from adjacent-frame statistics. Base and EEF share the same interpolation
coefficient. Gripper/control transitions are inserted as mandatory event knots;
therefore a discrete transition cannot silently disappear during resampling.

The base path is built only from measured `base_position/base_rotation`.
Commanded `base_motion` is never integrated and called ground-truth motion.

## Sampling and split

- task uniform;
- episode uniform inside each task;
- frame uniform inside each selected episode;
- seed `20260818`;
- 90% episodes for training and 10% held out for offline validation;
- 128 sampled starting frames per training episode.
- 20,000 matched optimizer steps, with a model-only checkpoint every 500 steps.

The source archives expose only a `train` split, so the held-out split above is
ours, not an official RoboCasa validation split.

## Required order

1. `scripts/setup_environment.sh`
2. `scripts/build_label_stats.py`
3. `scripts/validate_labels.py`
4. unit tests and a tiny-set overfit
5. matched B0/B1 training followed by B2 training
6. 30-sample-per-task held-out validation
7. fixed-seed RoboCasa365 target-split rollouts

## Exact RoboCasa365 closed-loop protocol

The old GR00T-pinned RoboCasa fork contains exact environments for the first
two tasks but predates the composite `DeliverStraw` class. Closed-loop
evaluation therefore uses an isolated, pinned RoboCasa365 v1.0.1 environment:

- RoboCasa commit `921c9a5736a8d0ea5589657898aadcfa55a6a195`;
- robosuite commit `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`;
- target-split gym IDs `robocasa/NavigateKitchen`,
  `robocasa/PickPlaceCounterToStove`, and `robocasa/DeliverStraw`;
- official v1.0.1 horizons 450, 600, and 2550 control steps;
- seeds 20260818 through 20260847, 30 rollouts per task and method;
- 8 primitive actions per policy query and 20 Hz execution.

`scripts/setup_robocasa365_eval.sh` creates this environment without changing
the GR00T training environment. `scripts/validate_robocasa365_envs.py` must
construct all three exact environments and verify their observation/action
schema before the 270-rollout matrix starts. New and legacy camera/language key
names are accepted by the policy server, but task proxies are not.

Long-running setup, training, and evaluation jobs must be launched through the
provided supervisor configs on Vast. Outputs are temporary on Vast and must be
synced back to `zeno-rp`.

On the current Vast image, PyTorch's bundled NCCL 2.26 stalls during communicator
initialization with the CUDA 13.3 driver. Training scripts preload the image's
`/usr/lib/x86_64-linux-gnu/libnccl.so.2.30.7`; both GPU pairs passed a two-rank
100 MB broadcast before training.
