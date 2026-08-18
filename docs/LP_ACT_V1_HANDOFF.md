# LP-ACT V1 Research Handoff

Last updated: 2026-08-14

This document transfers the usable shared understanding from an extended research conversation into the local `zeno-rp` project. It is not merely a polished method description: it records the reasoning, corrections, rejected interpretations, current V1 algorithm, risks, and the next work expected from a new GPT/Codex session.

## 1. Project goal and immediate scope

The project studies a whole-body mobile-manipulation policy based on ACT. The existing policy predicts a fixed-frequency action chunk containing mechanical-arm targets and mobile-base commands. The research question is whether a more geometric action representation can make the policy less tightly coupled to demonstration speed, controller frequency, and base timing.

The longer-term idea is a path–phase decomposition:

\[
Q(t)=\Gamma(u(t)),
\]

where `Gamma` is a whole-body geometric path and `u(t)` is progress along it. A future, stronger method might adapt the phase rate `du/dt` online.

However, the immediate task is deliberately narrower:

> Implement and validate LP-ACT V1: ACT predicts a local whole-body path-time chunk of 32 waypoints. Do not yet add an online learned phase controller.

V1 is a representation and pipeline validation baseline. Its purpose is to establish that labels, learning, reconstruction, and rolling execution are correct before making a novelty claim.

## 2. Why this direction was considered

Standard ACT uses a time-indexed action chunk:

\[
\mathcal A_t=(a_t,a_{t+1},\ldots,a_{t+31}).
\]

Query `j` means the action at the `j`-th future fixed time step. Geometry and execution timing are entangled: the distance between two outputs depends on how fast the demonstration moved.

LP-ACT V1 instead uses a path-indexed chunk:

\[
\mathcal P_t=(Q(u_1),Q(u_2),\ldots,Q(u_{32})),
\qquad u_j=j/32.
\]

Query `j` means the `j`-th geometric anchor on a local whole-body path. The original arrival time is retained so the path can be converted back to fixed-rate robot commands.

The hoped-for engineering benefit is that spatial intent is represented more consistently when demonstration speed varies. That benefit is not assumed; it must be measured against standard ACT.

## 3. Conceptual corrections already established

### 3.1 `u(t)` is progress, not physical distance

Use distinct symbols:

- `s(t)`: physical or metric arc length already traversed;
- `L`: total metric length of the current local path;
- `u(t)=s(t)/L`: normalized path progress;
- `Gamma(u)`: configuration at normalized progress `u`;
- `du/dt`: phase/progress rate.

Therefore `u=0.5` means halfway through the current local path under the chosen whole-body metric. It is not directly 0.5 metres, 0.5 radians, or 50% of the global task.

### 3.2 Every predicted base pose is relative to the same chunk origin

For a chunk predicted at time `t`, all base poses are expressed relative to the observed base frame `B_t`:

\[
P_j={}^{B_t}T_{B_{t_j}}.
\]

They are not a sequence of ungrounded pairwise deltas. Adjacent transforms are computed only for length measurement or command reconstruction.

### 3.3 A waypoint query remains visually conditioned

All 32 ACT queries cross-attend to the same current visual tokens. Query 1 predicts the path front, query 16 the middle, and query 32 the end, all conditioned on the current observation. A distant waypoint does not contain its own future image; partial observability is handled only approximately through a short local horizon and rolling replanning.

### 3.4 The 32 points are analogous to an ACT action chunk

Yes: they are a `path-indexed action chunk`. The difference from standard ACT is the meaning of the query index, not the basic Transformer decoder mechanism.

### 3.5 Integrated command velocity is not ground-truth motion

If only commanded base velocity is available, integration yields a `commanded local path`, not the actual executed geometric trajectory. Label-source priority is:

\[
\text{measured pose/odometry}
>
\text{integrated measured body velocity}
>
\text{integrated commanded velocity}.
\]

This distinction must remain explicit in code, plots, and papers. Before implementing labels, inspect whether the dataset contains odometry, localisation, measured twist, or only command twist.

### 3.6 Waypoints alone are not a sufficient novelty claim

Prior work such as AWE already extracts waypoints from demonstrations to shorten behavioural-cloning horizons. Geometry/time separation is also classical trajectory-planning knowledge. Thus “convert velocity to waypoints” is not, by itself, the intended paper contribution.

A stronger later hypothesis could be:

> Explicit path–phase decomposition improves robustness to execution-speed change, control-frequency change, base dynamics delay, or initial-position variation, especially when phase progress adapts online.

That is intentionally out of scope for the first implementation.

## 4. LP-ACT V1 data model

At each synchronized sample `k`, expected raw fields are conceptually:

\[
\{I_k,q_k^{obs},a_k^{arm},g_k,\xi_k^{base},t_k\}.
\]

- `I_k`: image observation(s);
- `q_k_obs`: measured arm joint positions;
- `a_k_arm`: arm target used by the existing ACT supervision;
- `g_k`: gripper target/state under the existing convention;
- `xi_k_base=[v_x,v_y,omega]`: body-frame base twist, preferably measured, otherwise commanded;
- `t_k`: real timestamp.

These names are conceptual. The new agent must verify actual repository fields, shapes, frames, units, timestamps, command/state alignment, arm count, and gripper convention before coding.

The V1 model input should remain close to the current ACT input. Do not add global base pose solely to support labels; each chunk starts from a local identity transform.

## 5. Label-generation algorithm

### 5.1 Integrate or derive the future local base path

At training time `t`, initialize:

\[
P_0=I_{SE(2)}.
\]

Preferred case, if measured global/local odometry poses `T_k` exist:

\[
P_k=T_t^{-1}T_{t+k}.
\]

Fallback case, for body-frame twist:

\[
P_{k+1}=P_k\operatorname{Exp}(\xi_{t+k}\Delta t_k),
\qquad \Delta t_k=t_{k+1}-t_k.
\]

Use the repository's actual SE(2) left/right multiplication and body-frame conventions consistently. Do not assume axes or signs.

### 5.2 Form the whole-body configuration path

For each future sample:

\[
Q_k=[P_k,a_{t+k}^{arm},g_{t+k}].
\]

Operationally:

\[
Q_k=[x_k^{rel},y_k^{rel},\theta_k^{rel},q_{k,1},\ldots,q_{k,n},g_k].
\]

- Base pose is relative to the common chunk origin.
- Arm targets remain absolute joint targets under the existing ACT convention.
- Do not convert arm joints to deltas in V1.
- Gripper does not contribute to the path arc-length metric.

The current configuration is:

\[
Q_0=[0,0,0,q_t^{obs},g_t].
\]

### 5.3 Measure whole-body path length

Compute each adjacent local base displacement:

\[
\delta\xi_k=\operatorname{Log}(P_k^{-1}P_{k+1})
=[\delta x_k,\delta y_k,\delta\theta_k].
\]

Use a dimensionless weighted metric such as:

\[
\Delta\sigma_k=
\sqrt{
(\delta x_k/l_{xy})^2+
(\delta y_k/l_{xy})^2+
(\delta\theta_k/l_\theta)^2+
\lambda_q\sum_r(\Delta q_{k,r}/s_{q,r})^2
}.
\]

Then:

\[
\sigma_0=0,\qquad
\sigma_k=\sum_{i<k}\Delta\sigma_i.
\]

`l_xy`, `l_theta`, joint scales `s_q`, and `lambda_q` must be reported and ablated or at least sensitivity-checked. Initially, use robust training-set scales rather than arbitrary raw-unit equality.

### 5.4 Choose a local window

Stop at the first future sample satisfying either:

\[
\sigma_k\ge L_\sigma
\]

or:

\[
t_k-t\ge T_{max},
\]

or episode end.

Provisional `T_max` is 4 seconds. A principled first `L_sigma` is the median whole-body metric length covered by the current standard ACT horizon across the training set. This aims to keep standard ACT and LP-ACT comparable in average motion coverage.

Important unresolved issue: if every sample uses its own attained `L_t` and is always resampled to 32 equal fractions, the geometric horizon can vary near episode ends and under `T_max`. Labels need a validity mask or a clearly specified terminal-padding rule. Do not hide this with repeated targets without recording the mask.

### 5.5 Resample to 32 path anchors

Let the attained window length be `L_t`. Define:

\[
s_j=(j/32)L_t,\qquad j=1,\ldots,32.
\]

Locate raw indices where:

\[
\sigma_k\le s_j\le\sigma_{k+1},
\]

and set:

\[
\alpha_j=
\frac{s_j-\sigma_k}{\sigma_{k+1}-\sigma_k}.
\]

Interpolate the base in SE(2):

\[
\bar P_j=P_k\operatorname{Exp}
(\alpha_j\operatorname{Log}(P_k^{-1}P_{k+1})).
\]

Interpolate arm targets linearly:

\[
\bar q_j=(1-\alpha_j)q_k+\alpha_jq_{k+1}.
\]

Use nearest-neighbour or zero-order hold for discrete gripper state:

\[
\bar g_j=g_k.
\]

Interpolate original arrival time:

\[
\tau_j=(t_k-t)+\alpha_j(t_{k+1}-t_k),
\]

and segment duration:

\[
\Delta\tau_j=\tau_j-\tau_{j-1},\qquad \tau_0=0.
\]

The per-query target is provisionally:

\[
y_j=[\bar x_j,\bar y_j,\bar\theta_j,\bar q_j,\bar g_j,
\log\Delta\tau_j].
\]

### 5.6 Static and dwell behaviour

Equal-arc-length resampling becomes ill-conditioned when the future trajectory is static or contains long holds, because many samples share zero metric increment. V1 needs an explicit deterministic policy:

- classify a window static if `L_t < epsilon_sigma`;
- emit repeated current pose/joints/gripper targets;
- assign durations from the real available window rather than an invented duration where possible;
- cap the training fraction of fully static samples (provisional maximum 20%);
- avoid division by zero when locating interpolation intervals.

Known limitation: a purely equal-arc-length path plus one duration per spatial segment does not elegantly represent “hold for a while, then move.” Do not add a hold token in V1 unless reconstruction tests show that this limitation blocks the task; document it for V2.

## 6. Model changes for V1

Preserve the current ACT image encoder, Transformer encoder/decoder, and CVAE structure unless repository inspection exposes an incompatibility.

Change:

1. The decoder query count to 32, if not already 32.
2. The action/output dimension to:

\[
D=3+n+1+1,
\]

corresponding to relative base pose, `n` arm joints, gripper, and log segment duration.
3. Dataset targets, normalization statistics, masking, output head, loss decomposition, inference post-processing, and execution adapter.

All continuous outputs should use training-set normalization. Angle loss can be periodic:

\[
\mathcal L_\theta=\frac1{32}\sum_j
[1-\cos(\hat\theta_j-\theta_j)].
\]

Other provisional losses are normalized L1 terms for base translation, arm targets, and log duration; gripper retains the current ACT regression/classification convention. Retain the ACT KL loss. Start with balanced normalized losses, but log each component separately.

Do not assume the earlier symbol `lambda_q` should be shared between the arc-length metric and training loss; use distinct config names in code.

## 7. Inference and rolling execution

At time `t`, predict 32 tuples:

\[
\{\hat P_j,\hat q_j,\hat g_j,
\widehat{\log\Delta\tau_j}\}_{j=1}^{32}.
\]

Recover bounded positive durations:

\[
\hat{\Delta\tau}_j=
\operatorname{clip}(\exp(\widehat{\log\Delta\tau_j}),
\Delta\tau_{min},\Delta\tau_{max}),
\]

then cumulative arrival times:

\[
\hat\tau_j=\sum_{i=1}^j\hat{\Delta\tau}_i.
\]

Prepend the measured current state:

\[
\hat P_0=I,\quad \hat q_0=q_t,\quad \hat\tau_0=0.
\]

Interpolate the predicted path-time samples to the robot control frequency, provisionally 30 Hz.

For adjacent desired base poses:

\[
\xi^{cmd}(h)=\frac1{\Delta t}
\operatorname{Log}(P^{des}(h)^{-1}P^{des}(h+\Delta t)).
\]

Pass the resulting body-frame twist through existing velocity and acceleration limits. Send interpolated arm joint-position targets under the current controller interface.

Initial rolling policy:

- predict a full 32-point path-time chunk;
- execute the first 6 control frames (about 0.2 s at 30 Hz);
- discard the unused remainder;
- observe again and replan;
- each new chunk has a new local origin.

This 6-frame execution length is provisional and should match the existing ACT evaluation protocol where possible.

Important implementation question: predictions are relative to observation-time frame `B_t`, while commands execute over a nonzero interval. The execution adapter must consistently interpret the planned local path from the chunk origin and use measured base feedback if the existing controller requires current-relative error. Do not naïvely reset the predicted path origin every control tick inside a single executed prefix.

## 8. Required validation gates before full training

### Gate A: data and frame audit

For representative episodes, report:

- exact field names, shapes, dtypes, units, sampling rates, and timestamp source;
- whether base twist is commanded or measured;
- whether odometry/localisation exists;
- body/world frame definitions and transform multiplication convention;
- action/state time offset and controller latency;
- missing timestamps, duplicate timestamps, and discontinuities.

### Gate B: synthetic SE(2) tests

Test pure positive `v_x`, pure positive `v_y`, pure positive `omega`, combined curved motion, and near-zero rotation. Verify Exp/Log round trips and visual direction signs.

### Gate C: label visualization

Overlay local base waypoint paths on a consistent top-down plot and plot each arm joint versus original time/path index. Display original samples, 32 anchors, reconstructed samples, arrival time, and validity mask. Inspect examples from motion, turning, manipulation-only, static, dwell, and episode-end windows.

### Gate D: oracle compression–reconstruction

Without a neural network:

\[
30\text{ Hz raw trajectory}
\rightarrow 32\text{ waypoints}
\rightarrow 30\text{ Hz reconstructed commands}.
\]

Measure at least:

- arm joint RMSE/max error;
- final and pathwise base pose error;
- base twist RMSE after reconstruction;
- duration error;
- gripper transition timing error;
- saturation rate after velocity/acceleration limits.

If oracle reconstruction is poor, do not train; fix the representation or label generator first.

### Gate E: tiny-set overfit

Overfit one batch or a very small trajectory set. Confirm accurate recovery of base path, arm path, gripper, durations, mask, and reconstruction. Plot predictions instead of relying only on scalar loss.

### Gate F: open-loop and low-risk closed-loop comparison

Only after earlier gates pass, compare standard ACT and LP-ACT V1 using matched data, backbone capacity, observations, training steps, and rolling-execution budget. First run offline/open-loop metrics, then simulation or a safe hardware subset before full robot evaluation.

## 9. Baselines and experiments that matter

Minimum comparison:

- existing standard ACT with fixed-time action chunks;
- LP-ACT V1 with path-indexed waypoints plus demonstrated segment time.

Fairness controls:

- same demonstrations and train/validation split;
- same visual encoder and Transformer capacity;
- comparable future motion coverage;
- same observation modalities;
- same inference rate and replanning prefix where feasible;
- report parameter-count and output-dimension changes.

The most informative stress tests for the eventual research claim are:

- execution speed scaling;
- policy/control frequency change;
- added base response delay or dynamics mismatch;
- initial base-position perturbation;
- variable demonstration speeds;
- long holds/contact phases.

For V1, first establish nominal parity and pipeline correctness. Do not interpret nominal improvement alone as proof of path–phase robustness.

## 10. Decisions deliberately deferred

Do not implement these in the first pass:

- learned online phase rate `du/dt`;
- closed-loop adaptive execution clock;
- HOLD token or separate dwell-time head;
- future-image or latent-world-model prediction for distant waypoints;
- automatic minimal waypoint extraction as in AWE;
- a new architecture replacing ACT;
- complex smoothness, feasibility, or collision losses before baseline reconstruction is understood.

These may become V2 only after V1 identifies a concrete failure mode.

## 11. Current uncertainties the next GPT must resolve from code/data

1. What exactly are the existing ACT action fields for arms, base, and gripper?
2. Is base velocity a command, measured twist, or both?
3. Does odometry/localisation exist, and is it synchronized accurately enough for labels?
4. Are arm actions absolute joint positions, deltas, velocities, or controller targets?
5. What are the real control/data frequencies and timestamp jitter?
6. How many arm joints and gripper dimensions are present?
7. What is the current ACT horizon, query count, execution prefix, temporal aggregation method, and normalization pipeline?
8. Are multiple cameras or asynchronous sensor streams involved?
9. How are episode ends and padded action chunks currently masked?
10. Can the existing base controller accept body-frame twist reconstructed from desired local poses?
11. How much of the raw trajectory can 32 anchors reconstruct before error becomes unacceptable?
12. How should `L_sigma` and the whole-body metric scales be chosen from actual data?

Until these are inspected, values such as `T_max=4 s`, 32 anchors, and 6 executed frames are provisional design defaults rather than verified repository facts.

## 12. Expected first task for the new GPT/Codex

The next agent should not immediately edit code. It should:

1. Read `AGENTS.md` and this document in full.
2. Locate repository instructions and the current ACT dataset/model/evaluation code.
3. Trace one sample end to end: stored episode fields -> dataset item -> ACT target -> model output -> inference action -> robot controller.
4. Produce a concrete schema/frame audit and identify the minimal files that would change for LP-ACT V1.
5. Highlight every place where this handoff conflicts with actual code or data.
6. Propose the smallest implementation sequence, beginning with an offline label/reconstruction tool.
7. Wait for user approval before modifying training or hardware-control code.

Suggested first prompt:

> 请先完整阅读根目录 `AGENTS.md` 和 `docs/LP_ACT_V1_HANDOFF.md`。然后只读检查当前仓库，追踪一条数据从 episode 存储、dataset、ACT target、模型输出、inference 到机器人 controller 的完整路径。请输出：（1）真实字段/单位/坐标系/时间戳审计；（2）哪些记录是实测量、哪些是命令量；（3）LP-ACT V1 最小需要修改的文件和接口；（4）handoff 与真实代码不一致之处；（5）先实现离线 label + reconstruction validator 的分步计划。暂时不要修改代码，也不要运行机器人。

## 13. Compact algorithm summary

Training label path:

```text
current observation at t
  -> derive future base poses relative to B_t
  -> combine relative base pose + absolute arm target + gripper
  -> compute weighted whole-body cumulative path length
  -> stop at geometric limit / time limit / episode end
  -> resample to 32 equal-metric path anchors
  -> retain/interpolate each anchor's demonstrated arrival time
  -> train ACT to predict 32 [base pose, arm, gripper, log duration] targets
```

Deployment path:

```text
observe image/state
  -> predict 32 local path-time anchors
  -> prepend current measured configuration
  -> reconstruct a 30 Hz desired trajectory
  -> convert adjacent desired base poses to body-frame twist using SE(2) Log
  -> apply controller limits and arm targets
  -> execute first 6 frames
  -> observe and replan from a new local origin
```

Core V1 equation:

\[
(I_t,q_t,\xi_t)
\xrightarrow{ACT}
\{\Gamma_t(u_j),\Delta\tau_j\}_{j=1}^{32}
\xrightarrow{interpolation+SE(2)\ Log}
(\xi^{base},q^{arm},g)_{30\,Hz}.
\]

The central discipline for this phase is:

> First prove that the representation is defined, reversible enough, learnable, and safely executable. Only then decide whether the adaptive path–phase idea is a research contribution worth extending.

## 14. 2026-08-18 RoboCasa / GR00T N1.6 update

This section supersedes the joint-space assumptions above only for the current
RoboCasa experiment. It records facts verified from the converted data and the
running implementation; it does not retroactively claim that every dataset has
the same schema.

### 14.1 Verified RoboCasa schema and timing

- Data rate is 20 Hz; a 32-step target spans 1.6 seconds.
- Each sample has a 16D measured state: 3D relative EEF position, 4D EEF
  quaternion, 2D gripper qpos, 3D base position, and 4D base quaternion.
- Each native action is 12D: 4D `base_motion`, 1D `control_mode`, 3D EEF
  position command, 3D EEF rotation command, and 1D `gripper_close`.
- LP pose labels are derived from measured base/EEF state. Native command
  integration is not reinterpreted as physical motion.
- This converted RoboCasa source exposes Cartesian EEF targets rather than the
  absolute arm joint targets assumed by the original ACT handoff. The current
  comparison therefore keeps the same EEF-space interface for all variants.

The three training datasets are `NavigateKitchen` (500 episodes),
`PickPlaceCounterToStove` (501), and `DeliverStraw` (504). The deterministic
90/10 episode split uses seed `20260818`, giving 1,353 train episodes and 152
held-out episodes. The episode-balanced sampler draws 128 valid 32-step starts
per train episode: 57,600, 57,600, and 57,984 selected chunks respectively.

### 14.2 Matched representation ablations

- **B0 Command-Time:** predict the original 32 native command frames.
- **B1 Pose-Time:** predict measured poses at the exact future frames
  `t1...t32` relative to the current state, plus the original 32 segment
  durations. No geometry-progress resampling occurs. Gripper and control-mode
  samples remain frame aligned.
- **B2 Path-Time:** derive the same measured local pose trajectory, resample its
  continuous channels to 32 equal geometry-progress anchors, and retain arrival
  time / segment duration. Discrete events use their explicit event-preserving
  semantics.
- **B3 Path-RateFree:** deferred. It is not part of this 20k comparison.

B1 is required to isolate two effects that were previously confounded:
replacing command supervision with measured-pose supervision, and replacing
the time index with a geometry-progress index. An A-B-A motion cannot disappear
in B1 because every original future frame is preserved.

### 14.3 Validation evidence before the full run

- 13 label/geometry/execution unit tests passed.
- The held-out B1 schema, timestamp, reconstruction, event-preservation, and
  finite-value gate passed on all three tasks.
- Independent B1 normalization statistics were computed from 57,600
  `NavigateKitchen`, 57,600 `PickPlaceCounterToStove`, and 57,984
  `DeliverStraw` chunks; all serialized values are finite.
- A 20-step B1 tiny-set overfit reduced logged loss from 0.2547 to 0.0734;
  final mean train loss was 0.1641. Peak memory was about 28.3 GB on each of two
  RTX 5090 GPUs, with no OOM, NaN, traceback, or NCCL failure.
- Model-only checkpoint saving was separately gated because a standard
  DeepSpeed tiny checkpoint occupied about 40 GB. The gated model-only
  checkpoint was inference-complete and avoided optimizer shards.

### 14.4 Current matched experiment

All variants start from the same RoboCasa-pretrained GR00T N1.6 checkpoint:
`grootn16-robocasa365/checkpoint-120000`. B0 and B1 train concurrently on two
RTX 5090 GPUs each; B2 starts on all four GPUs only after both finish.

Each model uses 20,000 optimizer steps, effective global batch 16, per-GPU
micro-batch 1, and gradient accumulation 8 for two-GPU runs (4 for the
four-GPU B2 run). Thus each model consumes 320,000 sampled chunks. Checkpoints
are uniquely named, saved every 500 steps with only the latest retained on
Vast, and 10k/20k inference checkpoints are archived to `zeno-rp`.

The planned terminal comparison is matched closed-loop evaluation of B0, B1,
and B2 over three tasks and 30 fixed seeds per task/model (270 rollouts). One
unresolved issue remains: the current RoboCasa simulator commit has exact gym
environments for `NavigateKitchen` and `PnPCounterToStove`, but an exact
`DeliverStraw` environment has not yet been verified. Do not silently replace
it with a proxy; resolve the executable task mapping before claiming the full
three-task result.

The next concrete experiment is to finish the matched 20k training, run
held-out open-loop diagnostics at the archived checkpoints, then run the fixed
seed closed-loop protocol with identical observation and execution budgets.

### 14.5 Exact RoboCasa365 evaluation environment decision

The unresolved `DeliverStraw` mapping has been traced to a simulator-version
mismatch rather than to a renamed old gym ID. The GR00T-pinned RoboCasa fork is
commit `d89d481ce9c76da7f179466981676e268aa842e5`; neither its source registry nor
its installed gym wrappers contain `DeliverStraw`. The official RoboCasa365
v1.0.1 repository at commit `921c9a5736a8d0ea5589657898aadcfa55a6a195`
contains the exact composite `DeliverStraw` class, its success predicate, and
the target-split dataset registry entry. It also exposes the same Panda-Omron
12D command groups and measured state groups through the new
`robocasa/<Task>` gym wrapper.

Decision: keep the pinned GR00T/RoboCasa training environment unchanged and
create a separate pinned RoboCasa365 evaluation environment using robosuite
commit `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`. Before closed-loop inference,
construct and schema-audit the exact target-split environments
`robocasa/NavigateKitchen`, `robocasa/PickPlaceCounterToStove`, and
`robocasa/DeliverStraw`. Use the official v1.0.1 horizons 450, 600, and 2550,
respectively. Do not use the earlier 720-step global cap for the composite
task, and do not substitute a proxy.

The fixed comparison remains 30 seeds per task and method, seeds 20260818 to
20260847, 8 executed primitive actions per policy call, for 270 total rollouts.
The environment installation, exact commit IDs, schema report, open-loop
reports, closed-loop JSON, and videos must all be archived back to `zeno-rp`.
