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

## 14. Implementation and training addendum — 2026-08-14

The repository/data audit and the first implementation gates are now complete
for `turning_on_radio` only. Verified facts and decisions:

- The dataset has 200 episodes and 429,928 frames at 30 Hz. The matched split is
  160 training episodes and 40 validation episodes.
- The 61D state contains measured generalized velocity for the base. The label
  generator integrates `state[t+1, 0:3]` over `t -> t+1`, because recorder
  alignment tests show that `action[t]` takes effect in `obs[t+1]`. This is an
  integrated measured-qvel path, not global odometry or ground-truth pose.
- LP-ACT V1 uses fixed `L_sigma`, an exact fractional endpoint when the path
  limit is crossed, one truthful terminal anchor for incomplete windows, and a
  mask for repeated padding. The earlier per-window attained-extent variant was
  rejected because it fabricated 32 valid subsegments at short episode tails.
- Oracle reconstruction passed on 512 held-out windows: mean base translation
  RMSE `2.95311e-05 m`, yaw RMSE `1.55540e-04 rad`, joint-target RMSE
  `4.19964e-04 rad`, and zero duration error.
- Both policies passed the tiny-set gate at 1,000 steps. Standard ACT normalized
  L1 fell from `0.912077` to `0.0414183`; LP-ACT V1 fell from `0.897542` to
  `0.0467029`.

The formal full runs use batch 32, 50,000 steps, learning rate `1e-5`, seed 0,
eight workers, validation every 1,000 steps, and checkpoints every 5,000 steps.
Only the output parameterization differs between the policies.

The original PyAV input path was rejected for full training because it sustained
only about 0.39 step/s and starved both GPUs. The active pipeline uses a verified
224x224 H.264 all-intra RGB-only view and TorchCodec 0.5 + CUDA 12.8. Dataset
timestamp tolerance is 1 ms; a known recoverable TorchCodec timestamp/packet
failure retries the same sample with PyAV rather than skipping it. Validation
uses 256 fixed samples decoded once with PyAV and cached.

LeRobot ACT does not expose VAE posterior parameters in inference/eval mode, so
its standard training `forward` cannot be used to compute evaluation KL. The
implemented validation metric is masked inference-time L1 from
`predict_action_chunk`; training still uses the unchanged ACT L1 + KL objective.

The active uniquely named runs are:

- `full_act_turning_radio_b32_seed0_v7` on GPU 0;
- `full_lpact_turning_radio_b32_seed0_v7` on GPU 1.

Both passed their first validation gates and remained under supervisor at the
latest recorded snapshot. The next concrete experiment is to finish 50,000
steps, verify best/final checkpoints, export curves, and compare both policies
after reconstructing predictions into the same physical action space. Raw
normalized L1 values from the two different output representations are not a
valid direct ranking.

## 15. Training completion and first official ACT evaluation — 2026-08-14

Both matched 50,000-step runs completed successfully. Standard ACT's best
offline validation checkpoint is step 34,000 with normalized inference L1
`0.0820688`; LP-ACT V1's best is step 48,000 with `0.110553`. These values are
not a cross-policy ranking because the target representations and normalization
statistics differ. Both final checkpoints load strictly with 234 model tensors
and complete optimizer state.

An official BEHAVIOR 2026 websocket adapter was added for standard ACT. It maps
the evaluator's flattened observations to the training schema:

- `robot_r1::proprio`: 61D state in the verified training order;
- three evaluator RGB observations: 224x224 RGBA, with alpha removed and the
  same ImageNet normalization used during training;
- output: raw 23D R1Pro controller action in the verified controller order.

The rolling execution rule predicts 32 actions and executes the first six before
replanning. Base commands and grippers are clipped to official controller
limits; absolute trunk and arm targets are left in the demonstrated convention.

The official public-test index 0 resolves to task instance 301. A five-step
smoke test passed end to end, including scene reset, websocket transport,
observation schema, model inference, 23D action application, and official JSON
generation. The full rollout then ran to the official 3,224-step timeout (the
evaluator reports 3,225 loop iterations) and obtained:

- success: false;
- final Q-score: `0.0`;
- base distance: `0.158674 m`;
- left/right end-effector displacement: `7.27319 / 7.86877`;
- simulated time: `107.5 s`.

The video shows an initial base rotation followed by little translational base
motion while the arms continue moving; the robot never reaches and turns on the
radio. Therefore public instances 1-9 must not be launched as a blind scale-up.
The next concrete experiment is to compare the rollout observations and action
statistics against held-out demonstrations, then replay/reconstruct a known
demonstration initial segment through the same adapter to isolate visual-domain,
state-order, action-normalization, and rolling-prefix mismatch before evaluating
LP-ACT.

The ACT rollout video and official metrics JSON were uploaded to the private
Hugging Face dataset `QRP123/lp-act-v1-behavior2026-evals` under
`act/turning_on_radio/public_instance_301/`.

## 16. First official LP-ACT V1 evaluation — 2026-08-14

The official websocket adapter now also supports LP-ACT V1. At each replan it
predicts 32 normalized path-time anchors, denormalizes them, prepends the
current measured base/trunk/arm configuration, decodes the anchors to a 30 Hz
23D controller trajectory, executes the first six frames, and then replans.
The decoder uses the same interpolation and SE(2) conventions that passed the
offline reconstruction validator. A saved official observation was decoded by
the step-48,000 checkpoint without dimension, finite-value, or queue errors,
and a five-step official smoke test passed end to end.

One complete official rollout was selected as the requested primary LP-ACT V1
result. Public-test index 0 resolves to task instance 301. It ran for the full
official timeout and obtained:

- success: false;
- final Q-score: `0.0`;
- evaluator loop iterations: `3,225`;
- base distance: `0.145497 m`;
- left/right end-effector displacement: `11.37446 / 10.20897`;
- simulated time: `107.5 s`.

Video inspection shows mostly in-place view changes and arm motion; the robot
does not approach and operate the radio. This proves that the LP-ACT
representation can be decoded and executed through the official evaluator, but
it does not establish task competence.

Before the request to stop after one result arrived, instances 302 and 303 had
also completed; both had success false and Q-score `0.0`. Instance 304 was
interrupted and is not counted. All remaining evaluator and policy-server jobs
were stopped, and both GPUs were released. Instance 301 remains the primary
reported result; the two extra completed runs are retained only as diagnostic
evidence and are not a reason to continue scaling evaluation.

Because standard ACT and LP-ACT both score zero on the first public start, this
evaluation does not yet distinguish the action representations. The next
concrete experiment should be a controlled adapter diagnosis on held-out and
known-demonstration initial states: compare observation distributions, decoded
physical-action distributions, rolling-prefix behavior, and short-horizon
reconstruction in the same 23D controller space before retraining or launching
more public rollouts.

## 17. Root-cause diagnosis of the zero-score rollouts — 2026-08-14

The optimization runs themselves completed and generalized reasonably under
teacher-forced held-out evaluation. Standard ACT finished with train L1
`0.06137` and best validation L1 `0.08207`; LP-ACT V1 finished with train L1
`0.07618` and best validation L1 `0.11055`. On all 40 held-out demonstration
starts, standard ACT's normalized L1 over the executed six-frame prefix was
only `0.02683`. These facts rule out an untrained checkpoint or failed GPU run
as the primary cause of Q-score zero.

Two stronger failure mechanisms were verified.

### 17.1 Receding-prefix no-op deadlock

The demonstrations contain non-causal operator delay at episode start. Using a
base-motion threshold of `0.02`, the median first-motion time is `1.23 s`.
Among the 200 demonstrations:

- 199/200 have no meaningful base motion in the first six frames (`0.2 s`);
- 109/200 have no meaningful base motion in the entire 32-frame ACT chunk
  (`1.07 s`);
- 55% still have not moved at `1.0 s`, 23.5% at `2.0 s`, and 12.5% at `3.0 s`.

The current evaluator executes only the first six predicted frames, discards
the remaining chunk, and replans from essentially the same observation. The
policy therefore repeatedly executes the demonstration's waiting prefix. Since
elapsed time is not an input, the same unchanged observation cannot tell the
policy that the human's arbitrary initial wait has expired. Low prefix loss at
demo starts is evidence that ACT learned this no-op target accurately, not
evidence that the closed-loop rollout is live.

This is not confined to the learned standard ACT head. With oracle LP-ACT
labels and the current LP decoder, 198/200 demonstration starts also produce no
meaningful base motion in the first six decoded frames. Thus the current
path-time label plus fixed six-frame replanning rule fails a liveness test even
before neural-network prediction error is introduced.

### 17.2 Reset-time velocity outliers

The official instance-301 first observation has two extreme normalized state
components:

- `state[57] = trunk_qvel[0]`: `+12.28` standard deviations;
- `state[58] = trunk_qvel[1]`: `-23.18` standard deviations.

All remaining components are within about `1.51` standard deviations. A
counterfactual inference that replaces only `state[57:61]` with the training
mean changes standard ACT's mean six-frame planar speed from about `0.066 m/s`
to `0.00032 m/s`. For LP-ACT, the raw reset state produces about `0.17 rad/s`
initial yaw, whereas the mean-velocity counterfactual largely removes that
rotation and still produces almost no translation. This explains the initial
rotation/spurious movement seen in both videos and then the near-stationary
behavior after reset transients decay.

The public head-camera view is visually similar to multiple demonstration
starts in the same scene, so gross RGB preprocessing or a wholly unseen visual
domain is not the leading explanation. It remains a secondary generalization
risk, not the first repair target.

### 17.3 Decision and next falsification experiment

Do not compare ACT versus LP-ACT task competence from these zero-score runs and
do not launch more public rollouts yet. The smallest next experiment is an
offline liveness correction shared by both policies:

1. Identify and trim only the leading demonstrator-idle segment of every
   episode, recording the threshold, original frame index, and removed time.
   Do not remove legitimate later task holds.
2. Rebuild standard and LP labels from the trimmed starts. Before training,
   require the oracle first-six-frame decoder to contain meaningful motion for
   the starts that later move; otherwise the six-frame rolling rule is still
   incompatible with the representation.
3. Exclude reset-only velocity transients from the policy query, preferably by
   a short simulator settling period and an explicit state-distribution check.
   Normalized clipping can be tested as a guard but is not a substitute for
   fixing the no-op labels.
4. Run the same tiny-set overfit gate, then a short matched ACT/LP-ACT training
   run. Evaluate liveness on all 200 recorded starts before spending another
   full 50,000-step run.

The falsification condition is simple: if the retrained policies still predict
no meaningful motion from recorded reset observations, or if the oracle LP
decoder still fails the first-prefix liveness check, stop before full training
and revise the label/execution contract.

## 18. RoboCasa base-only LP-ACT V1 implementation and paired training — 2026-08-15

The active benchmark was changed to RoboCasa `LineUpCondiments`. This is a
separate experiment from the BEHAVIOR `turning_on_radio` results above. The
RoboCasa implementation deliberately applies the path-time representation only
to the mobile chassis; torso, control mode, arm, and gripper targets remain
ordinary time-indexed ACT outputs. This matches the decision to isolate the
base representation before attempting a whole-body path metric.

Verified RoboCasa schema and representation:

- The LeRobot dataset contains 306 episodes and 133,309 frames at 20 Hz, with
  three 256x256 RGB cameras, a 16D state, and a 12D action.
- `action[0:3]` is the normalized mobile-base command. `action[3:12]` contains
  torso/control-mode/arm/gripper quantities and is not included in the LP arc
  length.
- The state contains measured global base position and an `xyzw` quaternion.
  LP labels use relative transforms from these measured poses. Commanded base
  velocity is not integrated and is not treated as ground-truth motion.
- Standard ACT predicts 32 time-indexed 12D actions. Base-only LP-ACT predicts
  32 hybrid 13D targets: local base pose `(x,y,yaw)`, `log delta_t`, and the
  nine ordinary non-base action dimensions.
- Both policies replan after executing eight 20 Hz controller frames, or 0.4 s.

The base metric was fitted only on training episodes. Its recorded parameters
are `xy_scale=0.0317450280`, `theta_scale=0.0290968977`, and
`path_length=25.66276536`, with a moving-window threshold of 0.01 m or 0.01
rad. The train-only controller calibration is
`vx/action=0.5860426367`, `vy/action=0.6258434605`, and
`yaw_rate/action=1.2192104494`.

Before full training, the following gates passed:

- synthetic SE(2), action-order, and phase-isolation tests: 5 passed;
- 512-window oracle decode at an eight-frame execution prefix: base-pose RMSE
  `0.0028649 m` and `0.0007946 rad`; calibrated command-integration diagnostic
  `0.0069147 m` and `0.0009322 rad`; command MAE `0.030824`;
- 16-sample, batch-16, 300-step overfit: standard normalized L1 fell from
  `0.7426` to `0.2096`; LP normalized L1 fell from `0.7770` to `0.2027`.

The first full-data PyAV run was rejected as the final training input path. It
reached only about 0.39 step/s and starved both GPUs, although its step-5,000
checkpoints were retained. A verified uint8 RGB memmap was then built at
`/workspace/lp-act-v1/cache/lineup_condiments_rgb256_v1.uint8`, shape
`[133309,3,256,256,3]`. Three sampled frame indices across all cameras were
pixel-identical to direct PyAV decode. Training resumed from the retained 5k
checkpoints and sustained roughly 16-17 steps/s with GPU utilization generally
around 70-80%.

The matched formal runs used batch 32, 50,000 total steps, eight workers,
learning rate `1e-5`, seed 0, validation every 1,000 steps on 256 fixed samples,
and checkpoints every 5,000 steps:

- standard ACT output:
  `robocasa_act_lineup_b32_s50000_seed0_v2_cache` on GPU 0;
- base-only LP-ACT output:
  `robocasa_lpact_base_only_lineup_b32_s50000_seed0_v2_cache` on GPU 1.

Both runs completed normally. Standard ACT's best checkpoint is step 43,000
with validation L1 `0.08038195`; LP-ACT's best checkpoint is step 38,000 with
validation L1 `0.08546836`. These normalized losses are defined in different
target spaces and must not be interpreted as a direct ranking.

Both best checkpoints were rolled out from the exact same held-out episode-0
state and XML. The initial state reconstruction error was `5.96e-8`. Each
policy ran the full 1,389 controller steps, issued 174 policy queries, and had
zero base-command clipping. Both rollouts ended `success=false`. Qualitative
video inspection shows substantial upper-body posture drift in both runs; the
LP-ACT rollout also knocks over the red condiment. This is a real negative
closed-loop result, not a successful task demonstration.

The paired videos and JSON reports were uploaded to the private Hugging Face
dataset `QRP123/lp-act-v1-behavior2026-evals` under
`robocasa/LineUpCondiments/base_only_lpact_v1/validation_episode_000/`.

Decision: do not claim that this one rollout ranks the base representations.
`LineUpCondiments` success requires upper-body manipulation, while the two
independently trained policies also make different non-base predictions. A
whole-task failure therefore confounds chassis quality with upper-body quality.
Also reject raw normalized validation L1 as a cross-representation comparison.

The next concrete experiment is a base-only held-out validator in common
physical space. Log predicted commands and measured `(x,y,yaw)` throughout the
rollout; compare standard ACT and decoded LP-ACT against the same held-out base
trajectory using displacement/yaw error, path progress, saturation, and
eight-frame reconstruction error. For a closed-loop causal test, either hold
the non-base command stream fixed across both methods or evaluate a task phase
whose success criterion depends only on chassis motion. Only after that
diagnostic should another full-task rollout or wider episode sweep be used to
assess the representation.

## 19. GR00T N1.7 NavigateKitchen single-task restart — 2026-08-21

This experiment is separate from the ACT/LP-ACT implementation above. Its
purpose is to establish a working single-task GR00T N1.7 baseline before any
new representation comparison. The only task in scope is RoboCasa
`NavigateKitchen`.

Verified facts from the previous N1.7 run:

- The source dataset has 500 episodes, 72,786 frames, 13 language goals, three
  256x256 RGB cameras, a 16D state, a 12D action, and a declared 20 Hz rate.
- The previous run started from generic `nvidia/GR00T-N1.7-3B`, used the custom
  RoboCasa data as a new post-training embodiment, an eight-step action chunk,
  effective batch 16, and stopped at checkpoint 20,000. There is no verified
  checkpoint 30,000 for that run.
- Its matched fixed-seed closed-loop evaluation was 0/30. A direct training
  frame probe showed incoherent, rapidly alternating predicted base commands
  while the demonstrated base command chunk was smooth.
- The older N1.6 120k checkpoint is not a clean from-scratch comparison: its
  recorded 300-dataset training mixture explicitly includes NavigateKitchen.
- The public N1.7 RoboCasa recipe does not provide a NavigateKitchen-trained
  checkpoint. It trains a different 24-task Panda-Omron manipulation mixture
  for 60,000 optimizer steps at global batch 512. Therefore its reported task
  success rates cannot be treated as a NavigateKitchen reference number.

Decision: do not spend another long run by changing only the step count. A
single-task N1.7 run must first pass an exact training reconstruction gate.
The first gate used all 170 valid eight-step windows from episode 0, trained
the full 12D command schema for 500 steps, and failed clearly:

- base XYZ MAE: `0.662682`;
- predicted versus target within-chunk adjacent difference: `0.810728` versus
  `0.021807`;
- predicted versus target sign-flip rate: `0.495575` versus `0.004630`.

This failure reproduces the closed-loop action instability offline and is not
a task-success metric. It blocks formal training.

The next, stricter falsification gate is intentionally base-only: one exact
eight-frame window from episode 0, only `base_motion` supervised, state dropout
and color jitter disabled, fixed full-frame crop, effective batch 16, and 1,000
optimizer steps on four GPUs. If the checkpoint cannot reconstruct this one
base chunk, stop and repair the GR00T action training/processor/decoder path;
do not attribute the failure to navigation difficulty or insufficient full-run
steps. Only a passed reconstruction gate authorizes the full 500-episode
single-task training and fixed-seed closed-loop evaluation.

That stricter 1,000-step gate also failed. Its final train loss was `0.905888`,
base XYZ MAE was `0.520995`, predicted versus target adjacent difference was
`0.827962` versus `0.118571`, and predicted versus target sign-flip rate was
`0.444444` versus `0`. Changing inference integration from 1 to 4, 16, or 64
denoising steps did not materially change the reconstruction. A weight audit
confirmed that training did update only the selected RoboCasa projector row
(ID 13); therefore this is not a missing-gradient or wrong-checkpoint-save
failure. The learned flow remains much too weak after 1,000 steps and leaves
the sampled-noise trajectory largely intact.

The final pre-full-run gate continues from this checkpoint with a fresh
optimizer and must be reported precisely as `1k AdamW + reset + 9k Adafactor`,
not continuous 10k. It uses the same single base-action window and the
single-task N1.7 recipe's Adafactor/gradient-checkpointing choice. Checkpoints
are retained at 3k, 6k, and 9k of stage 2. Formal 500-episode training remains
blocked until the stage-2 checkpoint reconstructs the exact base chunk.

The stage-2 run completed all 9,000 new optimizer steps in about 3 hours 31
minutes and wrote checkpoints 3,000, 6,000, and 9,000. Loss and gradients stayed
finite, but checkpoint 9,000 still failed the exact-window reconstruction gate:

- `base_motion` MAE: `0.448389`;
- base XYZ MAE: `0.597852`;
- predicted versus target adjacent difference: `0.869327` versus `0.118571`;
- predicted versus target sign-flip rate: `0.476190` versus `0`;
- predicted versus target coherence: `0.580195` versus `0.897100`.

This is an offline failure on the single training window, not a closed-loop
success rate. The formal 500-episode training and 30-seed evaluation were
correctly not started. More optimizer steps are no longer justified until the
training/evaluation interface is audited in normalized action space. The next
concrete diagnostic is to run one teacher-forced training batch through the
processor and flow-matching target construction, inspect the action mask, and
compare predicted flow against its exact supervised target before sampling or
de-normalization. Then verify that action preprocessing followed by
postprocessing is an exact round trip for the same eight-step base chunk.

## 20. From-scratch model choice for NavigateKitchen — 2026-08-21

The next baseline should use a small behavior-cloning policy initialized from
random weights, rather than another foundation-model post-training run. The
strongest candidate is RoboMimic's RoboCasa `BC-Transformer`, specialized to
the navigation-only action needed by `NavigateKitchen`.

Evidence and scope:

- `NavigateKitchen` is an atomic navigation task. Success requires the base to
  finish within 0.20 m of the target placement pose and satisfy
  `cos(yaw_error) >= 0.98`; it does not require manipulation.
- The current source dataset has 500 episodes, 72,786 frames, 13 goal strings,
  three RGB cameras, a 16D state, and 12D whole-body commands at 20 Hz.
- The original RoboCasa policy is a roughly 20M-parameter, six-layer
  BC-Transformer. It consumes ten observations, predicts ten actions, and
  replans after executing one action. The official configuration trains with
  AdamW at `1e-4` for 500k gradient steps.
- The official RoboMimic template sets the ResNet-18 visual encoder to
  `pretrained: false`, so the visual policy can genuinely start from random
  weights. For this single benchmark task, the 13 known language annotations
  can be represented by a learned goal-ID embedding; this avoids relying on a
  pretrained CLIP text encoder while preserving target conditioning.
- In the original RoboCasa comparison on a representative single-stage task,
  BC-Transformer achieved 56% versus 12% for Diffusion Policy. This is evidence
  about architecture suitability, not a published NavigateKitchen rate. The
  paper excluded NavigateKitchen from its MimicGen-generated comparison, so no
  verified official single-task NavigateKitchen success number was found.

Decision: the first strict from-scratch experiment should predict only the
three normalized `base_motion` commands `(v_x, v_y, omega)`. The remaining nine
action dimensions must be filled with a dataset-verified neutral hold command;
do not assume they are all zero. Inputs should contain the last ten synchronized
observations and a learned embedding for the exact goal annotation. Execute one
20 Hz command, observe again, and replan. This separates policy learnability
from the previously failing GR00T flow-matching/processor path.

Rejected as the first attempt:

- GR00T, pi0, and other large VLAs: they are foundation-model fine-tuning
  systems, not credible random-initialization learners for 500 demonstrations;
- Diffusion Policy: officially supported in RoboCasa 1.0.1 and valid as a
  second baseline, but heavier and weaker than BC-Transformer in the closest
  published single-task RoboCasa comparison;
- standard ACT as the first diagnostic: it can train from scratch, but its
  single-observation action chunk is a less direct fit than the verified
  ten-frame-history BC-Transformer for navigation;
- a classical planner using simulator target coordinates: it would test the
  controller but would bypass the image-and-goal learning problem.

Before a full run, require four gates: exact action/schema and neutral-hold
audit; one-window and one-episode overfit; normalized command reconstruction
with no sign flips or action-interface mismatch; and short seen-initial-state
closed-loop rollouts. A practical staged schedule is 50k steps first, then only
continue toward the official 500k-step recipe if these gates and fixed-seed
success improve. Diffusion Policy and full 12D ACT remain follow-up baselines,
not replacements for this first learnability test.

## 21. User-selected original ACT scratch baseline — 2026-08-21

The user overrode the preceding BC-Transformer-first recommendation and chose
the most direct original ACT baseline. The new experiment is therefore a clean
single-task `NavigateKitchen` ACT run from random initialization for 50,000
optimizer steps with global batch size 128. It predicts the native 12D
time-indexed RoboCasa action without LP waypoints, path labels, duration/rate
outputs, or a pretrained VLA.

The inspected dataset has 500 episodes and 72,786 frames at 20 Hz. The fixed
seed-20260820 split remains 450 train / 25 validation / 25 test episodes. The
policy consumes the three verified 256x256 RGB cameras and a 29D state: the
native 16D robot state concatenated with the exact 13-way task-ID one-hot.
Goal conditioning is necessary because `NavigateKitchen` contains 13 distinct
target descriptions; using the integer identity does not add a pretrained text
or vision model. ACT predicts 32 native 12D commands and the closed-loop adapter
will execute the first 8 commands before replanning.

Training is four-card DDP with batch 32 per GPU, hence global batch 128 per
optimizer update. The ResNet-18 weights are random
(`pretrained_backbone_weights=None`); the ACT transformer and VAE retain the
LeRobot v0.3.3 original-compatible defaults. AdamW uses LR `1e-5`, checkpoints
are model-only every 5,000 steps, and a single resume checkpoint is overwritten
inside the uniquely named run. Before formal training, three-camera videos are
decoded sequentially to a uint8 cache and a 32-sample/500-step overfit gate must
show finite loss/gradients and at least 15% L1 reduction.

Run root: `/workspace/act-navigate-kitchen-scratch-20260821`. Supervisor service:
`act-navigate-50k`. The concrete next checks are cache completion, tiny-overfit
pass, first formal step with all four GPUs active, and the first model-only
checkpoint at step 5,000. Training loss or checkpoint existence must not be
reported as task success; fixed-seed closed-loop evaluation remains required
after checkpoint 50,000.

## 22. Original ACT 50k completion and closed-loop start — 2026-08-21

The scratch ACT run completed all 50,000 optimizer steps in 3,077 seconds. The
final logged loss was `0.04649`, comprising L1 `0.02629` and KL `0.00202`, with
finite gradient norm `1.9196`. Ten model-only checkpoints exist at 5,000-step
intervals and the final model-only file is approximately 207 MB. These are
training-health facts only, not task-success evidence.

Closed-loop evaluation uses checkpoint 50,000, fixed seeds 20260818 through
20260847, RoboCasa split `target`, horizon 450, and executes 8 actions from each
32-action ACT prediction. The ACT server explicitly concatenates the same
verified 16D state order used for training, maps the exact simulator language
annotation to its 13-way task ID, and splits the native 12D prediction back into
the five RoboCasa action fields. The first launch stopped before any episode
because the server had not declared its 32-step horizon metadata; this was an
invalid 0/0 preflight, was repaired, and must not be counted. The corrected run
produced a real video and a valid first result for seed 20260818: failure (0/1
interim). The remaining 29 seeds were still running when this entry was added.

The corrected evaluation subsequently completed all 30 fixed seeds with exact
coverage 20260818 through 20260847. Successful seeds were 20260821, 20260836,
and 20260840: 3/30, or 10.0% success. `results.json`, `summary.json`, the
`EVALUATION_COMPLETE` marker, and 30 non-empty videos were verified. Client logs
contained the known Gymnasium observation-space warning but no rollout
Traceback, RuntimeError, ValueError, OOM, or NCCL failure. The invalid
horizon-metadata preflight remains excluded from these 30 episodes.

## 23. Corrected ACT-B2 Path-RateFree run — 2026-08-21

After the original ACT baseline reached 3/30 (10.0%), the user authorized a
matched four-GPU B2 run. This run starts from random initialization and retains
the same ACT image encoder, transformer/CVAE, three cameras, 29D input state,
32 queries, optimizer, global batch 128, and 50,000 optimizer steps. It does
not load the B0 checkpoint. Only the target representation and the eventual
base execution adapter change.

The historical B2 implementation was not reused unchanged: it sampled a
spatial path from exactly 32 future frames. Although it did not predict time,
that source window still implicitly depended on dataset frequency. The
corrected B2 instead uses the physical SE(2) metric

\[
d\sigma^2 = dx^2 + dy^2 + (0.25\,d\theta)^2,
\]

where translation is in metres and yaw is in radians. The fixed path extent is
the median measured moving-path coverage over a timestamp-defined 1.6-second
reference window on training episodes. Endpoint interpolation uses timestamps,
not frame counts. For this split the fitted extent is 0.861415 m-equivalent.
Each target contains 32 equal-progress measured relative SE(2) anchors. The
fourth base channel is constant zero padding; no timestamp, arrival time,
duration, velocity, or rate is supervised or predicted. Incomplete terminal or
four-second-bounded paths keep one truthful terminal anchor and explicitly mask
the remaining queries.

Upper-body command fields are aligned to the same source locations by linear
interpolation for continuous end-effector commands and zero-order hold for
control mode and gripper. Commanded base velocity is never integrated to create
labels; state base position and quaternion provide the measured trajectory.

The real-data gate passed on all 500 episodes / 72,786 frames. Train labels
cover 65,834 samples, have a median of 32 valid anchors, and 73.58% reach the
full physical extent. Re-sampling the same synthetic physical trajectory at
10, 20, and 40 Hz changed anchors by at most 8.20e-5. SE(2) round-trip error was
3.25e-15. A four-GPU 32-sample tiny-set test reduced mean logged normalized L1
from 0.299613 to 0.106675 (64.40%) in 500 steps.

The formal uniquely named run is
`/workspace/act-b2-ratefree-navigate-kitchen-scratch-20260821`, managed by
supervisor service `act-b2-ratefree-navigate-50k`. It uses four RTX 5090s,
batch 32 per GPU (global 128), and checkpoints every 5,000 steps. Closed-loop
success must not be inferred from these training facts. After 50,000 steps, B2
requires a controller-side feedback path follower and the same fixed seeds
20260818..20260847; a matched execute-prefix protocol must be documented because
the existing batched simulator interface does not expose intermediate odometry
within an eight-command prefix.

## 24. B2 failure audit and shorter geometric whole-body correction — 2026-08-21

The corrected B2 run completed 50,000 scratch-training steps and then completed
30 fixed-seed closed-loop episodes (20260818..20260847, target split, horizon
450). It scored 0/30 with 30 non-empty videos. This is a valid negative result,
not an incomplete preview. The policy did not collapse to zero commands: a
traced seed-20260821 rollout produced non-zero base commands for all 450 steps,
57 model replans, and 0.706 m of measured travel, but failed after following a
path into kitchen geometry.

The 512-sample validation audit isolated insufficient geometric precision.
Mean base-path component error was 0.0950 m in x, 0.0898 m in y, and 0.2836 rad
in yaw. Endpoint translation error averaged 0.2464 m and endpoint yaw error
averaged 0.4136 rad. Both exceed the NavigateKitchen success tolerances (0.20 m
and approximately 0.20 rad). First-anchor and endpoint direction were reversed
for 6.45% and 6.84% of samples respectively. Control-mode accuracy was 100%,
and no first anchor disabled the base, so control-mode prediction was not the
cause. Direct simulator command pulses also reproduced the calibration in free
space; the low net motion in the failed rollout came from collision, not a
global command-scale error.

The audit exposed a conceptual problem as well: section 23's B2 did not predict
time, but its 0.861415 m extent was fitted from a 1.6-second window and label
construction stopped after four seconds. It was therefore control-output
rate-free but not purely time-free in label construction.

The next version is `B2_Path_RateFree_Geometric0p10_WholeBody`. It uses the same
physical SE(2) metric with yaw radius 0.25 m, but fixes the path extent at 0.10
m-equivalent as an algorithm hyperparameter. For every observation, the label
generator scans the measured future trajectory until that geometric extent is
reached or the episode ends. It has no elapsed-time or frame-count cutoff and
still samples 32 equal-geometry anchors, separated by 0.003125 m-equivalent.
The output remains the same 12D whole-body target used in the 0/30 run: relative
base path, control mode, synchronized EEF position/rotation commands, and
gripper. Upper-body continuous commands are interpolated at the same geometric
source locations; control mode and gripper use zero-order hold. This experiment
does not remove the arm merely because NavigateKitchen is navigation-only.

The progress variable in this controlled correction remains base-SE(2)-defined,
as in the completed 0/30 run. A future fully coupled base-plus-EEF progress
metric is a separate algorithm change and must not be silently mixed into this
path-range ablation.

This version must use a new run directory and fresh scratch initialization.
Before formal training it must repeat the schema/frequency/SE(2) gates and the
four-GPU tiny-set overfit. After 50,000 steps it must be evaluated with the same
30 seeds and success criterion. The old 0/30 checkpoint and artifacts must be
retained as the rejected time-fitted B2 result.

## 25. B2 V3 spatial-resolution correction — 2026-08-21

The fixed `0.10 m / 32 anchors` correction was stopped at checkpoint 15,000
before closed-loop evaluation. Although its labels were independent of time and
frame count, it placed anchors only `0.003125 m` apart. That spacing was below
the deployed follower's `0.035 m` position tolerance, while its `0.08 m`
lookahead skipped about 26 anchors. Therefore it did not operationally use the
claimed 32-point local path. The retained checkpoint is diagnostic only.

B2 V3 chooses spatial resolution before training rather than copying ACT's
legacy 32 temporal queries. An oracle measured-feedback sweep evaluated eight
anchors with fixed spatial steps `{0.01, 0.015, 0.02, 0.025, 0.03, 0.04}` m.
The comparison used measured SE(2) increments, not native-command equality,
because the labels represent measured geometry and command disagreement can
include controller delay. The selected `0.025 m/anchor` configuration had the
lowest accepted physical reconstruction score: mean moving-direction cosine
`0.96229`, reverse fraction `0.00358`, no command saturation, and increment
RMSE `(0.01197 m, 0.00835 m, 0.01196 rad)`. With eight anchors its maximum
geometric lookahead is `0.20 m`; this value is a consequence of the selected
controller-resolvable spacing, not a time-window fit.

The V3 output is eight 12D path tokens:

```text
[relative base SE(2), control_mode, EEF position, EEF rotation,
 gripper, CONTINUE/STOP]
```

Every query is supervised. After the truthful terminal anchor, geometry and
upper-body commands are held while the final channel changes from `+1`
(`CONTINUE`) to `-1` (`STOP`). This removes the former train/eval mismatch in
which padded queries were excluded from loss but consumed by the follower.
Base path channels receive higher loss weight and increasing weight toward the
last valid anchor; STOP is supervised on all queries. Upper-body output uses
the follower's attained progress anchor, never its future lookahead target.

The measured-feedback follower uses one spatial-step lookahead, tolerance at
most `0.006 m`, forward path projection so a missed anchor cannot deadlock
progress, and a spatial-step-scaled translation gain corresponding to a fixed
`0.25 m/s` nominal external execution speed. The model still predicts no time,
duration, velocity, or rate. Its decoder remains a fixed external controller,
so the correct claim is rate-free output representation, not dynamics- or
control-rate-invariant behavior.

The full four-GPU run is gated by schema/SE(2)/frequency checks, the oracle
spatial sweep, and a 32-sample tiny overfit. After 50,000 steps, a fixed 512-
sample physical-space audit must pass endpoint translation `<0.05 m`, endpoint
yaw `<0.10 rad`, first-anchor reverse fraction `<0.02`, and STOP accuracy
`>0.90` before any closed-loop evaluation is allowed.

## 26. B2 V4 SE(2)-increment and geometry-loss correction — 2026-08-22

B2 V3 completed 50,000 steps with finite optimization but did not pass its
pre-declared physical-space gate. On 512 fixed validation samples its mean
endpoint translation error was `0.07217 m`, endpoint yaw error was
`0.13116 rad`, first-anchor reverse fraction was `0.046875`, and STOP accuracy
was `0.9653`. A sweep of checkpoints 5,000 through 50,000 showed these errors
already near a plateau by roughly step 10,000, so simply adding optimizer steps
was rejected. No closed-loop success rate was claimed for V3 because the
offline gate blocked evaluation.

B2 V4 retains the controller-resolvable V3 scale (`0.025 m/anchor`, eight
anchors, maximum lookahead `0.20 m`) and the same 12D whole-body token layout,
but changes the base parameterization. Instead of predicting eight absolute
poses sharing the chunk origin, it predicts body-frame SE(2) Lie increments

\[
\xi_k = \log\!\left(T_{k-1}^{-1}T_k\right), \qquad T_0=I,
\]

and reconstructs the path by

\[
\hat T_k = \hat T_{k-1}\exp(\hat\xi_k).
\]

This makes each token a local geometric transition while preserving the exact
absolute path under encode/decode. The other nine channels remain synchronized
whole-body commands at the attained geometric anchor: control mode, EEF
position, EEF rotation, gripper, and explicit CONTINUE/STOP. The model still
does not predict timestamp, duration, velocity, or rate.

Training keeps weighted token L1 and CVAE KL, and adds differentiable physical-
space terms after unnormalizing and composing the predicted increments:

- endpoint translation distance;
- periodic endpoint-yaw error `1 - cos(delta yaw)`;
- first-anchor direction cosine loss;
- adjacent-increment smoothness relative to the demonstrated path;
- per-segment SE(2) metric-progress error.

The geometry weights are respectively `1.0`, `0.2`, `0.05`, `0.25`, and `1.0`,
with yaw radius `0.25 m`. These are fixed before training and must not be tuned
from closed-loop seeds.

The uniquely named four-GPU pipeline is
`act-b2-se2delta-geometryloss-wholebody-v4-navigate-kitchen-scratch-20260822`.
It first repeats the oracle spatial sweep and schema/SE(2)/10-20-40 Hz label
gates, then requires a 32-sample, 500-step tiny overfit. Formal training uses
batch 32 per GPU (global 128). Stage 1 trains to step 10,000 and is allowed to
continue only when a fixed 256-sample audit beats the V3 plateau: endpoint
translation `<0.065 m`, endpoint yaw `<0.12 rad`, first-anchor reverse fraction
`<0.03`, and STOP accuracy `>0.90`. If it passes, training resumes the same
model and optimizer continuously for 40,000 more steps, rather than resetting
the optimizer. The final fixed 512-sample gate remains `<0.05 m`, `<0.10 rad`,
`<0.02`, and `>0.90` respectively. Closed-loop evaluation is forbidden unless
that final gate passes.

## 27. B2 V4 10k result and Diffusion Policy ablation — 2026-08-22

B2 V4 reached its planned 10,000-step checkpoint normally, but the fixed
256-sample physical audit did not authorize continuation. Mean endpoint
translation error was `0.072428 m` against the `<0.065 m` stage-1 threshold;
mean endpoint yaw error was `0.126500 rad` against `<0.12 rad`; first-anchor
reverse fraction was `0.058594` against `<0.03`. STOP accuracy passed at
`0.958008`. Predicted and target mean path lengths were close (`0.167460 m`
versus `0.164596 m`), indicating that output magnitude/progress was learned
while directional precision remained inadequate. The 40,000-step continuation
was therefore blocked and all four GPUs were released. This is an offline
representation/modeling result, not a closed-loop success rate.

The next controlled ablation replaces ACT with LeRobot Diffusion Policy while
holding the following fixed: the seed-20260820 train/validation split, three
256x256 cameras, 29D state including the 13-way goal ID, eight spatial tokens,
12D whole-body token schema, `0.025 m/anchor` SE(2) metric, body-frame Lie
increments, explicit CONTINUE/STOP, and all physical audit thresholds. It is a
scratch-trained conditional diffusion model, not an ACT checkpoint conversion.

To isolate policy family rather than silently create a new loss ablation, this
experiment uses the stock diffusion epsilon-prediction MSE objective and does
not reuse ACT V4's deterministic geometry loss or CVAE KL. The installed
LeRobot configuration has one observation step, horizon eight, DDPM with 100
training/inference steps, a shared random-initialized ResNet-18 visual encoder,
and a conditional 1D U-Net with widths `(512,1024,2048)`. It contains
`265,296,808` parameters. AdamW uses learning rate `1e-4` with 500-step warmup
and a cosine schedule declared for 50,000 steps.

Before formal training, the run must repeat label/SE(2)/frequency gates, pass a
32-sample 1,000-step tiny overfit in both noise loss and decoded physical path,
and pass a four-GPU global-batch-128 memory smoke. Formal stage 1 is 10,000
steps at batch 32 per GPU only if those gates pass. Its 256-sample continuation
thresholds are identical to B2 V4. A passing model resumes the same optimizer
and cosine scheduler for 40,000 additional steps; a failing model stops without
closed-loop evaluation. The uniquely named run is
`dp-b2-se2delta-ratefree-wholebody-v1-navigate-kitchen-scratch-20260822`.

The DP run subsequently completed stage 1 at step 10,000 in 1,184 seconds with
finite loss and gradients, but it also failed the fixed physical continuation
gate. On the fixed 256-sample validation set, mean endpoint translation error
was `0.078998 m`, endpoint yaw error was `0.160985 rad`, first-anchor reverse
fraction was `0.09375`, and STOP accuracy was `0.958008`. Path magnitude was
again close (`0.164625 m` predicted versus `0.160431 m` target), while
directional precision was worse than ACT V4. To rule out one unlucky diffusion
sample, the same checkpoint and exact same 256 examples were audited with four
noise seeds. Translation error ranged `0.078998--0.080419 m`, yaw error
`0.154828--0.160985 rad`, first-anchor reverse fraction
`0.089844--0.105469`, and STOP accuracy `0.953613--0.963867`. The failure is
therefore stable across sampled diffusion noise, not a single-draw artifact.
The 40,000-step continuation and closed-loop evaluation were blocked, and all
GPUs were released. This ablation rejects the hypothesis that replacing ACT's
CVAE with stock Diffusion Policy alone solves the B2 directional ambiguity.

## 28. Clean restart: common-origin labels and base-only ACT diagnostic — 2026-08-23

The representation discussion was restarted from the raw data contract rather
than extending B2 V4. For any geometric chunk constructed at observation index
`t`, every future pose must use the same observed base frame as its origin:

\[
P_k={}^{B_t}T_{B_{t+k}}=T_t^{-1}T_{t+k}.
\]

The target at query `k` must not silently change its reference frame to
`B_{t+k-1}`. Adjacent transforms may be derived later for numerical encoding or
control, but they are not independently grounded labels. A new replan at index
`t'` creates a new chunk with the single new origin `B_{t'}`.

The live NavigateKitchen task source was re-audited. Its success predicate uses
only the measured mobile-base pose: planar distance to the target placement
must be at most `0.20 m`, and `cos(target_yaw - base_yaw) >= 0.98`. It does not
inspect torso, end-effector, arm, or gripper state. This authorizes a clean
base-only diagnostic without changing task semantics.

The 500-episode / 72,786-frame dataset is timestamped at a stable `20 Hz`.
Action channels `0:3` are the normalized chassis commands `(vx, vy, omega)`;
action channel `3` is the torso component and is identically zero. During
base-only execution, the fixed native hold contract is therefore:

```text
base_motion = [predicted vx, predicted vy, predicted omega, 0]
control_mode = +1
end_effector_position = [0, 0, 0]
end_effector_rotation = [0, 0, 0]
gripper_close = -1
```

`control_mode=+1` selects the base-mode update rule in RoboSuite's
`HybridMobileBase`; the zero end-effector deltas keep the arm target fixed while
the base moves. These nine non-chassis values do not participate in the model
loss.

A 10,000-window audit of real 50-frame trajectories showed that 50 frames at
20 Hz cover 2.5 seconds and are too long to treat as an automatically local
path: median SE(2) metric path length was `1.3578 m-equivalent`, median endpoint
translation was `1.1200 m`, and median absolute endpoint yaw was `1.2180 rad`.
Loss-bounded adaptive polyline compression is more appropriate than selecting a
fixed spatial interval before seeing the data. With maximum interpolation error
`0.01 m / 0.02 rad`, the required point count had median `9`, 95th percentile
`13`, 99th percentile `15`, and sampled maximum `19`. For comparison, fixed
equal-arc-length reconstruction used 16 points with 95th-percentile maximum
error `0.0230 m / 0.0449 rad`, and 32 points with `0.0104 m / 0.0197 rad`.
These are data-audit results, not a final method hyperparameter; the full split
and oracle follower must be checked before selecting the geometric output.

The first controlled experiment is a scratch base-only ACT baseline, retaining
the previously successful ACT family and its native time-indexed representation.
It uses the same three cameras, 29D state and task conditioning, 32 queries,
eight-action replan prefix, global batch 128, and 50,000 optimizer steps, but
predicts only the three chassis commands. Because the source data and simulator
demonstrations are natively 20 Hz, ACT training and its primary closed-loop
evaluation remain on the native 20 Hz grid. The geometric branch also derives
labels directly from the native measured 20 Hz poses; it is not converted into
a synthetic 30 Hz training sequence. Only its measured-pose path follower is
calibrated and executed at 30 Hz. To separate representation from controller
frequency in the eventual ablation, the same geometric checkpoint must also be
evaluated with a 20 Hz follower; ACT itself is not retrained or resampled to
30 Hz.

The new unique run root is
`/workspace/act-base-only-navigate-kitchen-scratch-20260823`, managed by
supervisor service `act-base-only-navigate-50k`. The 32-sample, 500-step tiny
overfit gate reduced mean early L1 from `0.8522` to mean late L1 `0.3630` and
passed. Formal four-GPU training then started normally with all GPUs active.
Checkpoint, finite loss, and GPU utilization remain training-health evidence;
fixed-seed closed-loop results are still required before claiming success.

## 29. Train-only adaptive geometry audit and 30 Hz execution boundary — 2026-08-23

The clean restart discards the previous fixed eight-anchor, `0.025 m/anchor`,
`0.10 m` label horizon, and timestamp-fitted horizon choices. New geometric
labels use the measured base position and `xyzw` quaternion. For a sample at
`t`, the raw future curve is always

\[
P_{t,j}=T_t^{-1}T_j,
\]

so all retained anchors share the same current-base origin. Command velocity
is not integrated to produce a target.

Segmentation is now defined by physical reconstruction error rather than time,
frame count, speed, or a fixed travelled distance. A geometric RDP procedure
projects each measured pose onto a candidate SE(2) chord using only
`[x, y, r theta]` geometry; it never interpolates by sample index. Starting
from the current identity pose, the label retains the first `K` future knots.
If the complete remaining route needs more than `K` knots, the covered horizon
ends at the `K`-th knot; otherwise the label reaches the terminal route pose and
pads the unused model queries. Thus the physical horizon becomes shorter on a
complex turn and longer on an easy straight path while obeying the same error
contract.

The complete fixed seed-20260820 training split was audited: 450 episodes and
65,834 valid start states. Validation and test episodes were not used. For all
43,785 valid 50-frame windows, the number of points required by geometric
compression was:

| Maximum reconstruction error | median | p95 | p99 | maximum |
|---|---:|---:|---:|---:|
| `0.005 m / 0.01 rad` | 11 | 16 | 18 | 23 |
| `0.01 m / 0.02 rad` | 8 | 11 | 13 | 17 |
| `0.02 m / 0.05 rad` | 6 | 8 | 9 | 12 |

The suffix-to-route-end audit used every training start state. At
`0.01 m / 0.02 rad`, the complete future route required a median of 10, p95 of
24, p99 of 29, and maximum of 36 future knots. A 32-query ACT head therefore
covers the complete remaining route at 99.87696% of training start states. If
the 32-query budget is exhausted, the segment still ends at a truthful
geometric knot with the same bounded reconstruction error. The covered measured
path length has median `1.9242 m-equivalent` and p95
`4.5049 m-equivalent`. Source-frame and source-second coverage are recorded
only as diagnostics and never enter the label.

Decimating the measured curve from 20 Hz to 10 Hz changes the retained knot
count by a median of zero and p95 of two at `0.01 m / 0.02 rad`. When the 10 Hz
knots reconstruct the original 20 Hz samples, p95 error is `0.01041 m` and
`0.01994 rad`; maxima are `0.01632 m` and `0.02623 rad`. These maxima reflect
information removed by decimation and are not claimed as exact label
invariance. Synthetic re-sampling and oracle reconstruction remain required.

Implementation is separated into three modules:

- `adaptive_path.py`: common-origin, error-bounded geometric knot selection;
- `geometric_follower.py`: current-pose feedback, with the measured current
  pose explicitly prepended as the path origin;
- `calibrate_base_30hz.py`: physical pulse identification of normalized native
  base commands to measured body twist per second at real simulator 30 Hz.

The simulator has been directly instantiated with `control_freq=30` and reports
`control_timestep=0.0333333333 s`. Thirteen focused schema, adaptive-path,
follower, and calibration tests pass. This is still an interface/controller
gate, not policy success. Before geometric-policy training, the 30 Hz pulse fit,
measured tracking trial, frequency re-sampling gate, oracle reconstruction,
and tiny-set overfit must all pass. Native ACT remains trained and evaluated on
its original 20 Hz time grid.

## 30. Fixed-token labels, final native ACT baseline, and formal geometric run — 2026-08-23

The fixed model interface is now `K=32` common-origin absolute relative poses:

\[
Y_t=[P_{t,1},\ldots,P_{t,32}],\qquad P_{t,j}=T_t^{-1}T_j.
\]

RDP first selects future measured-pose knots under the physical
`0.01 m / 0.02 rad` reconstruction contract. If the remaining route requires
fewer than 32 future knots, every selected knot is preserved and the longest
SE(2) chords are bisected until exactly 32 supervised poses exist. If more than
32 knots are required, the label ends at the 32nd retained knot. There is no
padding mask, STOP prediction, timestamp, duration, velocity, rate, fixed
distance horizon, or command-velocity integration in the target.

The full 72,786-frame cache has shape `[72786,32,3]` and was built from native
20 Hz measured position plus `xyzw` quaternion. Training still uses only the
65,834 frames in the fixed 450-episode train split, and normalization statistics
were fitted only on those train labels. Across the complete cache, the maximum
source-curve reconstruction errors are `0.0099999000 m` and
`0.0199972791 rad`; 99.8887% of all start states reach the episode terminal
within the 32-knot budget.

An independent 2,048-sample train-only validation rebuilt labels from raw
poses. Cached-vs-rebuilt maximum absolute error was `2.38e-7`; the
common-origin local-to-world-to-local round trip was below `1.82e-15 m` and
`8.89e-16 rad`. The sampled source-curve maxima were `0.00999796 m` and
`0.01998484 rad`, within the declared bounds. A 12-example visualization was
written before training. The 32-sample, 1,000-step geometric tiny overfit
reduced early mean normalized L1 from `0.78065` to late mean `0.15667`
(`0.20069x`) and passed.

The 30 Hz pulse calibration used 180 simulator samples. Its measured body-rate
fit had per-channel RMSE `[0.0530,0.0463,0.0446]` and R2
`[0.5065,0.8288,0.9660]`; the weaker forward fit is retained as a calibration
risk rather than hidden. Direct free-space follower trials at real 30 Hz all
passed the `0.02 m / 0.04 rad` terminal gate: backward `0.10 m`, lateral
`0.10 m`, rotation `0.20 rad`, and a backward curve. This establishes local
controller tracking only, not NavigateKitchen policy success.

The final native base-only ACT baseline completed 50,000 scratch optimizer
steps. Its fixed-seed target-split closed loop completed all 30 episodes and 30
videos without evaluation failure: `4/30 = 13.33%`. The earlier matched
whole-body scratch ACT result was `3/30 = 10%`; a one-episode difference is not
treated as evidence that base-only is better. It shows that the scratch ACT
capability baseline is low but nonzero.

Formal geometric ACT training is a scratch, four-GPU, 50,000-step run with the
same three cameras, 29D state, random ResNet-18 ACT backbone, 32 queries,
learning rate `1e-5`, per-GPU batch 32, and global batch 128. The unique run
root is `/workspace/act-geometric-navigate-kitchen-scratch-20260823` and the
supervisor service is `act-geometric-navigate-50k`.

The primary geometric closed-loop protocol uses a real 30 Hz simulator and
measured-pose feedback every control tick. Model replanning occurs every 12
ticks, preserving the baseline's `0.4 s` replan period, and the horizon is 675
ticks, preserving the baseline's `22.5 s` wall-clock budget. The policy holds
torso/arm/gripper at the verified native defaults. A preflight must assert the
environment reports exactly 30 Hz before the 30 fixed seeds are counted.

## 31. Rejected wrapped-yaw run and continuous-yaw correction — 2026-08-23

The first formal fixed-token run was stopped and rejected at optimizer step
12,100 after an additional label-periodicity audit. Although every pose had the
correct common base origin, the first cache wrapped each output yaw
independently into `[-pi,pi)`. Across the 72,786-frame cache, 9,071 samples had
at least one adjacent-token jump larger than `pi`; there were 11,123 such jumps
and the maximum was approximately `2pi`. Ordinary normalized L1 therefore
treated physically adjacent, equivalent orientations as far apart. Its
checkpoint and logs remain preserved under
`/workspace/act-geometric-navigate-kitchen-scratch-20260823` but must not be
used as the method result.

The corrected target keeps measured relative yaw continuously unwrapped along
the future path. This does not add time or rate and does not change the shared
origin:

\[
P_{t,j}=(x_{t,j},y_{t,j},\tilde\theta_{t,j}),
\]

where `tilde theta` is the continuous lift of the measured heading sequence
starting at zero in `B_t`. The follower remains periodic when computing actual
heading error. The corrected cache spans roughly `[-6.406,6.450] rad`, has zero
adjacent jumps above `pi`, and its maximum adjacent yaw change is
`0.7574 rad`. A second bug was also caught by the unit gate: sparse retained
knots must not be passed through `unwrap` a second time because two valid sparse
knots may differ by more than `pi` even though the dense source curve is
continuous.

Twelve focused tests pass after the correction. The full label and offline
reconstruction gates pass with unchanged physical maxima. The corrected
32-sample tiny overfit reduced early mean L1 from `0.77259` to `0.12859`
(`0.16644x`). The new unique run root is
`/workspace/act-geometric-continuousyaw-navigate-kitchen-scratch-20260823`;
formal training restarts from scratch and no optimizer/model state from the
rejected run is reused.

When stopping the rejected supervisor job, its outer shell exited but the
`torchrun` process tree remained orphaned. Those processes were identified by
the exact rejected-run path and terminated without touching the corrected run.
GPU memory returned from approximately 10.8 GB to 5.4 GB per card. Relevant
supervisor configs now use process-group termination so a future stop also
reaps child ranks and data workers.

## 32. Corrected geometric ACT final result — 2026-08-23

The continuous-yaw common-origin geometric ACT run completed 50,000 scratch
optimizer steps on four GPUs with global batch size 128. No rejected-run model
or optimizer state was reused. The final 5,000-step window had mean total loss
`0.04228`, mean normalized L1 `0.04182`, and mean gradient norm `2.8067`; all
logged values were finite. The model-only checkpoint and independent
`TRAINING_COMPLETE` markers were both verified under
`/workspace/act-geometric-continuousyaw-navigate-kitchen-scratch-20260823`.
Training health alone was not treated as policy success.

Before counted evaluation, the simulator emitted
`ENVIRONMENT_30HZ_VERIFIED` with `control_hz=30.0` and
`timestep_s=0.03333333333333333`. The counted protocol then used the fixed
target-split seeds `20260818..20260847`, 675 control ticks per episode
(`22.5 s`), measured-pose feedback every tick, and model replanning every 12
ticks (`0.4 s`). This matches the native baseline's wall-clock horizon and
replan period while intentionally testing the geometric follower at 30 Hz.

All 30 geometric episodes, all 30 non-empty videos, `results.json`,
`summary.json`, and `EVALUATION_COMPLETE` were verified. The corrected method
achieved `5/30 = 16.67%`; successful seeds were `20260822`, `20260823`,
`20260826`, `20260836`, and `20260843`. The native 20 Hz base-command ACT
baseline achieved `4/30 = 13.33%` on the same seeds; its successful seeds were
`20260820`, `20260836`, `20260841`, and `20260844`.

The paired contingency is therefore: one seed successful for both methods,
four successful only for the geometric method, three successful only for the
native baseline, and 22 failed for both. The observed difference is only one
episode (`+3.33` percentage points). This run proves that the corrected
common-origin, rate-free path representation plus measured-pose 30 Hz follower
is executable and non-zero; it does **not** establish a statistically reliable
improvement over ACT. The next research experiment should use more seeds and/or
replicated training runs before making a performance claim, while preserving
the same labels, controller contract, and paired evaluation protocol.

After evaluation, the image transport boundary was hardened by copying decoded
camera arrays into owned writable C-contiguous `uint8` buffers before
`torch.from_numpy`. This removes PyTorch's read-only-buffer warning without
changing the evaluated tensor values. The synchronized Mac, zeno-rp, and Vast
source then passed Python compilation and the same 15 focused tests.

## 33. Matched 10/20/50 Hz closed-loop sweep — 2026-08-24

The completed base-only ACT and corrected common-origin continuous-yaw
geometric checkpoints were held fixed for a matched control-frequency sweep;
there was no frequency-specific retraining. Every condition used
`NavigateKitchen`, the target split, the same 30 seeds `20260818..20260847`,
and a 22.5-second wall-clock horizon. The simulator rates were verified as
10/20/50 Hz (`dt=0.1/0.05/0.02 s`), giving horizons of 225/450/1125 ticks.
The model replanning period remained 0.4 seconds, giving 4/8/20 replan ticks.

The native baseline's learned output remains a 20 Hz velocity-command chunk.
For target-rate execution it is converted using exact overlap-weighted
zero-order-hold interval averaging. This preserves the commanded velocity
integral over each 0.4-second replan interval and is exact identity at 20 Hz;
it does not silently reinterpret commanded motion as measured physical motion.
The geometric model uses the same time/rate-free path output at every rate and
follows it from measured SE(2) pose each control tick. It predicts no time,
duration, velocity, or rate. Upper-body state is held at the already verified
native defaults for both methods.

Rate-specific command-to-measured-twist calibration and four free-space
follower trials passed at all three rates before counted evaluation. The
counted closed-loop results were:

| Method | 10 Hz | 20 Hz | 50 Hz | Mean across conditions |
|---|---:|---:|---:|---:|
| native command ACT | `3/30` (10.0%) | `4/30` (13.33%) | `5/30` (16.67%) | 13.33% |
| geometric Path-RateFree ACT | `8/30` (26.67%) | `6/30` (20.0%) | `5/30` (16.67%) | 21.11% |

At 10 Hz the paired table was one both-success, seven geometric-only, two
baseline-only, and 20 both-fail. At 20 Hz it was `1/5/3/21`; at 50 Hz it was
`1/4/4/21`. The corresponding two-sided exact McNemar p-values were `0.1797`,
`0.7266`, and `1.0`. Five seeds succeeded for the geometric method at all three
rates (`20260822`, `20260829`, `20260835`, `20260836`, `20260843`), versus two
for the baseline (`20260836`, `20260841`).

All six conditions have exactly 30 ordered unique seeds, 30 non-empty videos,
`results.jsonl`, `results.json`, `summary.json`, and `EVALUATION_COMPLETE`.
Across the 180 counted episodes there was no `EVALUATION_FAILED` and no logged
Traceback, OOM, NCCL, or NaN match. Small results, controller gates, preflight
evidence, a structured paired analysis, and the completion audit are preserved
under `artifacts/20260824/frequency-sweep`.

This sweep shows that the corrected geometric policy/follower remains
executable and non-zero at 10, 20, and 50 Hz, with its largest observed margin
at 10 Hz and a tie at 50 Hz. It does not yet prove frequency invariance,
performance superiority, or novelty: only 30 paired seeds were used, all
confidence intervals are wide, and none of the between-method differences is
statistically reliable. A publication-grade next step is replicated training
seeds plus a denser frequency grid and an execution-only oracle-path control,
while keeping the same wall-clock/replan/seed protocol.

## 34. Deployment-frequency interpretation correction and native ACT control — 2026-08-24

The research question was clarified after Section 33: training data remains
fixed at 20 Hz; the intended claim concerns sensitivity to **deployment control
frequency**, not data-sampling-frequency invariance. Section 33's baseline had
already been given an explicit frequency-aware temporal adapter. It is therefore
renamed conceptually to **Time-corrected ACT** and must not be presented as
ordinary native ACT deployment.

A separate native ACT control was added and evaluated. It uses the exact same
base-only ACT checkpoint, binds one learned action token to one environment
control tick, executes eight tokens before replanning, and performs no temporal
resampling. Consequently its eight-token replan interval is 0.8, 0.4, and 0.16
seconds at 10, 20, and 50 Hz. Episode duration remains fixed at 22.5 seconds,
and the task, target split, checkpoint, fixed seeds `20260818..20260847`, and
upper-body hold remain unchanged. Its 20 Hz result exactly reproduces the
original baseline success set, validating the native deployment control.

The complete deployment-frequency table is:

| Deployment semantics | 10 Hz | 20 Hz | 50 Hz | Mean | Range |
|---|---:|---:|---:|---:|---:|
| Native ACT, token equals tick | `2/30` (6.67%) | `4/30` (13.33%) | `6/30` (20.0%) | 13.33% | 13.33 pp |
| Time-corrected ACT | `3/30` (10.0%) | `4/30` (13.33%) | `5/30` (16.67%) | 13.33% | 6.67 pp |
| Path-RateFree | `8/30` (26.67%) | `6/30` (20.0%) | `5/30` (16.67%) | 21.11% | 10.0 pp |

Native ACT succeeds at every rate on two of the seven seeds that succeed at any
rate (`28.57%` persistence), Time-corrected ACT on two of six (`33.33%`), and
Path-RateFree on five of nine (`55.56%`). Mean pairwise success-set Jaccard is
`0.421`, `0.528`, and `0.671`; relative success-rate CV is `0.408`, `0.204`, and
`0.197`, respectively. Native ACT therefore shows the largest observed
deployment-rate sensitivity, while Path-RateFree has the strongest successful-
seed persistence and the highest mean success.

This result requires a precise claim boundary. `Path-RateFree` refers to the
policy output: it predicts spatial path geometry and does not require temporal
action-token resampling. The complete deployment stack is not frequency-blind:
the measured-pose follower, calibration, slew limit, discrete update, and
scheduler still use the real control `dt`. Moreover, Time-corrected ACT has the
smallest absolute success-rate range in this 30-seed experiment. The defensible
claim is therefore that Path-RateFree provides rate-free **action semantics**
and stronger observed cross-rate successful-seed persistence than native ACT,
not that it is already proven insensitive relative to every frequency-aware ACT
adapter.

All nine conditions contain exactly 30 ordered unique seeds and 30 non-empty
videos, for 270 counted episodes and videos. Every condition has complete JSON
artifacts and `EVALUATION_COMPLETE`; no evaluation-failure marker or logged
Traceback, OOM, NCCL, or NaN match was found. The structured v2 analysis and
audit are stored under `artifacts/20260824/frequency-sweep`.

## 35. Asynchronous model re-prediction-frequency sweep — 2026-08-24

The research variable was refined from simulator control frequency to model
re-prediction request frequency. The final protocol fixes the simulator and
low-level controller at 50 Hz and requests a new policy prediction at 10, 20,
or 50 Hz. There is one inference worker and no queue: a due request is dropped
when the worker is busy, while the low-level controller continues the most
recent fully completed result.

The asynchronous ACT control keeps its native 20 Hz command-time semantics. A
completed chunk is aligned to its observation-capture tick, stale prefix time
is skipped, and exact overlap-weighted ZOH interval averaging produces the 50
Hz command. The Path-RateFree branch anchors each completed local path using
the pose captured with its observation, not the later inference-completion
pose. It projects the current measured pose onto that world path and replaces
the plan without resetting the previous command, avoiding artificial braking
at high request rates.

RoboCasa rendering runs slower than real-time even though its declared control
timestep is exactly 0.02 seconds. Therefore GPU busy duration is measured in
wall time but applied on the simulated 50 Hz control clock. An initial pilot
that used simulator wall progress directly was preserved but rejected from the
final comparison. A second audit also found that four-concurrent and
two-concurrent waves had different P95 latency. The final table uses exactly
two concurrent simulator/model conditions at every frequency; latency P95 is
matched within about 1.5 ms between methods at each rate.

On fixed seeds `20260818..20260847`, the final matched closed-loop results are:

| Requested model inference | Time-aligned ACT | Path-RateFree |
|---:|---:|---:|
| 10 Hz | `3/30` (10.00%) | `6/30` (20.00%) |
| 20 Hz | `3/30` (10.00%) | `8/30` (26.67%) |
| 50 Hz | `3/30` (10.00%) | `7/30` (23.33%) |

Achieved completed inference rates were ACT/Path `9.95/9.95`,
`18.41/18.45`, and `36.26/34.44` predictions per simulated control second.
At a 50 Hz request rate, drop-on-busy discarded `27.33%/30.96%` of due
requests, so neither model actually completed 50 predictions per second.

The paired exact McNemar p-values at 10/20/50 Hz are `0.4531`, `0.1797`, and
`0.3438`. A seed-clustered sign-flip test over the repeated three-rate outcomes
gave approximately `p=0.159`. Thus Path-RateFree has a higher observed nominal
success rate at every request frequency in this run, but the difference is not
statistically reliable.

Crucially, this experiment does not support the stronger claim that
Path-RateFree is less sensitive to model re-prediction frequency. ACT is exactly
`10%` at all three rates (0 percentage-point range), whereas Path-RateFree is
`20/26.67/23.33%` (6.67-point range). ACT also has higher all-rate successful-
seed persistence (`2/4` versus `4/11`) and mean success-set Jaccard (`0.667`
versus `0.500`). The defensible current result is higher nominal Path-RateFree
task success under asynchronous scheduling, not demonstrated frequency
invariance.

The selected six conditions contain exactly 180 ordered fixed-seed results and
180 non-empty videos. All JSON and completion markers were verified; no selected
run contains an evaluation-failure marker, Traceback, OOM, NCCL, NaN, or ACT
chunk exhaustion. The detailed protocol, latency/drop metrics, paired tables,
and claim boundary are recorded in
`artifacts/20260824/async-inference-sweep/FINAL_RESULTS.md`.

## 36. Frozen-plan geometric rate and stale-alignment gates — 2026-08-24

The research story was narrowed from generic "time invariance" to two
separable executor claims: preserve one predicted spatial path when its
requested traversal speed changes, and align a stale plan to actual robot
progress rather than blindly restarting its time index. Existing base-only
native ACT and continuous-yaw Point checkpoints were held fixed; neither model
was retrained. These gates isolate representation/execution and are not
closed-loop task-success evaluations.

Two early pilots were rejected. A cross-process same-integer-seed sweep was not
strictly paired because RoboCasa sometimes selected different fixtures and
language instances after fresh initialization. A subsequent exact-state sweep
tested only ACT's first 0.4 seconds, during which the base moved about 0.08 mm;
its apparent invariance was therefore a no-motion artifact. Both are preserved
as diagnostics but must not be cited as results.

The selected rate gate uses seeds `20260818..20260827`, exact MuJoCo-state
replay, one byte-identical model output per seed/method, 50 Hz control, and a
five-second horizon. Native ACT plays its complete 32-token / 1.6-second
velocity chunk as `u(alpha*t)` without amplitude compensation. A separate
diagnostic oracle uses `alpha*u(alpha*t)` and must not be called native ACT.
Point follows one fixed 0.40 m-equivalent world path with external progress rate
`alpha*0.25 m-equivalent/s`. Requested speeds are 0.5x, 1x, and 1.5x. All three
executors passed a 0.05 m-equivalent meaningful-motion gate at 1x.

Mean deviations from each executor's own 1x trajectory, pooling 0.5x and 1.5x,
were:

| Executor | endpoint translation | endpoint yaw | Fréchet |
|---|---:|---:|---:|
| native ACT | `0.20685 m` | `0.21296 rad` | `0.21716 m-eq` |
| compensated ACT oracle | `0.08481 m` | `0.06031 rad` | `0.08683 m-eq` |
| Point measured-pose pointer | `0.01167 m` | `0.00868 rad` | `0.02498 m-eq` |

Relative to native ACT, Point reduced endpoint deviation by `94.36%` and
Fréchet deviation by `88.50%`; it was better for all `10/10` independent
seed-level pairs on both metrics (two-sided exact sign-test `p=0.001953`). It
also reduced the same deviations by `86.25%` and `71.23%` relative to the
compensated oracle, again `10/10` seed-level pairs. The narrow supported claim
is that the geometric pointer preserves one predicted base path substantially
better under requested speed changes than velocity-token playback. This is not
yet a whole-body or task-success claim.

The stale-alignment gate uses the same exact-state and frozen-output protocol.
A locally consistent plan advances while the captured prediction is treated as
arriving after 0.4 or 0.8 seconds. Native ACT restarts the stale chunk at token
zero. Point keeps the stale path anchored at the original capture pose and
restores its pointer by projecting the current measured pose onto that world
path. Point's mean real-clock, tick-aligned RMSE was `0.02164 m-eq`, versus
`0.07055 m-eq` for ACT, a `69.33%` reduction and `10/10` seed-level wins
(`p=0.001953`).

The endpoint/progress-resampled result is intentionally different: ACT ended
`0.00159 m` from its uninterrupted reference versus Point's `0.00237 m`, and
ACT Fréchet was `0.00166 m-eq` versus Point's `0.00976 m-eq`. ACT's repeated
early prefix primarily delayed the same path, and the five-second horizon let
both controllers settle. Therefore the stale gate supports reduced
time-alignment error during execution, not superior Point terminal geometry.
It also uses a locally consistent in-flight plan, so the next causal gate must
use independently generated consecutive asynchronous predictions before any
claim about general inference mismatch.

All 10 rate seeds produced 90 non-empty videos/traces; all 10 alignment seeds
produced 60. Exact-state errors were at most `1e-12`, input/output hashes were
consistent, and no selected log contains Traceback, OOM, NCCL, or NaN. Small
artifacts and the full claim boundary are stored under
`artifacts/20260824/path-pointer-gates/FINAL_RESULTS.md`; the 150 videos remain
on Vast.

## 37. ACT-to-path-to-pointer causal ablation — 2026-08-25

The Section 36 rate gate showed that a learned Point path plus measured-pose
pointer preserves geometry under traversal-rate changes, but it did not isolate
whether the benefit came from the learned representation or the executor. A
new frozen-output ablation therefore inserted the same pointer after the
existing ACT output. ACT's normalized base commands were mapped to predicted
body twist with the measured 50 Hz calibration and integrated in SE(2). This is
a command-derived predicted path, not ground-truth physical motion.

The selected protocol uses fixed seeds `20260818..20260827`, exact MuJoCo-state
replay, one frozen output per method/seed, 50 Hz control, a five-second horizon,
and rates `0.5x/1.0x/1.5x`. Five executors were compared: native ACT, an
amplitude-compensated ACT retiming diagnostic, ACT calibrated path plus pointer,
the non-deployable measured ACT path plus pointer oracle, and learned Point plus
the same pointer. Every executor passed the 0.05 m-equivalent meaningful-motion
gate at `1.0x`.

The command-derived ACT path reconstructs actual `1.0x` ACT motion with mean
endpoint translation error `0.05242 m`, yaw error `0.04297 rad`, and Fréchet
error `0.05467 m-equivalent`. Its mean length (`0.24732 m-equivalent`) is close
to the measured path (`0.24827 m-equivalent`), but the conversion is explicitly
not exact or ground truth.

Pooling `0.5x` and `1.5x` deviations from each method's own `1.0x` path, native
ACT has normalized endpoint/Fréchet errors of `75.17%/78.98%`; compensated ACT
has `29.85%/30.60%`; ACT calibrated path plus pointer has `2.67%/4.10%`; the
measured-path oracle has `3.09%/3.82%`; and learned Point plus pointer has
`2.78%/6.20%`. ACT calibrated path plus pointer beats native ACT on all `10/10`
seed pairs for both normalized metrics, reducing their means by `96.45%` and
`94.81%` (`p=0.001953`).

The important negative result is that learned Point does not improve on ACT
calibrated path plus pointer: it wins only `4/10` normalized endpoint pairs
(`p=0.7539`) and `2/10` normalized Fréchet pairs (`p=0.1094`), with slightly
worse means. The effective causal component is currently the path/progress
execution adapter, not demonstrated retraining to a Point output. The next
task-success evaluation must therefore include native ACT, strong ACT retiming,
ACT calibrated path plus pointer, and learned Point plus pointer under matched
observations, fixed 0.4-second replanning, fixed 50 Hz control, and the same
seeds. Detailed results are stored in
`artifacts/20260825/act-path-causal-ablation/FINAL_RESULTS.md`.
