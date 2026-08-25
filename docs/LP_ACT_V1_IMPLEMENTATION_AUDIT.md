# LP-ACT V1 implementation audit — 2026-08-14

## Scope decision

Phase 1 trains only `turning_on_radio`. Both standard ACT and LP-ACT V1 use the
same 160 training episodes, 40 validation episodes, three RGB views resized to
224x224, 61D proprioception, ResNet-18 ACT backbone, 32 decoder queries, and VAE
objective. `bringing_water` and `sweeping_garage` are installed but excluded.

## Verified alignment

The recorder stores `obs[t]` with the action applied immediately afterward. The
effect of `action[t]` is measured in `obs[t+1]`; all three base dimensions have
their strongest command/measurement cross-correlation at lag +1. LP labels
therefore integrate `state[t+1, 0:3]` over interval `t -> t+1` and compare that
motion with `action[t, 0:3]`. This is an integrated measured-qvel trajectory,
not odometry or ground-truth global motion.

## Label decision

The rejected `attained` variant resampled every available short terminal window
to 32 valid anchors. For a one-interval episode tail this fabricated 32 tiny
segments. V1 instead uses fixed extent `L_sigma` with these rules:

1. End a complete chunk exactly at `L_sigma`, interpolating inside the first raw
   interval whose whole-body metric increment would overshoot it.
2. If the episode or four-second cap arrives first, keep the final incomplete
   segment as a valid terminal anchor.
3. Repeat the terminal value only in padded slots and mask those slots from the
   ACT loss.

This preserves complete-window comparability while keeping terminal labels
truthful and reconstructible.

## Validation evidence

- Synthetic tests: 10 tests covering SE(2) signs, Exp/Log, composition,
  straight/curved integration, static labels, fixed padding, overshoot clipping,
  and LP-anchor execution decoding.
- Dataset: 200 episodes, 429,928 frames, 30 Hz; split by episode modulo five.
- Oracle: 512 held-out windows, fixed-extent v2.
- Mean base translation RMSE: 0.0000295311 m.
- Mean base yaw RMSE: 0.000155540 rad.
- Mean measured-twist RMSE: 0.00180578.
- Mean joint target RMSE: 0.000419964 rad.
- Duration absolute error: 0 s.
- Padding: 70/512 windows; 442/512 use all 32 anchors.
- Report SHA-256:
  `a5ffb2432735a0d32e3bd604801b67c7e6180d33e2f27fd307b3e7a5d689f670`.

## Next falsification gate

Compute train-only normalization statistics, then require both policies to
overfit the same deterministic tiny sample set before full two-GPU training.
Failure of LP-ACT to reduce normalized anchor L1 on that set blocks scaling.

## Tiny-set result

The two policies trained concurrently on the same 16 sampled frames for 1,000
steps, batch size 16, learning rate 1e-4, seed 0. Standard ACT normalized L1
fell from 0.912077 to 0.0414183; LP-ACT V1 fell from 0.897542 to 0.0467029.
Both passed the overfit gate. Full runs use independent validation episodes and
save both periodic and best-validation checkpoints.

## Full-run configuration

Each RTX 5090 trains one policy independently. Both runs use batch size 32,
50,000 optimizer steps, learning rate 1e-5, seed 0, eight video-decoding workers,
256 fixed held-out validation samples evaluated every 1,000 steps, and a
checkpoint every 5,000 steps. This is 1.6 million sampled frames per policy,
about 4.7 passes over the 342,547-frame training split.

The first full-run launch exposed a video-input bottleneck: the source metadata
caused LeRobot to decode three unused depth streams in addition to the three RGB
streams. A symlink-only `demos_turning_radio_rgb` view now retains every physical
parquet column but declares only the three RGB video features. With batch 32 and
eight workers, its CPU-only loader benchmark reached about 7.5 steps/s; full-run
outputs were restarted under unique `v2` names.

Further profiling showed HEVC decode itself still saturated CPU and starved the
GPUs. The final training view transcodes only task-0 RGB to 224x224 H.264
all-intra video. All ten files preserve exact per-file frame counts, and every
camera sums to 429,928 frames. The view occupies about 4.9 GB and improves the
batch-32, eight-worker loader benchmark to about 11.9 steps/s. Because every
frame is independently seekable, the full runs restore global random sampling.

## Full-run input and validation correction

The all-intra view was necessary but PyAV was still not sufficient for random
training: concurrent `v4` runs sustained only about 0.39 step/s and left both
GPUs mostly idle. The environment's matching TorchCodec 0.5 + CUDA 12.8 backend
was enabled after installing the missing CUDA 12 NPP runtime. A batch-32,
100-step ACT benchmark then completed in 6.89 seconds; after first-batch startup,
the measured rate was about 21.5 step/s.

TorchCodec uses a stricter timestamp check than the source videos warrant. The
dataset tolerance is therefore 1 ms, still far below one 30 Hz frame interval.
For the two known recoverable decode failures (timestamp tolerance and invalid
packet), the loader retries the exact same sample with PyAV instead of skipping
or changing its label. Validation samples are loaded once with PyAV and cached.

LeRobot ACT's training `forward` cannot compute its KL term in evaluation mode,
because the VAE posterior is intentionally absent during inference. Validation
therefore measures masked inference-time L1 from `predict_action_chunk`; the KL
term remains unchanged in training. This is also the metric used to select the
best checkpoint.

The formal `v7` runs passed the first validation gate. At the recorded snapshot,
standard ACT had passed validations through step 6,000 and LP-ACT V1 through
step 4,000; both services remained supervised and running. The observed rates
were approximately 19.1 step/s for ACT and 12.7 step/s for LP-ACT. No PyAV
fallback had been required in either run at that snapshot.

## Remaining experiment

Let both policies reach 50,000 steps, verify final and best-validation
checkpoints, export the complete train/validation curves, and then perform the
matched offline open-loop comparison. Do not interpret normalized L1 values
across the two different action parameterizations as a task-level winner without
reconstructing both outputs into the same physical action space.
