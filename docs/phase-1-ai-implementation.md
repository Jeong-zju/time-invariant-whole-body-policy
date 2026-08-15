# Phase 1（M1）AI 实施规格

更新日期：2026-08-16

> 本文只面向执行代码、训练和远程实验的 AI agent。面向研究者的解释见[技术路线](time-invariant-whole-body-policy.md)。若两份文档冲突，以机器可读的 [`arena-g1-phase-1-m1-v1.yaml`](../configs/arena-g1-phase-1-m1-v1.yaml) 和冻结上游 commit 为准；不得靠猜测补齐接口。

## 1. 任务目标和禁止扩展

实现研究 Phase 1 / 方法 M1：保持 GR00T N1.5 的视觉语言 backbone 和 `16×32` action head 外形不变，把每行最后三维从 `vx/vy/yaw_rate` 改为“相对本次计划起点的底盘 `SE(2)` 位姿”，其余 29 维仍是 28 维上身关节/手部绝对位置和 1 维 base height。每一行都有显式的秒级查询时间。

本阶段只检验 action target/执行接口。严禁顺手加入：

- 半群/partition loss、Neural ODE、state-flow head；
- 随机重规划 schedule 训练或 closed-partition loss；
- 自适应、事件触发或 PACE 式重规划；
- 新视觉语言 backbone、更多相机或新任务；
- 每个方法单独调 tracker 增益；
- 把命令积分轨迹写成“实测 odometry”。

这些内容分别属于 Phase 2、Phase 3 或新的对照实验。M1 若已通过，停止增加复杂机制。

## 2. 开工前必须读取的文件

按顺序读取：

1. [`configs/arena-g1-phase-1-m1-v1.yaml`](../configs/arena-g1-phase-1-m1-v1.yaml)：唯一机器可读 M1 契约；
2. [`configs/arena-g1-box-pick-place-v0.yaml`](../configs/arena-g1-box-pick-place-v0.yaml)：上游版本、32/50 维 action 契约；
3. [`docs/arena-g1-bm0-deployment.md`](arena-g1-bm0-deployment.md)：远端目录、训练和闭环入口；
4. [`docs/arena-g1-gate-n-frequency.md`](arena-g1-gate-n-frequency.md)：现有频率问题、telemetry 和 seed 协议；
5. [`src/whole_body_policy/phase1.py`](../src/whole_body_policy/phase1.py) 与 [`se2.py`](../src/whole_body_policy/se2.py)：已实现并有本地测试的核心语义；
6. [`tests/test_phase1_m1.py`](../tests/test_phase1_m1.py)：不得破坏的不变量。

先运行：

```bash
python3 -m unittest discover -s tests -v
```

若远端仓库存在额外 `AGENTS.md` 或宿主机运维指南，先读它们。检查 `git status --short`、磁盘、内存和 GPU；保护用户现有 checkpoint、日志和未提交修改。

## 3. 研究调研转成的工程决策

| 来源 | 可复用事实 | M1 的处理 | 不能冒充的创新 |
|---|---|---|---|
| [Isaac-GR00T 数据准备](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_preparation.md)、[data config](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_config.md)、[Policy API](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/policy.md) | GR00T 用 LeRobot v2 + `modality.json`，action 形状为 `[B,T,D]`；`delta_indices` 改动后必须重算统计量 | 复用官方视频/语言输入和 `16×32` 头；新 target 单独重算 stats | “能输出 action chunk”不是本项目贡献 |
| [Isaac Lab Arena G1 官方闭环教程](https://isaac-sim.github.io/IsaacLab-Arena/release/0.1.0/pages/example_workflows/locomanipulation/step_4_evaluation.html) | 官方闭环是 16 行反馈、G1 WBC、头部相机和任务调优 checkpoint | 锁定原环境、WBC、checkpoint、seed 和 1,200-step evaluator | 不得说 GR00T 直接输出 43 个下肢/全身关节 |
| [AWE](https://proceedings.mlr.press/v229/shi23b.html) 与 [HYDRA](https://openreview.net/forum?id=_A15qsPswaK) | waypoint/relabeling 已是成熟 imitation-learning 路线 | M1 把 learned whole-body position waypoint 当强而便宜的第一假设 | “把速度换成 waypoint”本身不是 novelty |
| [NIAF](https://arxiv.org/abs/2603.01766)、[Spline Policy](https://arxiv.org/abs/2606.07386)、[B-spline Policy 论文](https://arxiv.org/abs/2607.09648)及其[官方代码](https://github.com/B-spline-policy/bspline-policy) | 连续 action function、解析导数、任意时间查询已有优秀方案 | M1 暂用离散但带真实时间的路线图；这些进入 B3/Phase 2 强基线 | 不得把“连续可查询”当首次提出 |
| [Neural Dynamic Policies](https://www.cs.cmu.edu/~sbahl2/ndps/) 与 [Autonomous NDP](https://arxiv.org/abs/2305.12886) | 动力系统 action representation、稳定/反应性已有先例 | 只在 M1 失败后作为 B4/Phase 2 比较 | 不得把 Neural ODE/自主流本身写成 novelty |
| [RTR 官方代码](https://github.com/tars-robotics/RTR)、[LeRobot RTC 示例](https://github.com/huggingface/lerobot/blob/main/examples/rtc/eval_with_real_robot.py)、[PACE](https://arxiv.org/abs/2606.00537) | action frequency、异步 chunk seam 和执行 horizon 会显著影响结果 | telemetry 必须记录 plan age、激活时间和边界 jump；固定执行规则 | M1 不能把所有收益都归因于模型 target |

关键判断：

- `[事实]` 当前 Arena LeRobot 数据没有实测 root `SE(2)`，只有导航命令；上身 28 维是 Policy 输出，43 个最终关节不是全由 GR00T 预测。
- `[判断]` 第一版保持 `16×32` 能最大限度复用 checkpoint 和公平训练，但“输出维度相同”不代表旧 action statistics 可复用。
- `[判断]` 命令积分代理只适合数据 smoke。若 320 ms 局部窗口无法贴合 expert replay odometry，就必须补 measured root trajectory，不能调松门槛后继续。
- 该判断可能错的条件：如果 Arena recorder 能从原始生成数据无损恢复每帧 root pose，直接走 measured-odometry 分支并删除 proxy 依赖。

## 4. 精确 action 契约

M1 模型输出仍为：

```text
shape [batch, 16, 32]

0:7    left arm absolute position
7:14   right arm absolute position
14:21  left hand absolute position
21:28  right hand absolute position
28:29  base height absolute position
29:32  base relative SE(2) pose [dx_body, dy_body, dyaw]
```

16 行查询时间固定为：

```text
[0.02, 0.04, 0.06, ..., 0.30, 0.32] seconds
```

`base_relative_se2[j]` 必须表示从**同一个本次计划 anchor** 到 `anchor_time + query_times[j]` 的群上相对位姿。不能让第 `j` 行相对第 `j-1` 行，也不能把三维值留作速度却改字段名。

这里的 whole-body 是 Policy/WBC 分层意义：M1 同时给上身位置、base height 和底盘位姿路线；Homie v2/WBC 仍生成下身关节。代码、文档和论文都不得写成“VLA 端到端输出 43 关节”。

## 5. Step A：先建立数据真值，不得先启动训练

### A1. 从 expert replay 导出逐帧数据

每个 episode 导出一个 NPZ，最低字段为：

```text
timestamps_s              [N]       严格递增的真实/仿真秒
upper_body_position       [N, 28]   实测关节/手部位置，顺序匹配模型 action
base_height               [N]       实测优先；没有时明确记录 command proxy
base_pose_se2             [N, 3]    root x/y/yaw 实测 telemetry
base_twist_body           [N, 3]    专家 navigate vx/vy/yaw_rate
episode_index             scalar
```

不要从 Policy rollout 反向制造 expert label。使用 demonstration replay；Policy rollout 只用于评测。

### A2. 验证命令积分代理

先对每个 episode 运行：

```bash
python3 scripts/research_phase_1/validate_base_proxy.py \
  --input artifacts/phase_1_m1/replay/episode-000.npz \
  --output artifacts/phase_1_m1/data-gate/episode-000.json \
  --horizon-s 0.32 \
  --max-xy-p95-m 0.05 \
  --max-yaw-p95-rad 0.08
```

聚合时要求全部选定训练 episode 都有报告，不得只保留通过的 episode。任一主数据子集明显失败时：

```text
decision = capture_measured_root_odometry_and_rebuild_targets
```

此时仍可用 `base_pose_se2` 真值继续，不允许把 proxy gate 调宽到通过。阈值是本项目在看结果前冻结的工程容差，不是论文常数。

### A3. 构造多跨度 target

优先执行 measured 分支：

```bash
python3 scripts/research_phase_1/build_m1_targets.py \
  --input artifacts/phase_1_m1/replay/episode-000.npz \
  --output artifacts/phase_1_m1/targets/episode-000.npz \
  --report artifacts/phase_1_m1/targets/episode-000.json \
  --source measured_odometry
```

只有 proxy data gate 通过后才可将 `--source` 改为 `command_integration_proxy`。生成器必须：

- 用秒查询未来状态，而不是默认 frame index 等于时间；
- 上身/base height 线性插值，底盘用 SE(2) `Log/Exp`；
- episode 末尾 horizon 不足的 anchor 直接删除，不复制最后一帧；
- artifact 写入 `source`，后续训练/报告保留来源；
- 当前 image/language/state 是输入，未来 observation 只用于 label，绝不进入输入。

## 6. Step B：接入 GR00T 训练

### B1. 不要用错误的逐行 parquet 重写

相对底盘 target 依赖当前 anchor。同一个未来全局 pose 对不同 anchor 有不同的相对值，因此不能简单把每个 parquet row 的 `navigate_command` 离线替换一次，再让标准 `delta_indices` 拼 horizon。这会让第 2–16 行相对各自的 row，而不是同一计划起点。

可接受实现二选一：

1. 自定义 GR00T episode dataset wrapper：给定 anchor，一次取当前 observation 和 16 个未来 measured states，调用 `build_multi_horizon_targets` 返回 `[16,32]`；推荐。
2. 在派生数据中存全局 `base_pose_se2`，loader 同时取得 anchor pose 和未来 pose，进入 normalization 前统一转换成 anchor-relative `SE(2)`；不要把 episode 的任意全局坐标送进模型。

无论选哪一种，都要复用官方视频、语言、augmentation 和 train split。不得把 16 个 horizon 展开成 16 倍“独立样本”后暗中增加优化 sample budget。

### B2. 统计量和 mask

- 对派生的 `[16,32]` M1 target 重算 action min/max 或项目采用的统计量；
- 旧 `navigate_command` velocity stats 禁止复用到 relative pose；
- action mask 仍为 `[16,32]` 有效位；
- stats 只使用 train episodes，validation/test 不得参与；
- 保存 stats revision、数据 manifest、target-source 比例和可用 anchor 数。

### B3. 模型改动最小化

- 从冻结 task-tuned N1.5 初始化；
- LLM、视觉塔冻结；训练 projector 和 diffusion/DiT action head；
- 输出形状仍为 `16×32`，不新增 state-flow latent；
- 训练预算固定 2 epochs、47,468 optimizer steps、batch 4、LoRA rank/alpha 32；
- 至少先做 1 batch overfit、1 episode overfit、seed 0 闭环 smoke；
- loss 下降不等于通过，必须解归一化后检查真实单位和路线单调性。

## 7. Step C：执行 adapter

模型的最后三维现在是 pose，绝不能让 Arena 原 adapter 直接当 `vx/vy/yaw_rate` 执行。正确顺序：

```text
GR00T 输出 [16,32] M1 plan
  → 用结果生效时的实测 root / 上身 / base height 建 WholeBodyPlan
  → 每个 50 Hz control tick 用 monotonic wall-clock 查询 plan
  → SE(2) position error + 路线导数得到 base twist
  → 组成原 Arena 语义的 32-D command
  → 再交给官方 32→50 adapter 和 Homie v2/WBC
```

复用 [`Phase1ExecutionAdapter`](../src/whole_body_policy/phase1.py)。新计划返回时必须从实测状态 re-anchor；计划超过 320 ms 时保持最后位置并把 feed-forward 置零，同时立即请求新计划。严禁保持最后一个非零速度。

Arena 当前 32/50 维接口没有上身速度 feed-forward 槽，所以：

- base 使用 position feedback + derivative feed-forward；
- 上身仍下发 absolute joint position；
- 上身导数只进入 telemetry；
- 不得在报告里声称 Arena 已执行 arm velocity feed-forward。

每步至少记录：

```text
observation_timestamp
inference_start/end
plan_activation_timestamp
control_timestamp
plan_anchor_root_se2
plan_age_s
query_interval_index
position/yaw_tracking_error
reference and executed base command
upper reference and measured position
replan position/velocity jump
clamped_to_horizon
```

## 8. Step D：训练与评测顺序

必须按 gate 执行：

1. 本地 unit tests；
2. 100% 数据 schema/timestamp/provenance 检查；
3. proxy-vs-odometry gate；
4. 1 batch overfit；
5. 1 episode overfit，并离线画 16 个状态点；
6. seed 0 闭环，确认最后三维没有被当 pose 直接执行；
7. seeds `0..9` 的 `3.125/6.25/12.5 Hz` 配对闭环；
8. 固定 `10/15/20/30 Hz` 和随机 `10–30 Hz`；
9. M1 冻结后才扩展到 3 个训练 seed/更多对象。

公平参考优先使用同预算 matched LoRA；同时报告官方 B0。M1 使用与 B2-WB/B3/M2 相同 tracker、限幅和 control rate。

M1 通过必须同时满足：

- 默认频率成功率相对 matched LoRA 下降不超过 10 pp；
- 6.25 和 12.5 Hz 的 root XY RMS 都 `<0.10 m`；
- yaw RMS 都 `<0.15 rad`；
- 两档成功率相对本方法默认频率的绝对变化都 `<20 pp`；
- 无 material safety regression；
- artifact、日志、逐 seed telemetry 完整。

若 M1 通过，最终 decision 必须是：

```text
stop_and_report_action_representation_as_sufficient
```

不得继续 Phase 2 来“增加论文复杂度”。若 M1 失败，先分类为 data target、normalization、tracker、默认能力或跨频率失败，再决定是否允许 Phase 2。

## 9. 必须生成的 artifact

```text
artifacts/phase_1_m1/
├── replay/episode-*.npz
├── data-gate/episode-*.json
├── data-gate/report.json
├── targets/episode-*.npz
├── targets/episode-*.json
└── dataset-manifest.json

checkpoints/arena_g1/phase_1_m1/
├── config.json
├── target-stats.json
├── training-manifest.json
└── merged checkpoint files

artifacts/validation/arena-g1-phase-1-m1/
├── fixed-*/seed-*/rollout.json
├── fixed-*/seed-*/rollout.npz
├── jitter-10-30hz/seed-*/...
├── report.json
└── gate-evaluation.json
```

`gate-evaluation.json` 至少包含：上游 commits、dataset/checkpoint revision、target source 数量、训练 seed/budget、各频率逐 seed 完整性、success/CI、轨迹指标、默认能力门槛、最终 decision。

## 10. 交付前自检

- `python3 -m unittest discover -s tests -v` 全过；
- 代码中没有用普通角度相减代替 SE(2) `Log/Exp`；
- 没有把 nominal Hz 代替真实 wall-clock telemetry；
- 每个 plan 的 16 个 base pose 都相对同一 anchor；
- 推理后 last 3 dims 已由 tracker 转回速度，未直接下发；
- horizon 超时是 hold position + zero feed-forward；
- target stats 来自 train split 的新表示；
- proxy artifact 仍标为 proxy；
- 没有修改冻结 seed、1,200-step budget 或 Gate N 门槛；
- 没有启动 Phase 2/3 范围的工作。
