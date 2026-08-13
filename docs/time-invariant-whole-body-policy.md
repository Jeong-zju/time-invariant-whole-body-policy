# 时间重参数化不变的全身移动 Policy

## 1. 技术路线（人话版）

当前全身移动 Policy 的一个根本问题是：机械臂输出的是“要到哪里”，底盘输出的是“这一秒要跑多快”。前者描述位置，后者描述速度，因此底盘动作会天然绑定数据采集频率和 Policy 运行频率。直接把底盘速度从整段轨迹起点开始积分，又会引入全局原点，使模型可能记住训练轨迹出现在哪里。

我们的做法是把底盘和机械臂统一表示成一条“全身几何路径”：底盘用相对当前时刻的局部位姿表示，机械臂仍用关节位置表示。轨迹不再按照第几秒采样，而是按照“整条动作已经走了百分之多少”采样。这样，相同动作无论采集得快、慢、密、疏，送给模型的几何路径都应基本一致。

Policy 只负责预测“怎么走”，执行层单独决定“多快走”。部署时，低层控制器可以按照真实控制频率在几何路径上插值，并通过一个进度速度控制快放、慢放。当前底盘速度、关节速度、加速度和静止时长仍作为动态信息单独输入，避免把接触、停顿等真正与时间有关的信息丢掉。

简化后的数据流如下：

```text
带时间戳的底盘速度 + 机械臂关节轨迹
              ↓
积分并转换到当前机器人坐标系
              ↓
形成 SE(2) 底盘位姿 + 关节位置的全身路径
              ↓
按全身运动进度重采样，而不是按时间重采样
              ↓
Policy 预测几何路径 + 可选的名义执行时长
              ↓
在线时间标定器按照真实控制频率和安全约束执行
```

需要明确：这里追求的是**时间重参数化不变性**，不是让机器人动力学对速度不敏感。轨迹倍速后，速度、加速度和接触力仍会变化，所以实际倍速必须由时间标定器根据速度、加速度、碰撞和接触约束裁剪。

## 2. 需要被实验验证的核心假设

Benchmark 不应只比较最终成功率，而应逐项验证以下假设：

1. **空间原点不变性**：整段任务在世界坐标中平移或旋转后，Policy 预测的当前相对路径不变。
2. **时间重参数化不变性**：同一动作被匀速或非匀速地快放、慢放后，Policy 预测的几何路径不变。
3. **频率泛化**：改变数据采样频率、Policy 推理频率或底层控制频率时，性能下降较小。
4. **窗口鲁棒性**：改变输入历史长度后，Policy 不会因为固定时间窗的语义改变而崩溃。
5. **全身协调性**：倍速或换频率后，底盘和机械臂仍在正确的动作阶段相互配合。
6. **名义性能不退化**：在训练频率和 1 倍速下，新表示不能以明显牺牲任务成功率为代价换取不变性。
7. **动态任务不过度不变**：在开门、推物、倒水等速度会改变物理结果的任务中，几何头保持稳定，时间标定器能够主动限制不安全的倍速。

## 3. Phase -1：先锁定 Benchmark 和数据集

远程部署的版本、路径、校验结果与尚未完成的闭环准入项见 [Phase -1 远程部署记录](phase-minus-1-deployment.md)。

截至 2026-08-13，新 Vast.ai 实例已经从官方源完成全量部署。RTX 5090 / NVIDIA 580.126.09 上的 Isaac Sim 5.1 能正常载入场景；官方 GR00T checkpoint 也已经通过 WebSocket 完成 51-step 闭环 smoke，并生成 JSON 与视频。因此 Phase -1 的基础设施 gate 已通过，可以进入 Phase 0。第 5 项所需的逐步时间戳、Policy 延迟、底盘和关节轨迹记录器尚需补齐，并被明确列为 Phase 0 的第一个工程任务，不能在正式 baseline 实验中省略。

### 3.1 结论：主 Benchmark 选 BEHAVIOR 2026

**决策**：主实验固定使用 [2026 BEHAVIOR Challenge](https://behavior.stanford.edu/challenge/index.html)，默认机器人使用 R1Pro，训练数据使用 [`behavior-1k/2026-challenge-demos`](https://huggingface.co/datasets/behavior-1k/2026-challenge-demos) 的 LeRobot v3 数据；[`2026-challenge-rawdata`](https://huggingface.co/datasets/behavior-1k/2026-challenge-rawdata) 只作为精确仿真回放和轨迹核验数据，不作为第一版训练输入。

这是目前最贴合本问题的选择，原因不是它的任务数量最多，而是它的机器人控制接口恰好复现了我们要解决的矛盾：官方 R1Pro 配置的底盘是 `HolonomicBaseJointController + velocity`，躯干和双臂是 `JointController + position`。因此它不是一个需要人为改造出来的“相似问题”，而是原问题本身。

它还同时满足以下条件：

- 有 100 个长时程家庭任务，每个任务有 200 条全身遥操作示范，总计 20,000 条、约 1,950 小时；
- 有底盘、躯干、双臂和夹爪的逐帧低层 state/action，以及 3 路 RGB、3 路 depth 和语言标注；
- 官方提供 OmniGibson 闭环 evaluator、任务 partial success、执行时间和运动距离指标；
- 官方已经打通 [GR00T N1.7 和 π0.5 baseline](https://behavior.stanford.edu/challenge/baselines.html)，并为 `turning_on_radio` 提供可直接执行的 checkpoint；
- 正式评测不允许 Policy 读取机器人全局位姿，只允许 RGB、depth 和 proprioception。这正好阻止模型利用全局原点作弊，也与“只使用当前相对 SE(2)”的目标一致。

需要注意，BEHAVIOR 原生分数并不评价时间重参数化不变性。它提供任务、数据、物理环境和闭环成功判据；本文第 9–11 节定义的频率、time warp、路径一致性和 base-arm 同步指标需要作为额外 evaluation wrapper 叠加在它上面。

### 3.2 数据字段与本项目的对应关系

截至本文锁定的数据 revision `4f50b44796641a4d526a19d9aeadc8aa51e2f2c2`，LeRobot 数据的关键结构如下：

| 项目 | 数据内容 | 在本项目中的用途 |
|---|---|---|
| 采样 | `fps=30`、逐帧 `timestamp` | 用真实相邻时间差做积分；构造降采样和 time warp |
| `observation.state` | 61 维；`[0:3]` 是 measured `base_qvel`，其余包含双臂、夹爪、躯干的 qpos/qvel 和末端状态 | 构造当前动态输入、历史局部路径和 future executed path |
| `action` | 23 维；`[0:3]` 是底盘速度命令，`[3:23]` 是躯干、双臂和夹爪目标 | 直接复现 B0；与 measured trajectory 比较控制误差 |
| 视觉 | 720×720 头部 RGB-D，左右 480×480 腕部 RGB-D，均为 30 fps | 沿用官方 GR00T 视觉输入，不改视觉骨干 |
| 规模 | 20,000 episodes、210,916,774 frames、100 tasks | 支持单任务 smoke test、三任务小套件和后续全量实验 |

精确切片应以官方 GR00T fork 的 [`examples/b1k/r1pro.py`](https://github.com/wensi-ai/Isaac-GR00T/blob/behavior/examples/b1k/r1pro.py) 为准，不能根据维度猜测。第一版数据使用方式固定为：

1. **B0/B1** 直接学习数据中的 23 维原始 action。
2. **B2–B4** 使用未来 `observation.state` 中实测的 `base_qvel` 和关节 qpos 构造 executed whole-body path；底盘用逐帧 `timestamp` 积分，并在每个当前时刻重新置零为局部坐标。
3. 原始 action 仍保留，用于分析命令与实际执行之间的偏差，但不优先把它当作几何 ground truth。
4. 若积分误差妨碍分析，再下载对应任务的 raw HDF5，在 OmniGibson 中精确回放并导出 simulator pose。导出的全局 pose 只能用于生成监督和评价，不能作为 Policy observation。

这里的 `timestamp` 来源于固定 30 Hz 数据，足够做基准积分，但它本身没有覆盖真实系统中的异步、抖动和丢帧。因此这些条件必须由第 9 节的反事实时间扰动主动构造，不能声称原数据天然覆盖了频率泛化。

### 3.3 第一轮只下载三个任务

全量 LeRobot 数据约 3.27 TB，Phase 0 前不应全部下载。开发顺序固定为：

| 阶段 | task id / chunk | 任务 | 选择原因 | 当前下载量估计 |
|---|---|---|---|---|
| Smoke | 0 / `chunk-000` | `turning_on_radio` | 最小闭环检查；官方同时提供 GR00T 和 π0.5 checkpoint | 约 4.87 GiB |
| Mini-suite A | 57 / `chunk-057` | `sweeping_garage` | 连续工具接触、底盘与上肢协调，对速度变化敏感 | 约 5.91 GiB |
| Mini-suite B | 17 / `chunk-017` | `bringing_water` | 多房间导航、抓取、搬运、放置，覆盖 MOVE/HOLD 和阶段切换 | 约 21.17 GiB |

三个 chunk 合计约 31.95 GiB，以上是 2026-08-13 对当前 Hugging Face revision 的 data 与 6 路视频文件估计，不包含 OmniGibson assets、环境和模型 checkpoint。

`turning_on_radio` 只用于验证数据—模型—evaluator 全链路，不能单独支撑“时间不变性成立”的结论。第一张可信的对比表至少要同时包含 `sweeping_garage` 和 `bringing_water`：前者主要暴露连续全身同步与接触问题，后者主要暴露长时程阶段切换和底盘停走问题。

数据划分按 episode 做，并保证同一条 episode 的所有重采样/time-warp 版本只能出现在同一个 split。闭环评测采用：

- 训练和离线验证：每个任务的 200 条示范，固定 episode-level train/validation split；
- 主报告：官方 public instance `0–9`，每个 instance、每种条件运行一次；
- 冻结方法后的补充测试：public instance `10–19`；
- hidden instance `20–39` 不参与调参。

### 3.4 下载和版本锁定

先只下载 task 0 的低维数据，检查 schema 和轨迹积分，不下载视频：

```bash
export B1K_DATA_ROOT=/data/behavior-2026
export B1K_DATA_REV=4f50b44796641a4d526a19d9aeadc8aa51e2f2c2
export B1K_TASK_ID=0
export B1K_CHUNK=$(printf "chunk-%03d" "$B1K_TASK_ID")

huggingface-cli download behavior-1k/2026-challenge-demos \
  --repo-type dataset \
  --revision "$B1K_DATA_REV" \
  --local-dir "$B1K_DATA_ROOT" \
  --include "data/$B1K_CHUNK/**" \
  --include "meta/episodes/$B1K_CHUNK/**" \
  --include "meta/info.json" \
  --include "meta/stats.json" \
  --include "meta/tasks.jsonl" \
  --include "meta/tasks.parquet"
```

低维检查通过后，再下载三个任务的视觉数据：

```bash
for B1K_TASK_ID in 0 17 57; do
  B1K_CHUNK=$(printf "chunk-%03d" "$B1K_TASK_ID")
  huggingface-cli download behavior-1k/2026-challenge-demos \
    --repo-type dataset \
    --revision "$B1K_DATA_REV" \
    --local-dir "$B1K_DATA_ROOT" \
    --include "data/$B1K_CHUNK/**" \
    --include "meta/episodes/$B1K_CHUNK/**" \
    --include "videos/*/$B1K_CHUNK/**" \
    --include "meta/info.json" \
    --include "meta/stats.json" \
    --include "meta/tasks.jsonl" \
    --include "meta/tasks.parquet"
done
```

仿真环境固定使用 `BEHAVIOR-1K v3.9.1`，不要使用旧的 `v3.9.0`。官方 GR00T fork 和 checkpoint 也需要在首次成功运行后记录 commit/hash；不能长期依赖会移动的 branch 名称来复现实验。

### 3.5 进入 Phase 0 的准入条件

只有下面五项都完成，才开始 Phase 0：

1. task 0 的 episode 能被 LeRobot loader 读取，并断言 `fps=30`、state 61 维、action 23 维、timestamp 单调。
2. 可视化一条 episode 的 `base_qvel`、积分后的 current-relative SE(2)、双臂 qpos 和原始 action，确认单位、方向和索引没有错。
3. 官方 `turning_on_radio` GR00T checkpoint 能通过 websocket 在 OmniGibson 中完成至少一次 rollout，并生成 JSON metrics 与视频；此处只验链路，不要求它成功完成任务。
4. 固定三个 task id、数据 revision、episode split、public instance 和随机种子，并写入实验配置。
5. 在 Phase 0 的首次正式 baseline 前，为 evaluator 增加真实 action timestamp、Policy 返回时间、底盘轨迹和关节轨迹记录，以便后续计算 `E_inv`、`E_sync` 和 deadline miss。Phase -1 的 51-step smoke 已证明接口可运行，但不替代这项 telemetry。

### 3.6 为什么其他候选不做主 Benchmark

| 候选 | 优点 | 不作为主线的原因 | 定位 |
|---|---|---|---|
| [M³Bench](https://zeyuzhang.com/papers/m3bench/) | 30k 全身轨迹、119 场景，base-arm 几何协调很强 | 输入偏向 3D scene scan、目标 mask 和开环 whole-body motion generation，不是 onboard RGB-D 的闭环 VLA；与当前混合控制接口也不完全一致 | 后续离线几何泛化补充实验 |
| [ManiSkill-HAB](https://arth-shukla.github.io/mshab/) | GPU 并行快，低层 whole-body control、RL/IL 和长时程 rearrangement 完整 | 数据与官方模型链路没有 BEHAVIOR 对 GR00T/π0.5 那么直接，也没有同样明确的“底盘速度 + 上肢位置”现成接口 | 算力不足时的轻量闭环备选 |
| [HomeRobot OVMM](https://ovmm.github.io/) | 仿真和真机兼有，开放词汇移动操作成熟 | 感知、搜索和模块化规划占比高，容易掩盖 action representation 的收益 | 后期跨平台泛化验证 |

因此第一阶段不并行维护多个 simulator。先在 BEHAVIOR 2026 上把因果链跑通；只有 B3/B4 已经在三任务 mini-suite 上稳定优于 B0/B1，才增加 M³Bench 或 ManiSkill-HAB 作为外部复现。

## 4. 从哪个基础模型开始

### 4.1 默认选择：GR00T N1.7 `NEW_EMBODIMENT`

如果项目是从零开始，建议直接从 BEHAVIOR 官方 [Isaac-GR00T challenge fork](https://github.com/wensi-ai/Isaac-GR00T/tree/behavior) 的 R1Pro baseline 开始，底座为 `nvidia/GR00T-N1.7-3B`。这个 baseline 已经把 R1Pro 注册为 `NEW_EMBODIMENT`，第一阶段不需要再从 SO100 示例接入一遍机器人。

选择它不是因为这项技术依赖 GR00T，而是因为它能让第一版尽量少改模型代码：

- 动作头是连续动作的 flow-matching DiT，适合输出位姿、关节位置和时间参数；
- 自定义 embodiment 可以在 modality config 中声明新的 state/action 字段；
- action tensor 本来就是 `action_horizon × action_dim`，模型结构可以接收一组未来进度锚点；但标准 dataset sampler 仍按时间索引取 chunk，需要在数据层替换；
- N1.7 默认最大 action horizon 为 40、state/action 维度为 132，足以容纳全身局部轨迹；
- Challenge fork 已经提供 R1Pro 的 LeRobot v3 loader、16-step action 配置、训练脚本、Policy Server 和 OmniGibson websocket evaluator。

第一版**不要修改 VLM、视觉编码器或 DiT 结构**。只修改数据转换、dataset/window sampler、modality config、归一化、输出解释和机器人端路径跟踪器。这样才能确认收益来自路径表示，而不是来自同时更换网络架构。

建议第一版取：

```text
base model      = nvidia/GR00T-N1.7-3B
embodiment      = NEW_EMBODIMENT
action_horizon  = 16 个进度锚点
history anchors = 8 个历史进度锚点
training        = 先只训练 action head / 官方默认可训练部分
```

这里的“官方默认可训练部分”具体是 projector 和 diffusion action head，vision encoder 与 LLM 保持冻结。第一轮保持这个设置和官方训练预算不变，避免模型训练策略成为新的变量。

### 4.2 已有模型和算力不足时怎么选

| 情况 | 选择 | 原因 |
|---|---|---|
| 从零开始，训练卡有约 40 GB 以上显存 | GR00T N1.7 | 自定义 embodiment、动作维度和部署链路最完整 |
| 工程已经基于 π0/π0.5 | 保留现有模型 | 在 openpi 的 `Inputs`、`Outputs`、`DataConfig` 中改表示即可，没有必要迁移 |
| 只有消费级显卡，先验证想法 | [SmolVLA](https://github.com/huggingface/lerobot/blob/main/docs/source/smolvla.mdx) | 训练成本低，适合先证伪数据表示和控制接口 |
| 需要确认结论不依赖某个模型 | 主实验 GR00T，复现实验 π0.5 | 两个连续动作生成模型上的同向结果更有说服力 |

[openpi](https://github.com/Physical-Intelligence/openpi) 已把机器人相关输入输出映射和模型主体分开，并提供 π0.5 自定义 LeRobot 数据微调流程。因此，如果现有 Policy 已经使用 π0.5，执行顺序与下文完全相同，只需把“modality config”替换成 openpi 的 `Inputs/Outputs/DataConfig`。

### 4.3 在 GR00T 仓库中具体从哪里改

先按照 BEHAVIOR 的 baseline walkthrough 跑通 `turning_on_radio`，再从 challenge fork 的这些现成文件分支修改：

```text
examples/b1k/r1pro.py               # state/action modality 和 action 表示
examples/b1k/r1pro.json             # 数据集 meta/modality.json 模板
scripts/b1k/deploy_modality.py       # 部署并检查 61/23 维 schema
scripts/b1k/train_b1k.py             # GR00T N1.7 微调入口
scripts/b1k/serve_b1k.py             # websocket Policy Server
gr00t/eval/eval_b1k_wrapper.py       # 仿真 observation/action adapter
```

不要覆盖这些官方文件。新建 `r1pro_progress.py`、`r1pro_progress.json` 和独立的数据变换/采样模块，使 B0 与 B2–B4 可以在同一 commit 中并存并复现。

建议在 `modality.json` 和 config 中拆成这些语义字段：

```text
state.current_proprio    # arm_q, arm_qdot, base_twist, base_acc, dwell
state.history_path       # 当前坐标系下的历史全身进度路径
state.history_mask

action.base_path         # [rel_x, rel_y, sin_yaw, cos_yaw]
action.arm_path          # 关节位置
action.gripper
action.timing            # Phase 5 再加入
```

`action.base_path` 虽然描述的是“相对当前帧的位姿”，但写入 GR00T 时应配置成 `ActionRepresentation.ABSOLUTE`：这里的网络目标就是局部坐标系中的完整数值，不能再让框架减一次 current state。`arm_path`、`gripper` 和 timing 字段也先使用 `ABSOLUTE + NON_EEF`。

最关键的代码改动是增加一个 `ProgressChunkSampler`：

```text
输入：当前原始样本索引 t
  1. 读取 t 时刻的图像、语言和 current state
  2. 读取或计算 future_path[t]，形状为 [N, D]
  3. 读取 history_path[t]、history_mask[t]
  4. 直接把 future_path[t] 作为模型 action chunk 返回

输出：
  observation state = current dynamics + progress history
  action             = progress anchors [N, D]
```

不要直接沿用下面这种标准配置并假设它已经实现进度采样：

```python
delta_indices=list(range(16))
```

在官方数据管线中，它默认表示未来 16 个**时间索引**。有两种正确实现：

1. **推荐**：保留原始带时间戳数据，为每个观测预计算 `future_path[t, N, D]`，由自定义 sampler/collator 返回；
2. **简单原型**：把整个 episode 离线转成等进度采样的“虚拟 episode”，再使用标准 `delta_indices`。这种办法会折叠停顿，不适合作为最终 B4。

所以第一版需要改的是数据取窗逻辑，而不是 flow-matching DiT。模型仍接收标准 `[batch, action_horizon, action_dim]` tensor。

建议先分别检查三个中间产物：

- 一个样本的 current-relative path 可视化是否正确；
- sampler 返回的第 `j` 行是否对应 `u_j`，而不是原始 `t+j`；
- Policy output adapter 是否把网络输出反归一化为同一套局部路径定义。

## 5. 分阶段执行路线

### Phase 0：冻结接口和实验条件

开始训练前，先固定以下内容：

- Benchmark 为 BEHAVIOR-1K `v3.9.1`、机器人为 R1Pro，LeRobot 数据 revision 为 Phase -1 中锁定的 commit；
- 机器人关节顺序、单位和上下限；
- 底盘 twist 是 body frame 还是 world frame；
- 每个传感器、状态和命令的真实时间戳；
- `f_data`、`f_policy`、`f_ctrl` 三种频率；
- 底盘速度、加速度，机械臂速度、加速度、jerk 和接触安全限制；
- 三任务 mini-suite、episode split、public instance、随机种子和 evaluator 配置。

数据必须同时保存“命令值”和“实际测量值”：底盘命令速度、里程计/定位器测得的位姿和速度、机械臂目标关节、实际关节位置，以及图像和语言。训练几何目标优先使用同步后的实际执行轨迹，因为它与图像中真正发生的运动一致；命令值用于控制器分析和故障诊断。

**验收条件**：任意两个模态可以按时间戳重放；不能只依赖“第几帧”对齐。

### Phase 1：先复现原始 velocity baseline

先在选定基础模型上复现当前方案：输入图像、语言、关节状态和底盘速度，输出固定频率的 velocity + joint-position action chunk，也就是 B0。

这一步不追求创新，只为了确认：

- 数据到模型再到机器人的完整链路能跑通；
- action normalization、关节顺序和单位正确；
- Policy Server 延迟和控制器行为已被记录；
- 能得到后续所有方法共用的名义成功率。

**验收条件**：B0 在训练频率下达到当前系统可接受的成功率；否则不能直接把后续失败归因于新表示。

随后立即训练 B1：仍输出 velocity chunk，但把每个样本的真实 `Δt` 或目标执行时长作为条件，并加入时间缩放增强。B1 是必须保留的强基线；如果它已经解决绝大多数频率问题，就不应把完整几何方案的收益夸大。

### Phase 2：实现确定性的轨迹转换器

这一阶段不训练模型。对每个样本时刻 `t`，执行以下转换：

1. 用真实 `Δt` 对底盘 measured twist 做 SE(2) 积分，或直接使用融合后的 odometry pose。
2. 把未来所有底盘位姿转换到当前机体坐标系：`T_rel(i) = T(t)^-1 T(i)`。
3. 将底盘相对位姿与机械臂关节位置组合成全身构型 `Q(i)`。
4. 累计全身运动距离 `σ`，未来范围按固定几何长度 `L_σ` 截取，不能按固定秒数截取。
5. 在归一化进度 `u∈[0,1]` 上重采样为固定的 `N=16` 个锚点。
6. 历史状态同样按固定历史运动距离重采样为 `M=8` 个锚点；历史不足时使用 mask，不能伪造运动。
7. 保存原始持续时间和各段 `Δt`，但不要让它们改变几何锚点。

全身进度可先用：

\[
\Delta\sigma_i^2=
\Delta x_i^2+\Delta y_i^2+\ell^2\Delta\theta_i^2+
\lambda_q\|W_q\Delta q_i\|^2.
\]

建议第一版的每个几何 action 锚点为：

```text
[rel_x, rel_y, sin(rel_yaw), cos(rel_yaw),
 arm_q_1, ..., arm_q_n, gripper]
```

这里底盘位置相对当前帧，机械臂关节位置保持绝对编码器位置。`sin/cos` 用于避免角度在 `-π/π` 处跳变。

历史 state 可以先展平为：

```text
current:
  arm_q, arm_qdot, base_twist, base_acc, dwell_time

history_path[8]:
  rel_x, rel_y, sin(rel_yaw), cos(rel_yaw), arm_q...

history_mask[8]
```

如果展平后超过基础模型的 state dimension，再减少历史锚点或增加单独的 history encoder；第一版不要一开始就改网络。

训练时可以把这些路径预计算到数据集中以提高吞吐，但部署时必须在 `policy_adapter` 前放置同一份无参数在线前端：它维护一个带时间戳的环形缓冲区，把最近的 measured base twist 积分成当前相对历史路径，再按进度重采样。这样“速度变成位置性质”发生在 Policy 输入管线内，同时不需要向网络提供全局 odometry 原点。训练与部署必须共用同一套转换代码，避免 preprocessing skew。

未来路径只作为监督标签，不能进入 observation；历史转换器在时刻 `t` 只能读取 `≤t` 的信号，单元测试中需要专门检查没有 future leakage。

转换器必须有以下单元测试：

- 对整段轨迹施加任意全局 SE(2) 变换，输出完全不变；
- 将同一轨迹重采样到 0.5×、1×、2×频率，输出几何锚点近似不变；
- 将时间戳统一缩放但保持轨迹不变，几何锚点不变、持续时间按比例变化；
- 纯平移、纯旋转、底盘不动手臂运动均能产生有效进度；
- 全身完全静止时不会除零，而是产生 HOLD 样本。

**验收条件**：先通过上述确定性测试，再生成训练数据。模型不能补救错误的数据几何。

### Phase 3：先把低层执行器做对

在训练新 Policy 前，用 ground-truth 几何路径驱动仿真机器人，建立“oracle path replay”：

1. 对底盘相对位姿锚点做 SE(2) 插值；
2. 对关节锚点做样条或限速插值；
3. 使用同一个进度 `u(t)` 同时查询底盘和机械臂目标；
4. 底盘跟踪器把相对位姿路径变成实时速度命令；
5. 机械臂控制器输出当前位置控制目标；
6. 时间标定器按照真实 `Δt_real` 推进 `u`，并根据速度、加速度和 jerk 限制自动缩小 `du/dt`。

Policy 每次重规划后，新路径以本次推理时刻为原点。收到新路径时，应把当前实测构型投影到新路径上，从最接近的未执行进度继续，不能机械地从 `u=0` 重播，否则 Policy 延迟会造成停顿和回跳。

**验收条件**：ground-truth 路径在 0.5×、1×、1.5×速度和不同 `f_ctrl` 下都能保持几何与 base-arm 同步；如果 oracle 都跟踪失败，先修控制器，不进入模型训练。

### Phase 4：只改数据表示，训练几何 Policy

在 GR00T 中创建 `NEW_EMBODIMENT` modality config：

- observation state 使用 Phase 2 的 current dynamics + progress history；
- action horizon 设为 16；
- action 字段使用 16 个几何锚点；
- 重新计算新 state/action 的 normalization statistics；
- 模型结构、视觉输入、语言输入和训练预算与 B0 相同。

这里 action horizon 的第 `j` 行不再表示“未来第 `j` 个控制周期”，而表示固定的进度位置 `u_j=j/(N-1)`。模型无需知道原始数据频率。

先训练两个版本：

- **B2**：当前相对 SE(2)，但仍按时间采样；
- **B3**：当前相对 SE(2)，并按全身进度采样。

先做离线 `E_inv` 和几何误差，再做闭环。若 B3 在频率扰动下没有明显优于 B2，应暂停增加 timing head，优先检查轨迹转换和历史窗口。

**验收条件**：B3 的名义性能没有实质退化，并且时间/频率扰动下的 `E_inv` 明显低于 B0/B2。

### Phase 5：加入时间、停顿和动作模式

几何表示通过后，再处理真正依赖时间的部分。第一版不必建立新的神经网络 head，可在每个进度锚点追加：

```text
[log_delta_t, motion_gate]
```

- `log_delta_t` 表示从上一个进度锚点到当前锚点的名义时间；
- `motion_gate` 表示 MOVE/HOLD，终点另用 STOP 或 episode mask 表示；
- 部署时先恢复名义 `u(t)`，再乘用户期望速度倍率 `γ`，最后由安全约束裁剪；
- 停顿时间记入下一个运动段的 `Δt`，全身长期静止则输出 HOLD。

这样在语义上已经把几何和时间分开，但仍复用同一个 DiT action head。只有在消融发现 timing channel 明显干扰几何预测时，才增加独立 timing MLP/head，并为几何损失和时间损失分别设置权重。

**验收条件**：B4 相比 B3 能处理停顿和接触敏感任务，同时不能增大几何不变性误差。

### Phase 6：增加时间扰动训练，而不是只靠测试

对训练轨迹在线构造：

- 0.5–2× 数据重采样；
- 匀速和分段非匀速 time warp；
- 时间戳抖动、丢帧和 Policy 延迟；
- 0.5×、1×、2× 历史范围；
- 全局 SE(2) gauge augmentation。

同一条轨迹的不同时间参数化应共享几何 target。可以进一步加入一致性损失，使两种输入的几何预测接近，但一致性损失应在普通监督训练稳定后再加入。

**验收条件**：增强不能只改善某个指定倍率；在未参与训练的非均匀 time warp 上也应降低 `E_inv`。

### Phase 7：仿真到真机

执行顺序固定为：

1. 离线表征一致性；
2. oracle path 仿真回放；
3. Policy 仿真闭环；
4. 真机 0.5× 低速；
5. 真机 1× 名义速度；
6. 在安全约束允许时测试 1.5×；
7. 最后测试推理延迟、丢帧、里程计漂移和组合扰动。

每次只放开一个新变量。任何情况下，急停、速度限制和碰撞过滤器都不能由 Policy 绕过。

## 6. 推荐的代码模块边界

无论选 GR00T、π0.5 还是 SmolVLA，都建议把项目拆成以下六个独立模块：

```text
trajectory_builder
  raw timestamped signals -> current-relative whole-body trajectory

progress_resampler
  whole-body trajectory -> fixed progress anchors + masks + timing

policy_adapter
  robot observation/action <-> foundation-model tensors

path_retimer
  geometry + nominal timing + speed scale -> safe u(t)

whole_body_tracker
  Q(u) -> base velocity command + arm position command

temporal_benchmark
  time warp / frequency perturbation / E_inv / E_sync / G_time
```

转换器和控制器必须独立于 VLA 仓库。这样更换 π0.5、GR00T 或 SmolVLA 时，只需替换 `policy_adapter`，也能保证不同基础模型使用完全相同的路径定义和评价协议。

## 7. 实际开发顺序和交付物

| 顺序 | 交付物 | 通过后才能进入 |
|---|---|---|
| 0 | Phase -1：task 0 数据检查、官方 checkpoint rollout、三任务配置冻结 | 确认 Benchmark 可执行 |
| 1 | B0 基础模型 baseline 和日志 | 确认原系统可比较 |
| 2 | B1 velocity + `Δt` + time-warp baseline | 排除简单条件化已经足够的可能 |
| 3 | 轨迹转换器、可视化和单元测试 | 确认空间/时间数学正确 |
| 4 | ground-truth path replay 控制器 | 确认新接口可执行 |
| 5 | B2 固定时间相对 waypoint | 隔离空间原点收益 |
| 6 | B3 进度 waypoint | 验证时间重参数化不变性 |
| 7 | B4 timing/HOLD | 恢复停顿和动态任务能力 |
| 8 | 时间增强和一致性训练 | 提高组合扰动鲁棒性 |
| 9 | 完整仿真 Benchmark | 决定是否值得真机 |
| 10 | 分阶段真机测试 | 最终部署结论 |

最容易走错的顺序是直接修改模型 action head，然后上真机看成功率。正确顺序应先证明数据转换是 invariant 的，再证明 ground-truth 路径可被任意频率执行，最后才让 Policy 学习这套表示。

## 8. Benchmark 结构

建议采用三层 Benchmark。三层解决的问题不同，不能相互替代。

### 8.1 Layer A：离线表征一致性

这一层不运行机器人，只使用保留了原始时间戳的测试轨迹。对每条轨迹构造反事实版本：

- 随机施加全局 SE(2) 平移和旋转；
- 重采样为训练频率的 0.5、1、2 倍；
- 以 0.5、0.75、1、1.5、2 倍速做统一时间缩放；
- 施加分段变速、短暂停顿等非均匀时间扭曲；
- 改变历史窗口长度，并加入时间戳抖动和少量丢帧。

原轨迹和变换后的轨迹在按运动进度对齐后，几何输出应保持一致。这一层最能直接回答“表示是否真的消除了原点、频率和速度依赖”，不应被视觉识别或控制器误差干扰。

### 8.2 Layer B：仿真闭环执行

至少包含五类任务：

| 任务类型 | 示例 | 主要验证内容 |
|---|---|---|
| 底盘几何任务 | 直线、S 弯、8 字、原地旋转 | SE(2) 积分、纯旋转和路径保持 |
| 同步全身任务 | 移动中伸手、端持物体移动 | 底盘—机械臂共享进度与协调 |
| 分阶段任务 | 接近物体、底盘停住、抓取、撤离 | 底盘静止但手臂继续运动的情况 |
| 停顿任务 | 等待障碍物或门打开后继续 | 零运动进度和 dwell token |
| 动态/接触任务 | 开门、拉抽屉、推物、搬运易洒物 | 时间标定、安全约束和物理可行性 |

主任务底座固定为 Phase -1 选定的 BEHAVIOR 2026 / R1Pro。表中的底盘几何和同步全身任务先作为同一 OmniGibson/R1Pro 控制栈内的 diagnostic scenes，用来隔离积分器和 tracker；正式任务结果则报告三任务 mini-suite。不要在第一轮混入其他 simulator，因为平台差异会妨碍归因。

M³Bench 只在主结论成立后补充离线 whole-body motion 泛化；ManiSkill-HAB 只在需要更高吞吐的低层闭环复现时加入。它们都不替代 BEHAVIOR 的主结果。

### 8.3 Layer C：真机部署

从仿真中选择三个代表任务：纯底盘路径、同步全身搬运、接触敏感任务。真机重点测试：

- 训练频率与部署频率不一致；
- Policy 推理延迟和抖动；
- 底层控制频率变化；
- 0.5、1、1.5 倍速执行；
- 里程计漂移、轻微打滑和丢帧。

真机不需要覆盖所有组合，但必须包含仿真中最容易拉开方法差距的条件。

## 9. 测试矩阵：必须区分三种频率

“频率变化”不能只用一个变量表示。实际系统至少有三只时钟：

| 变量 | 含义 | 推荐测试值 |
|---|---|---|
| 数据/传感器频率 `f_data` | 观测和示范的采样频率 | 0.5×、1×、2×训练频率 |
| Policy 频率 `f_policy` | VLA 多久重新预测一次路径 | 0.5×、1×、2×名义频率，并加入延迟抖动 |
| 控制频率 `f_ctrl` | 低层控制器跟踪路径的频率 | 0.5×、1×、2×名义频率 |
| 速度倍率 `γ` | 沿同一几何路径前进的速度 | 0.5、0.75、1、1.5、2 |
| 历史范围 | Policy 看多少历史 | 0.5×、1×、2×名义范围 |

不必穷举所有笛卡尔积。先逐个变量测试，再增加少量困难组合，例如低 Policy 频率、抖动传感器、高控制频率和 1.5 倍速同时出现。

速度倍率只在动力学可行范围内计入“时间不变性”主结果。超出速度、加速度或接触约束的倍率应作为安全压力测试：正确行为是自动降速，而不是为了满足目标时长强行执行。

BEHAVIOR 官方 `1.5× mean human length` timeout 只原样用于 `γ=1` 的官方可比结果。时间不变性套件不能让 0.5× 方法仅仅因为墙钟/仿真时长加倍而失败，建议使用

\[
T_{max}(\gamma)=\max(T_{max}(1), T_{max}(1)/\gamma).
\]

这样慢放获得相应的完成时间，快放也不会因为更短 timeout 被额外惩罚；是否真正达到目标倍率由 `E_pace` 单独评价。所有频率都按 simulator time 调度，不能用不同机器上的 wall-clock FPS 当作 `f_policy` 或 `f_ctrl`。

## 10. 对照组和消融

| 编号 | 方法 | 要回答的问题 |
|---|---|---|
| B0 | 原始固定频率 velocity action chunk | 当前系统基线 |
| B1 | velocity chunk + 真实 `Δt` 条件 + 时间缩放增强 | 简单地告诉模型时间间隔是否已经足够 |
| B2 | 当前相对 SE(2) waypoint，但仍按固定时间采样 | 只消除空间原点是否足够 |
| B3 | 按运动进度采样的几何 waypoint，不单独建模时间 | 几何表示本身带来多少收益 |
| B4 | 几何路径 + 独立进度速度/执行时长 + 在线时间标定 | 完整方案 |

完整方案至少做以下消融：

- 全身联合进度 vs. 只用底盘路程定义进度；
- 固定时间历史 vs. 固定运动进度历史 vs. 多尺度历史；
- 有无当前速度、加速度和静止时长输入；
- 离散 waypoint vs. 连续样条控制点；
- 有无在线速度、加速度和 jerk 约束；
- 当前相对坐标 vs. episode 起点坐标或全局坐标。

所有方法应使用相同数据、视觉骨干、参数规模和训练步数，并共享相同的机器人动力学、执行器限制与安全过滤器。不同动作接口必需的最小转换层应公开并单独报告；例如 waypoint 方法需要路径跟踪器，而 velocity 方法不应被人为削弱成一个不合理的控制器。否则无法把改进归因于动作表示。

## 11. 评价指标

### 11.1 统一的全身路径距离

先定义相同进度位置上两个全身构型之间的距离：

\[
d_Q^2(Q_1,Q_2)=
\|\Delta p\|^2+
\ell^2\operatorname{wrap}(\Delta\theta)^2+
\lambda_q\|W_q\Delta q\|^2.
\]

其中平移、旋转和关节误差应同时单独报告，组合距离只用于比较整体趋势。权重需要按机器人尺寸、关节范围或控制容差在实验前固定。

### 11.2 表征层指标

1. **几何不变性误差 `E_inv`**：同一轨迹的原始输入和变换输入，分别预测出路径后，在统一进度上计算平均 `d_Q`。分别报告全局 SE(2) 变换、采样频率变化、匀速缩放和非均匀时间扭曲的结果。越低越好。

   \[
   E_{inv}(\phi)=\frac{1}{N}\sum_{j=1}^{N}
   d_Q\left(\hat Q(u_j),\,\mathcal C_\phi[\hat Q_\phi(u_j)]\right),
   \]

   `C_φ` 表示把变换后的预测重新放回统一的当前相对坐标和进度域。

2. **几何模仿误差**：对 ground truth 进度路径报告底盘平移 RMSE、航向 MAE、关节归一化 RMSE、末端位置/姿态误差。

3. **路径 Fréchet distance**：评价两条曲线的几何接近程度，不要求它们按相同时间速度经过各点。它比逐时间点 MSE 更适合本项目。

4. **终点误差**：底盘终点位置/航向误差、关节终点误差和末端执行器终点误差。该指标不能代替整条路径指标。

### 11.3 执行层指标

1. **路径保持误差**：不同 `γ` 和不同频率执行后，与 1 倍速执行轨迹在进度域中的平均/P95 距离。
2. **横向跟踪误差**：底盘到命令路径的 mean/P95 cross-track error。
3. **全身同步误差**：分别把实测底盘和机械臂投影到命令路径，计算二者进度差：

   \[
   E_{sync}=\frac{1}{T}\int_0^T |u_{base}(t)-u_{arm}(t)|\,dt.
   \]

   离散实现必须用真实 `Δt` 加权，不能直接对控制帧求平均，否则该指标本身也会随控制频率改变。

4. **速度倍率跟随误差**：若 `γ` 表示期望速度倍率，则期望时长为 `T_1/γ`：

   \[
   E_{pace}=\left|\frac{T_\gamma}{T_1/\gamma}-1\right|.
   \]

5. **平滑性与约束**：速度、加速度、jerk 的 mean/P95/max，命令总变差，以及超过硬件限制的时间占比。
6. **安全指标**：碰撞次数和冲量、最小障碍距离、物体掉落率、接触力超限率、急停或人工接管次数。
7. **实时性指标**：Policy 推理延迟、控制 deadline miss 比例、每秒 Policy 调用次数及算力占用。

### 11.4 任务层指标

1. **任务成功率 `SR`**：必须按任务、频率和倍率分别报告，而不是只给总平均。
2. **时间泛化退化量**：

   \[
   G_{time}(c)=SR(\text{train frequency},\gamma=1)-SR(c).
   \]

   越低越好；负值表示 OOD 条件反而优于名义条件。

3. **OOD 平均成功率和最差条件成功率**：平均值衡量总体鲁棒性，最差条件防止平均数掩盖某个频率下的崩溃。
4. **完成时间**：仅在成功回合中统计，并与 `E_pace` 一起解释。
5. **有效速度范围**：在满足任务成功率下降上限和全部安全约束的前提下，可稳定执行的 `[γ_min, γ_max]`。

建议把以下五项作为主表指标：

| 主指标 | 回答的问题 |
|---|---|
| `E_inv` | 表征是否真的对原点、频率和时间缩放不敏感 |
| 进度对齐后的路径误差 / Fréchet distance | 快放慢放后是否还是同一条路 |
| OOD 平均 SR、最差 SR、`G_time` | 换频率和速度后任务是否还能完成 |
| `E_sync` | 底盘和上肢是否仍协调 |
| 安全约束违反率 | 性能是否通过不安全动作换来 |

不建议把这些指标强行压成一个总分。一个方法可能路径非常稳定，但速度倍率跟随较差；另一个方法可能任务成功，却靠碰撞或控制器饱和完成。分项结果对技术判断更有价值。

## 12. 最小可执行实验

第一轮不需要马上跑完整公开 Benchmark，可以先完成一个能证伪方案的最小实验：

1. 先在 `turning_on_radio` 上复现官方 GR00T B0，再使用相同的 `turning_on_radio + bringing_water + sweeping_garage` 三任务数据训练 B0–B4 五种表示。
2. 每条离线测试轨迹生成全局坐标变换、0.5/2 倍频率、0.5/1.5 倍速和非均匀变速版本。
3. 计算 `E_inv`、全身路径误差和终点误差。
4. 先用 8 字底盘、移动中伸手和接近—抓取—撤离三个 diagnostic scenes 验证 tracker，再在三个正式 BEHAVIOR 任务的 public instance `0–9` 上闭环执行。
5. 在训练频率以及 0.5/2 倍 Policy 频率下，分别执行 0.5、1、1.5 倍速；正式主表至少报告 1× 名义条件和预先冻结的 OOD 条件子集，不能事后挑选最有利组合。
6. 报告名义 SR、OOD 平均/最差 SR、`G_time`、`E_sync`、路径误差、`E_pace` 和安全指标。

实验应对各方法复用完全相同的任务初始状态和随机种子。仿真连续指标使用配对 bootstrap 置信区间；成功率使用二项分布置信区间。训练建议至少使用多个随机种子，真机则采用随机区组顺序，避免电量、温度和地面状态随实验时间系统性偏向某个方法。

## 13. 判定标准

完整方案只有同时满足以下条件，才能说明技术路径成立：

- 相比原始 velocity chunk，时间/频率变换下的 `E_inv` 显著降低；
- 名义条件下任务成功率没有实质性下降；
- OOD 条件的成功率下降和最差条件性能明显优于固定时间表示；
- 改变倍率后，进度对齐的几何路径和 base-arm 同步关系基本保持；
- 在动力学不可行的倍率下，系统会安全降速，而不是增加碰撞、抖动或控制饱和。

如果 B3 已经获得几乎全部收益，而 B4 只增加复杂度，则独立时间头可以推迟；如果 B3 在接触任务或停顿任务上明显失败，而 B4 能正确调整速度，那么“几何—时间解耦”才被完整验证。如果 B1 已接近 B4，则说明主要问题可以由 `Δt` 条件和数据增强解决，没有必要为新表示承担额外系统复杂度。
