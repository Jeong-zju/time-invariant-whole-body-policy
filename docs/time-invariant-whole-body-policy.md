# 时间分割一致的全身 Policy：技术路线

更新日期：2026-08-16

## 0. 这项研究到底想做什么

当前 Arena G1 Policy 的 action 是混合的：双臂和双手是绝对关节位置，底盘是 `vx/vy/yaw_rate`。目标部署场景可以按 `30 Hz` 采集示教，但受算力和调度影响，VLA 的实际推理/重规划频率可能在 `10–30 Hz` 之间随机变化，底层控制器则仍以更高的固定频率运行。现有 Arena 代理数据实际是理想化 `50 Hz`，Gate N 测的是 `3.125/6.25/12.5 Hz` 重规划；二者必须分开记录，不能把目标场景参数倒写成已有数据事实。

这里必须区分两件事：

- 如果 `vx` 的确是以 m/s 表示的物理速度，那么同一速度执行更久、产生更大位移本身是正确物理规律；
- 真正的问题是，固定频率数据学出的 action slot 往往同时携带“这一帧该做什么”和“这一帧默认持续多久”的隐含语义。部署时改变两个推理 step 之间的真实时间，再继续保持单步速度或消费旧 chunk 中的速度计划，就会改变底盘的积分位移，而上肢位置目标不会以同样方式积累，最终造成底盘漂移和 base–arm phase mismatch。

位置控制也不是对频率绝对免疫：不同重规划频率仍会改变反馈和新观测。但同一个绝对位置目标多保持一会儿，只会继续收敛到该目标，不会像非零速度那样随保持时间无界累积位移。这正是重规划后的位置参考通常更稳定的原因。

用人话画成一条因果链：

```text
固定频率示教
  → action slot 偷偷绑定默认时长
  → 部署推理间隔随机变化
  → 底盘速度按不同 Δt 积分，上肢位置却继续追同一目标
  → 底盘与手臂错相、漂移、成功率下降

新方案：Policy 直接回答“真实 Δt 后全身应该到哪里”
  → 一个长步必须等价于多个短步
  → tracker 跟位置参考，并使用同一状态流的速度前馈
```

因此本文不再把问题描述为“快慢演示是否产生不一致”，也不把时间原点等变性当作频率鲁棒性的充分条件。核心问题改为：

> **同一个 Policy 面对 `10–30 Hz` 的随机推理间隔时，能否让同一段物理时间无论被分成一个长 step 还是多个短 step，都产生一致的全身状态转移？**

若用 `X=(Q,h)` 表示物理全身状态 `Q` 和模型内部 action latent `h`，用 `\Phi_\theta^{\Delta t}` 表示经过真实时间 `\Delta t` 的状态转移，那么需要直接逼近：

\[
\Phi_\theta^{\Delta t_1+\Delta t_2}(X,z)
\approx
\Phi_\theta^{\Delta t_2}\!\left(
  \Phi_\theta^{\Delta t_1}(X,z),z
\right).
\]

这叫**时间分割一致性**，也就是状态流的半群性质。它直接对应频率问题：执行一次 `100 ms` 的状态转移，应当等价于连续执行三次约 `33 ms` 的状态转移。本文暂称该方法为 **Semigroup-Consistent Whole-Body Policy（SC-WBP）**。

### 阅读时只需要记住的术语

| 文中术语 | 自然语言含义 |
|---|---|
| Policy | 根据画面、指令和机器人状态决定动作的模型，也就是机器人的“动作大脑” |
| `Q` / whole-body state | 底盘 SE(2) 位姿、躯干/双臂关节位置和夹爪状态组成的联合状态 |
| `X=(Q,h)` | 在 `Q` 之外带一个随状态流传递的内部动作状态 `h`，用于记住一段动作进行到了哪个内部阶段 |
| `delta_t` / `Δt` | 两次推理或两个状态参考之间真实经过的秒数，不是 frame index |
| state transition `Φ^{Δt}` | 从当前全身状态出发，预测经过 `Δt` 后应该到达的全身状态 |
| time-partition consistency | 同一总时长被切成一个长间隔或多个短间隔，组合后的状态转移应一致 |
| semigroup property | 时间分割一致性的数学名字；`走 0.1 s` 等价于 `先走 0.04 s，再走 0.06 s` |
| replanning frequency | VLA 多久根据新观测生成一条新的状态流；本文主要扰动的频率 |
| control frequency | 底层 tracker/WBC 更新位置和速度命令的频率，必须与推理频率分开报告 |
| re-anchor | 新推理结果到达时，从实测当前状态重新生成后续状态流，而不是从旧预测起点硬接 |
| Gate | 进入下一阶段前的检查点，不是新技术 |
| Phase | 一次只增加一个可独立验证的模型或训练改动 |

## 1. 对旧路线的关键修正

### 1.1 时间原点等变只处理时延和 action chunk 接缝

旧路线希望不同推理时刻对同一绝对未来时刻给出相同预测，可写成：

\[
A_t(\tau+\delta)\approx A_{t+\delta}(\tau).
\]

它适合检验“晚 `\delta` 秒开始的预测是否仍与旧预测重叠”，因此主要改善推理时延、异步更新和 chunk seam。它并没有约束一个长时间步是否等于多个短时间步，也就不能单独保证随机推理频率下的稳定性。

所以时间原点等变性可以保留为辅助连续性指标，但不再是论文的核心机制和主 claim。

### 1.2 单纯输入真实 `delta_t` 没有定义正确的学习对象

Gate N 的 timestamp-aware consumer 已经把真实 `delta_t` 输入 Policy，但 action target 仍是原来固定频率下的速度/位置行。模型看到的虽然多了一个数字，却没有被要求学习“经过该时间后应该到哪里”，也没有被要求满足长步与短步的组合关系。

换句话说，这个基线回答的是“告诉原 action generator 下一次推理要等多久是否有用”，而不是“模型能否学习一个按真实时间查询的状态转移算子”。它失败不能说明真实时间无用，说明的是**把时间作为普通条件而不改变目标和结构不够**。

### 1.3 SE(2) waypoint 说明位置参考有价值，但还不是完整答案

Gate N 的 SE(2) waypoint 把底盘速度后处理成相对位姿目标，确实显著减小了高频条件下的 XY 分叉；这与“位置参考不会随 hold 时长继续积分漂移”的判断一致。但它只改了底盘、没有把上肢纳入同一个可组合的状态转移，而且使用的是训练后转换和未联合优化的 tracker，默认成功率从 `8/10` 降到 `5/10`。

因此它既不是失败得毫无信息，也不能直接作为最终方案。它提示下一步应当在模型内部端到端预测**全身状态流**，同时由该状态流提供位置参考和解析速度前馈，而不是在原速度 Policy 外再套一层只处理底盘的转换器。

### 1.4 研究范围不再扩展到事件触发或通用可中断规划

本文只研究一个明确部署变量：推理/重规划间隔在 `10–30 Hz` 范围内随机变化。事件触发、何时主动调用大模型、长时程技能切换和任意中断恢复都不是当前方法的必要组成，也不进入主要 novelty。

需要同时承认边界：当高频推理让 Policy 看到了新的障碍、接触或物体运动，而低频推理没有看到时，两者做出不同动作是合理的。本文要消除的是**调度时间分割本身造成的差异**，不是强迫不同信息条件下的行为逐点相同。

## 2. 新路线的核心假设

### 假设一：实际推理频率变化会破坏当前混合 action Policy

Gate N 已经用冻结 checkpoint 和配对初始条件验证：只改变重规划频率，成功率会从默认频率的 `8/10` 降到 `5/10` 和 `4/10`。因此问题定义不再是“快慢示教是否标签冲突”，而是固定频率学到的动作在可变 sample-and-hold 间隔下是否仍表示同一物理状态转移。

### 假设二：全身位置状态流比逐 step 混合 action 更适合作为模型输出

模型不再独立输出“底盘速度 + 上肢位置”序列，而是从当前全身状态 `Q_t` 和视觉语言语义 `z_t` 出发，定义可按真实 `Δt` 查询的联合状态转移：

\[
Q_{t+\Delta t}=\Pi_Q\Phi_\theta^{\Delta t}(X_t,z_t),
\qquad Q\in SE(2)\times\mathbb R^n.
\]

这里的底盘和上肢共享同一个物理时间参数。tracker 跟踪 `Q_{t+Δt}`，同时使用状态流的导数作为底盘 twist 和关节速度前馈。这样既保留位置参考抗积分漂移的优点，也避免纯 position-only waypoint 因缺少速度信息而损失动态性能。

### 假设三：只有显式训练时间分割一致性，才可能获得可解释的频率鲁棒性

连续曲线或 Neural ODE 本身不等于频率鲁棒。核心训练约束必须直接比较“一步走完”和“拆步走完”的结果：

\[
\mathcal L_{partition}
=d_{\mathcal X}\!\left(
\Phi_\theta^{\delta_1+\delta_2}(X_t,z_t),
\Phi_\theta^{\delta_2}
\left(\Phi_\theta^{\delta_1}(X_t,z_t),z_t\right)
\right).
\]

比较两边时固定同一 observation/语义 latent `z_t`，专门隔离时间分割的影响。若这个约束成立，则对总时长相同的任意分割，理论上都应得到同一终态；这提供了“为什么模型应对频率变化稳定”的直接、可检验机制。

### Benchmark 已切换到原生支持 GR00T 的简单移动操作

2026-08-14 起，主开发 benchmark 正式改为 **NVIDIA Isaac Lab Arena G1 Loco-Manipulation — Box Pick-and-Place**。任务要求 Unitree G1 从货架抓起棕色盒子，并移动、协同全身关节后把盒子放入右侧桌上的蓝色箱子。第一轮不混入其他物体、技能链或开放世界语言泛化。

选择它的关键不是任务更容易，而是归因链更完整：Arena 官方同时提供仿真环境、G1 demonstrations、GR00T 原生 43-DoF data/action adapter 和在当前任务上调优的 checkpoint。官方冻结 GR00T N1.5 已在远端 seed 0 闭环得到 `1/1` 成功，证明环境、动作表示、checkpoint 与 evaluator 至少能够形成非零闭环；这比在 MS-HAB 上自行桥接 Fetch 低层命令更适合作为第一条可学习基线。

MS-HAB 的数据、reference 和单轨迹失败诊断全部保留，但不再承担主开发模型选择。BEHAVIOR 2026 继续作为方法冻结后的复杂长时程压力测试。现行版本、训练预算、结果和停止条件见 [Arena G1 GR00T BM-0 部署记录](arena-g1-bm0-deployment.md)；上一轮 MS-HAB 决策已标记为 superseded。

一个硬性规则仍然成立：**官方任务调优 checkpoint 的成功只定义 frozen B0，不替代公平训练。** B1–M3 必须在同一 Arena demonstrations、seed/initial-state manifest、视觉输入、action adapter 和训练预算下比较；`1/1` 只算链路 gate，正式报告必须扩展 episodes 和 seeds。

## 3. 整套系统如何工作

### 第一层：用真实时间构造多跨度全身状态监督

目标系统的 `30 Hz` 示教必须保留真实时间戳，并把底盘 SE(2) 位姿、躯干/双臂关节和夹爪组成联合状态 `Q_t`。训练样本不再只有“下一帧 action”，而是从同一起点构造多个真实时长的终态：

\[
(Q_t,\Delta t,Q_{t+\Delta t}),
\qquad \Delta t\in\{1,2,3,\ldots\}\text{ 个数据间隔}.
\]

对真实 `30 Hz` 数据，至少覆盖约 `33/67/100 ms`，对应部署中的 `30/15/10 Hz`。当前 Arena demonstrations 是理想化 `50 Hz`；发布的预转换 LeRobot 缺少实测 root pose，但同一冻结 revision 的完整 HDF5 保留了逐帧 `robot_pos/robot_quat`。Phase 1 已按官方转换器从 HDF5 重建严格对齐的视频、关节和实测 root 轨迹，再以 `20 ms` 整数跨度构造相对 SE(2) target。插值得到的中间时刻仍只是建模假设，不能冒充真实观测；随机 jitter 的可信监督由闭环仿真 rollout 验证，未来真实数据仍必须保留 timestamp/odometry。

无论数据源是哪一个，target 都必须随 `Δt` 改变：模型看到 `100 ms` 时预测 `100 ms` 后的状态，而不是对所有 `Δt` 复用同一行 action。若命令积分得到的 Arena 位姿与专家 replay 的实测 root 轨迹误差过大，M1 数据 Gate 直接失败，不能继续训练。

### 第二层：VLA action expert 输出一个可查询的全身状态流

视觉、语言和本体状态编码器产生语义 latent `z_t`，action expert 输出状态转移算子 `\Phi_\theta` 的参数。第一版可以用增广的自主连续动力系统实现：

\[
X=(Q,h),\qquad \dot X=F_\theta(X;z_t),
\]

其中 `h` 是随状态流一起演化的 action latent，用来表达“先移动、再伸手”等内部阶段。也可以使用具有同等组合性质的离散 flow。关键不是必须叫 Neural ODE，而是 action head 的原生输出就是 `Q_t \rightarrow Q_{t+\Delta t}`，并在训练中接受半群约束。这样端到端 VLA 直接决定跨 `Δt` 的全身运动，不依赖部署后把速度临时积分成 waypoint。

### 第三层：同一状态流同时给出位置参考和速度前馈

控制时按真实 wall-clock 查询状态流：

\[
Q^{ref}(t)=\Pi_Q\Phi_\theta^{t-t_0}(X_{t_0},z_{t_0}).
\]

对底盘位姿 `T_b(t)`，状态流的导数给出 body twist `\big(T_b^{-1}\dot T_b\big)^\vee`；对关节位置则给出 `\dot q`。固定 tracker/WBC 同时接收位置参考和这些导数前馈。position reference 防止长 hold 继续积累位移，velocity feedforward 则减少纯位置 waypoint 的迟滞和默认性能损失。

tracker 不是论文的 novelty，所有 position、spline、Neural Dynamic Policy 和 SC-WBP 基线都必须使用同一 tracker、同一限幅和同一控制频率。

当前 Arena action adapter 只给上身绝对关节位置，没有独立的 arm-velocity feed-forward 槽。因此 Phase 1 真正下发的是底盘 twist 前馈；上身导数先只作 telemetry 和后续接口预留。只有换成明确支持位置+速度的 WBC 接口后，才能声称执行了全身速度前馈。

### 第四层：随机推理到达时从实测状态重新锚定

VLA 可以在 `10–30 Hz` 的任意时刻返回新状态流。两次推理之间，控制器继续按真实经过时间查询上一条流；新结果到达后，从机器人**实测当前状态** `Q_t` 重新锚定。这样，算力偶尔变慢只会让旧计划多承担几十毫秒，不会把一条默认 `33 ms` 的速度 action 错当成任意时长的位移增量。

重锚定仍会引入闭环差异，所以第三阶段还要随机化重规划 schedule 训练和评测。但这里没有事件触发器：何时推理仍由外部调度器决定，研究对象只是模型对已发生的真实 `Δt` 是否具有稳定的状态转移语义。

## 4. Gate 与三个研究 Phase

前面的 Gate 用来准备实验或判断问题是否存在；只有真正改变模型或训练目标的步骤才叫 Phase。

### Gate 0：先固定原始 GR00T 的闭环基线

Gate 0 不训练新模型，也不检验本文提出的时间表示。它只回答一个更基础的问题：在官方原生支持的任务、动作 adapter 和任务调优 checkpoint 上，未经本项目改造的 GR00T 到底能做到什么程度。后续 B1–B4 和 M1–M3 都必须使用同一评测协议；否则新方法即使数字更高，也无法归因。

Gate 0 固定为：

| 项目 | 冻结值 |
|---|---|
| Benchmark | NVIDIA Isaac Lab Arena G1 Loco-Manipulation — Box Pick-and-Place |
| Policy | 官方任务调优 GR00T N1.5，未经本项目训练或微调 |
| Checkpoint | `nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation` @ `629479fedb1cf97c2f11ddc49eed951c5b750139` |
| Arena / Isaac Lab / GR00T | `f479817` / `3c6e67b` / `3bce553` |
| Rollouts | seeds `0..49`，每个 seed 恰好一次，共 50 次 |
| 单次预算 | 最多 1,200 environment steps；首次 termination 后立即停止 |
| 超时规则 | 1,200 steps 内未 termination 计为一次失败，不从分母删除 |
| 汇总 | 成功次数 / 50，并报告二项 Wilson 95% CI 和逐 seed JSON |

单次 seed 0 的 `1/1` 只证明 pipeline 能运行，不能作为 Gate 0 成绩。Gate 0 的正式结果必须来自完整的 50-seed manifest；运行版本、日志和机器可读结果同时遵循 [Arena G1 BM-0 部署手册](arena-g1-bm0-deployment.md) 中的冻结协议。

#### Gate 0 正式结果（2026-08-15）

| 指标 | 结果 |
|---|---|
| Rollouts | 50（seeds `0..49`，无缺失、无重复） |
| 成功 / 失败 | 44 / 6 |
| 成功率 | **88.0%** |
| Wilson 95% CI | **[76.2%, 94.4%]** |
| 失败 seeds | `2, 22, 31, 32, 37, 41` |
| 失败类型 | 6 次均运行满 1,200 steps 未 termination，按预注册规则计为失败 |

Gate 0 **通过**。这说明官方任务调优 GR00T N1.5 在冻结 Arena G1 环境中具有稳定的非零闭环能力，可以作为后续方法的原始 B0，而不是只能偶然通过 seed 0。同时，88.0% 并非饱和成绩：后续方法既要与相同 seeds 上的 B0 配对比较，也不能通过删除超时 seed 或延长单次预算获得优势。

机器可读证据保存在远端：

```text
/workspace/time-invariant-whole-body-policy/artifacts/validation/arena-g1-bm0/
  gate0-frozen-official-seeds-0-49/summary.json
  gate0-frozen-official-seeds-0-49/runs/seed-0.json ... seed-49.json
/workspace/time-invariant-whole-body-policy/logs/arena_g1_bm0/
  gate0-frozen-official-seeds-0-49/seed-0.log ... seed-49.log
```

### Gate N：先确认重规划频率问题真实存在

Gate N 已重新定义。旧版把“同一任务的快慢示教处于不同进度”视为未来动作标签歧义，但同一任务只保证目标相同，不保证相同墙钟时刻具有相同状态和动作。当前数据还缺少实测 root velocity、物体 pose 和接触状态，因此旧匹配无法排除未观测状态差异。旧版离线结果已经否定该支线并归档在 [Gate N v0 记录](arena-g1-gate-n.md)，不再作为项目主问题。

新版 Gate N 研究部署端的时钟一致性。Arena 以 200 Hz 做物理仿真、50 Hz 消费 action；GR00T 一次输出 16 行，官方默认完整执行后才重新推理，所以实际重规划频率是 `50/16=3.125 Hz`。同一行中，上肢是绝对关节位置，底盘却是 `vx/vy/yaw_rate`。速度会随时间积分，位置目标不会以同样方式无界累积，因此混合 action 的隐式时间语义可能造成底盘漂移和 base–arm phase mismatch。

第一轮固定 checkpoint、seed、200 Hz physics、50 Hz control 和每行 `0.02 s`，只比较执行 `16/8/4` 行后重新推理，即 `3.125/6.25/12.5 Hz`。每个条件使用相同的 10 个 seed；候选条件执行参考条件保存的同一首块动作，并要求到候选第一次重规划边界为止的 root、关节、双腕和命令 telemetry 逐位一致。完整相机输入哈希由于跨进程渲染不确定性只作诊断，不作有效性门槛。主指标是配对底盘 XY/yaw 轨迹误差、base–arm 进度相位和任务成功率。

新版 Gate N 已完成 3 个频率、每个 10 seeds 的 30 条跨频率闭环。所有配对都执行相同首块动作，且到候选条件第一次重规划之前的 root、关节、双腕和命令 telemetry 逐位相同。相对 3.125 Hz，6.25/12.5 Hz 的底盘 XY 轨迹 RMS 配对中位数为 `0.543/0.632 m`，成功率从 `8/10` 降到 `5/10` 和 `4/10`；两者均越过预注册的轨迹与成功率门槛，主筛查为阳性。yaw RMS 中位数只有 `0.044/0.052 rad`，没有越过 yaw 门槛，所以跨频率信号主要位于底盘平移路线和任务结果，而非普遍的大幅转向误差。

追加的 10-seed 同频复现中，完全保持 3.125 Hz 时底盘 XY 轨迹 RMS 中位数仍有 `0.123 m`，成功率从 `8/10` 到 `7/10`。这说明跨进程闭环本身会放大渲染和数值微扰，因此不能把全部 XY 差异都严格归因于 inference frequency，更不能直接断言根因就是 `delta_t`。但这项审计不再推翻前面的工程事实：实际部署只改重规划频率后，成功率下降了 `30/40` 个百分点，明显大于同频复现的 `10` 个百分点下降，原 Policy 的频率鲁棒性确实不够。

为避免再把两个问题混在一起，Gate N 现在分成两段：**Gate N-A 问问题是否真实且值得处理，已经通过；Gate N-B 问普通同预算微调、真实 `delta_t` 输入和 SE(2) waypoint 是否都无能为力，只有这一段也通过，才允许声称需要新的移动 VLA 策略。** 同频审计保留为因果解释与置信度限制，而不是用来把已观察到的部署失败改写成“问题不存在”。冻结协议、逐步人话解释和机器可读结果见 [Gate N frequency v1](arena-g1-gate-n-frequency.md)。

### Gate N-B 第 1 步：把 `timestamp-aware` 的意思说准确

这里的 `timestamp-aware` 不是在 Policy 外面做动作平均，也不是把旧 action chunk 按时间插值。它的定义是：把这次闭环重规划的真实间隔直接作为 Policy 的一个数值状态输入。

三个部署条件分别输入 `0.32/0.16/0.08 s`。实现中使用
`log2(delta_t / 0.16)`，所以三个值变成 `+1/0/-1`，再追加到普通机器人状态之后。模型看到的图像、语言、关节状态和 action target 都不变。

训练数据本身没有“同一个场景在三种闭环重规划频率下应该输出什么不同动作”的标签。因此训练时把每个样本随机配上三个 `delta_t` 之一，但仍预测原示教 action chunk。这一基线回答的是一个很窄但必要的问题：**只让现有行为克隆 Policy 知道真实时钟，是否已经足够？** 如果失败，不能外推成“所有 timestamp-aware Policy 都无效”；它只说明还需要能教会模型处理计划截断或多频率闭环的一致性监督。

#### 用人话说，这一步说明什么

我们真的把“多久会再问我一次”告诉了模型，不是替模型在外面把动作抹平。但示教没有告诉它“如果四步后计划就会被打断，该怎样改写这 16 步”，所以这一步同时检验：一个时钟数字本身够不够。

### Gate N-B 第 2 步：先冻结公平对照和失败标准

`delta_t` Policy 使用 2 epoch、47,468 steps、LoRA rank 32、学习率 `1e-4`；LLM 与视觉塔冻结，只训练 projector 与 diffusion action head。普通 matched LoRA 使用相同数据和同一训练预算，唯一差别是不输入 `delta_t`。这样可以排除“多训练两轮自然会变好”。SE(2) waypoint 则不训练模型，直接使用官方冻结 Policy；它只改底盘 `vx/vy/yaw_rate` 三维，上肢与 base height 原样执行。

每种方法都跑 `3 个频率 × 10 seeds = 30` 条闭环。一个方法必须同时满足下面四项才算抵抗频率变化：

1. 6.25 Hz 和 12.5 Hz 的配对 root XY RMS 都小于 `0.10 m`；
2. 两档 yaw RMS 都小于 `0.15 rad`；
3. 两档成功率相对该方法自己的 3.125 Hz 变化都小于 `20` 个百分点；
4. 该方法在默认 3.125 Hz 的成功率，不能比公平参考下降超过 `10` 个百分点。

matched LoRA 与 SE(2) 的跨频率配对强制执行相同首块，并验证分叉前物理 telemetry 相同。`delta_t` Policy 不强制首块相同，因为不同 `delta_t` 正是要测试的输入；若把 0.32 s 条件的首块强塞给其他频率，反而会抹掉处理效应。

#### 用人话说，这一步说明什么

不能把机器人在三个频率下一起变差叫“频率鲁棒”，也不能看到 `delta_t` 版本更好就忘了普通微调可能同样有效。这张考卷要求它既守住默认能力，又让两种实际改频都不再明显改变路线和成功率。

### Gate N-B 第 3 步：普通同预算 LoRA 没有解决问题

| 重规划频率 | 3.125 Hz | 6.25 Hz | 12.5 Hz |
|---:|---:|---:|---:|
| 成功 / 10 | **10/10** | **2/10** | **1/10** |
| 相对默认成功率变化 | — | **-80 pp** | **-90 pp** |
| 相对默认 root XY RMS | — | **0.633 m** | **0.645 m** |
| 相对默认 root yaw RMS | — | 0.049 rad | 0.051 rad |

默认频率比原始 Policy 的 `8/10` 更好，但提前重规划时反而只剩 `2/10` 和 `1/10`。两档 XY 与成功率变化都远超门槛。

#### 用人话说，这一步说明什么

模型不是“训练得还不够”。同样多训两轮可以让它更擅长训练时熟悉的默认节奏，却没有让它学会计划被提前打断；它甚至变得更依赖这个节奏。

### Gate N-B 第 4 步：输入真实 `delta_t` 仍然不够

正式训练完整达到 `47,468/47,468` steps，2 epochs，最终 train loss 为 `0.00417`。合并后的 `delta_t` Policy 在三档频率上的结果是：

| 重规划频率 | 3.125 Hz | 6.25 Hz | 12.5 Hz |
|---:|---:|---:|---:|
| 成功 / 10 | **9/10** | **1/10** | **4/10** |
| 相对默认成功率变化 | — | **-80 pp** | **-50 pp** |
| 相对默认 root XY RMS | — | **0.658 m** | **0.266 m** |
| 相对默认 root yaw RMS | — | 0.051 rad | 0.060 rad |

它的默认成功率比 matched LoRA 低 `10` 个百分点，刚好还在允许范围内；但两档非默认频率的 XY 与成功率变化都没有过线，因此总体失败。

#### 用人话说，这一步说明什么

模型确实收到了真实 `delta_t`，但“知道多久后会重新推理”不等于“知道怎样让反复重规划后的闭环结果保持一致”。当前训练目标对三个时钟给的是同一份示教答案，最容易学到的是忽略这个数字，而不是学会补偿计划截断。后续若研究时钟条件，必须加入多重规划节奏下的监督或一致性目标，不能只再加一个 scalar。

### Gate N-B 第 5 步：SE(2) waypoint 改变了趋势，但也没有解决

SE(2) 基线把每个底盘 twist chunk 从实测 root pose 积分成带物理时刻的世界系位姿参考；重叠计划按同一个控制 step 对齐，再由 feed-forward 加比例反馈的 tracker 执行。结果是：

| 重规划频率 | 3.125 Hz | 6.25 Hz | 12.5 Hz |
|---:|---:|---:|---:|
| 成功 / 10 | **5/10** | **6/10** | **8/10** |
| 相对默认成功率变化 | — | +10 pp | **+30 pp** |
| 相对默认 root XY RMS | — | **0.417 m** | **0.231 m** |
| 相对默认 root yaw RMS | — | 0.059 rad | 0.060 rad |

它在高频下不再像 raw velocity 那样掉成功率，甚至变好；但默认 3.125 Hz 从原始 Policy 的 `8/10` 降到 `5/10`，损失 `30` 个百分点。两档 XY 差异仍高于 `0.10 m`，12.5 Hz 的成功率变化也高于 `20` 个百分点。因此它是一个有启发性的组件，不是合格的频率鲁棒修复。

#### 用人话说，这一步说明什么

waypoint tracker 确实能把“更频繁更新就更差”扭成另一种行为，但它没有让同一个任务在三种频率下得到同一种后果，还伤害了原本的默认表现。把速度积分成位置目标不是这项研究的终点。

### Gate N-B 第 6 步：最终判定

| 方法 | 3.125 / 6.25 / 12.5 Hz 成功 | 6.25 / 12.5 Hz XY RMS | 是否守住默认能力 | 是否抵抗频率变化 |
|---|---:|---:|---:|---:|
| 原始官方 Policy | 8 / 5 / 4 | 0.543 / 0.632 m | 参考 | 否 |
| 普通 matched LoRA | 10 / 2 / 1 | 0.633 / 0.645 m | 是 | 否 |
| 真实 `delta_t` Policy | 9 / 1 / 4 | 0.658 / 0.266 m | 是 | 否 |
| SE(2) waypoint | 5 / 6 / 8 | 0.417 / 0.231 m | **否** | 否 |

所有 90 条新 rollout 与 90 份 telemetry 都完整，matched LoRA 和 SE(2) 的首块/分叉前物理轨迹校验通过，日志中没有 traceback、OOM、磁盘写满或 segmentation fault。机器判定为：

```text
gate_n_a_frequency_problem_observed: true
gate_n_a_passed: true
all_simple_baselines_failed: true
gate_n_b_new_policy_necessity_passed: true
mobile_vla_policy_research_warranted: true
decision: proceed_to_mobile_vla_policy_research
```

因此 **Gate N 最终通过**。这里的“通过”不是说当前方法成功，而是说两件事都成立：原始移动操作 Policy 在实际推理频率变化下确实会明显改变后果；真实 `delta_t` 输入、普通同预算微调和非学习 SE(2) waypoint 都没有以可接受代价解决它。现在有依据进入面向移动机器人的新 VLA 策略研究。

#### 最终用人话说，这一步说明什么

我们先证明机器人换个推理节奏就会明显换一种表现，又认真试了“多训练一点”“把真实时间告诉模型”“把底盘速度换成位置点执行”三条便宜路线。它们都没交出同时保住默认能力和跨频率一致性的答案。所以现在研究新的 Policy 机制不是为了包装一个已有简单解法，而是在解决这些强基线确实没解决的问题。

结论边界也必须保留：这只是当前 Arena G1 Box Pick-and-Place、当前 GR00T checkpoint 和 10 seeds 的证据，不证明所有移动机器人或所有 timestamp-aware 训练都会失败。它证明的是：**在原 action target 不变时追加真实 `delta_t` 不够；训练后只把底盘 twist 转成 waypoint 也不够。** 下一阶段应直接学习带真实时间语义的全身状态转移，并检验时间分割一致性。冻结的 [Gate N-B 协议](../configs/arena-g1-gate-n-consumers-v1.yaml) 与 [机器判定](../artifacts/validation/arena-g1-gate-n/consumer-v1/gate-evaluation.json) 可直接复核。

### Phase 1：先让模型回答“过一会儿，全身应该在哪里”

Phase 1 只改一件事：**把“上身去哪个关节位置、底盘开多快”的混合答案，改成“在这些真实时刻，全身参考应该在哪里”。** 它不加半群损失、不换 backbone，也不训练随机重规划；先用最小改动判断位置路线图本身是否已经够用。

具体代码契约已经落在 [`arena-g1-phase-1-m1-v1.yaml`](../configs/arena-g1-phase-1-m1-v1.yaml)，SE(2)、多跨度 target、路线查询和 tracker 的可测试实现位于 [`src/whole_body_policy`](../src/whole_body_policy)，自动化测试位于 [`tests/test_phase1_m1.py`](../tests/test_phase1_m1.py)。远端数据、训练、接线和 artifact 的逐项执行要求专门放在 [Phase 1 AI 实施规格](phase-1-ai-implementation.md)，避免把给机器看的接口细节塞进本文。

#### 1. 先把“正确答案”做对

一条示教不再只提供“下一行 action”。以当前时刻为起点，数据处理器会同时问：`20 ms、40 ms、……、320 ms` 后，上身关节、手、base height 和底盘分别应该在哪里。目标机器人的 `30 Hz` 数据同理按真实 timestamp 查询约 `33/67/100 ms` 等时刻，而不是把 frame index 当成秒。

底盘答案用“相对当前底盘的前后/左右位移和转角”，上身继续用对应未来时刻的绝对关节位置。这样每一行都在回答同一个物理时刻，不再让底盘讲速度、手臂讲位置。

这里有一个必须先过的现实门槛：当前 Arena LeRobot 数据没有实测 root pose。代码允许用导航速度积分出一个代理路线，但 artifact 必须写明 `command_integration_proxy`，并在 `320 ms` 局部窗口与 expert replay 的实测 root telemetry 比较。预注册上限是 XY p95 `0.05 m`、yaw p95 `0.08 rad`；超过就停用代理并补实测 odometry，不能把代理改名成真值或事后放宽门槛。

#### 2. 模型仍输出 `16×32`，但最后三维换了含义

为了复用官方任务 checkpoint，第一版不扩大 action head：

```text
前 28 维   双臂和双手的未来绝对位置
第 29 维   未来 base height
后 3 维    相对本次计划起点的底盘 SE(2) 位姿
每一行     分别对应 0.02、0.04、……、0.32 秒
```

可以把这 16 行理解成一张 320 ms 的短路线图。这里的“全身”是 Policy/WBC 分层意义：GR00T 同时安排上身、base height 和底盘路线，Homie v2/WBC 仍负责把底盘意图变成下身关节动作；不能写成 GR00T 直接预测全部 43 个关节。

训练时不能简单把 parquet 每一行的速度改成一次相对位姿。原因很直白：同一个未来位置相对不同起点会有不同答案。loader 必须先固定当前 anchor，再一次构造相对这个 anchor 的 16 个未来 target。新表示也必须重新计算 train-split statistics，旧速度统计量不能继续套用。

#### 3. 执行器按表走，不把位置当速度下发

模型结果生效时，执行器从实测当前状态重新立一个路线图起点。50 Hz tracker 每个 tick 读取真实 wall clock，在相邻两个位置点之间查询参考；底盘的相邻位姿还会给出速度前馈，再叠加当前位姿误差反馈。超过 `320 ms` 仍没有新计划时，保持最后位置并把前馈速度清零。

这一步必须插在 GR00T 32 维输出和 Arena 官方 `32→50` adapter 之间。如果把 M1 最后三维直接交给旧 adapter，它会把 `dx/dy/dyaw` 当成 `vx/vy/yaw_rate`，实验立即失效。

Arena 当前接口没有上身速度前馈槽，所以第一版只真正执行底盘速度前馈；上身导数只记录 telemetry，上身仍执行绝对关节位置。论文不能声称当前 Arena 已执行 arm velocity feed-forward。

#### 4. Phase 1 能证明什么，不能证明什么

Phase 1 只回答：**学习一张带真实时间刻度的 whole-body position 路线图，是否已经足以解决当前频率敏感性？**

它没有强制“一次预测 100 ms”必须等于“连续预测两次 50 ms”，也没有训练模型在中途看到新图像后仍保持组合一致。这两个问题分别留给 Phase 2 和 Phase 3。若 M1 失败，先判断是数据真值、归一化、tracker、默认能力还是跨频率闭环失败，不能自动把失败解释成“半群网络必需”。

#### 5. 怎样判断 Phase 1 是否已经够用

M1 使用和强基线相同的 tracker、限幅、训练预算与 seeds，并接受三组检查：

1. 默认频率成功率相对同预算 matched LoRA 下降不超过 `10` 个百分点；
2. 现有 `3.125/6.25/12.5 Hz` Gate N 中，两档非默认频率的 root XY RMS 都小于 `0.10 m`、yaw RMS 都小于 `0.15 rad`、成功率变化绝对值都小于 `20` 个百分点；
3. 目标设备固定 `10/15/20/30 Hz` 和随机 `10–30 Hz` 下，底盘漂移、边界跳变、base–arm phase error 和安全指标不退化。

如果 Phase 1 全部通过，结论必须收缩为：**学习全身位置参考已经够用，没有证据证明 Phase 2 的半群机制是必要的。** 此时停止增加模型复杂度。

#### 6. 2026-08-16 远程全流程结果

Phase 1 已在冻结远程环境中从数据重建跑到 80 条正式闭环，不再是待执行方案。完整 HDF5 经官方转换器重建后得到 100 条示教、85,889 帧和 84,289 个有效 anchor。命令积分代理在 100/100 条 episode 上都没有通过预注册误差门槛，因此正式 target 全部改用原始数据中的实测 root pose；门槛没有放宽。

训练前的单 batch 和单 episode overfit 都通过。正式训练严格执行 47,468 个 optimizer step，loss 从 2.5935 降至 0.0278，最小值 0.0074；合并后的 checkpoint 约 7.1 GB。单 episode 解归一化检查的上身 RMSE 为 0.0672 rad，base height 为 0.00409 m，底盘 XY 为 0.0106 m、yaw 为 0.00756 rad。这些结果排除了明显的 shape、NaN 和单样本接线错误，但不能证明全量数据上的 target/归一化一定正确。

正式闭环覆盖 10 个 seeds 和 8 种 schedule：`3.125/6.25/12.5/10/15/20/30 Hz` 加 `10–30 Hz` jitter，共 80/80 条 artifact 完整。所有条件都是 0/10 成功，全部 80 条均通过有限值、速度限幅和提前终止安全检查。因此 M1 **没有通过 Phase 1**：默认 3.125 Hz 相对 matched LoRA 的 10/10 下降了 100 个百分点，远超允许的 10 个百分点。

跨频率轨迹也没有守住门槛。6.25 Hz 相对默认档的 root XY/yaw RMS 中位数为 0.330 m/0.897 rad；12.5 Hz 为 0.599 m/1.010 rad，均高于 0.10 m/0.15 rad。不过这部分只能作为描述性失败：虽然配对回合执行的首个模型 action chunk 相同，分叉前实测物理轨迹并不逐位一致，所以不能把差异干净地归因于重规划频率。

目前最强的故障线索在 tracker 与仿真的时钟域。路线图只覆盖 0.32 s，而默认档跨 seeds 的 plan-age 中位数为 0.401 s，每条 1,200-step rollout 中位有 705.5 步已经越过 horizon、进入“保持末位姿且前馈清零”。这说明非实时仿真使用进程墙钟查询物理时间路线图会产生系统性错位；它是首要候选原因，不是已经证明的唯一原因，因为没有 horizon clamp 的高频条件同样 0/10，仍需排查全量 target/statistics 和 re-anchor 接线。

冻结决策为 `stop_and_classify_phase_1_m1_failure_before_any_phase_2_work`：当前不得把失败写成“半群机制必需”，也不得直接启动 Phase 2。下一步只允许做有边界的 Phase 1 时钟域、归一化和 tracker ablation。逐字段执行记录、revision、checksum 和 artifact 位置见 [Phase 1 AI 实施规格](phase-1-ai-implementation.md#11-2026-08-16-远程执行记录)。

### Phase 2：让同一张短路线图不怕被切成不同时间步

先把最重要的边界说清楚：**按照 Phase 1 的执行方式，控制器直接根据真实墙钟时间查询同一张路线图，并不会为了得到 `100 ms` 目标而递归调用两次 `50 ms` 模型。** 因此 Phase 2 不是执行 Phase 1 的必要条件，也不直接等于 Gate N 的解法。

Phase 2 是一个额外的模型机制实验：如果我们希望 action head 本身不仅会输出一组位置点，还能被解释成一个可以反复组合的状态转移规律，那么它的一步预测和拆步预测就应该一致。

Phase 1 的各个未来位置可能是分别猜出来的。例如它可能认为：

\[
\text{从现在直接走 }100\mathrm{ms}\text{ 后，底盘前进 }10\mathrm{cm},
\]

同时又认为：

\[
\text{先走 }50\mathrm{ms}\text{ 前进 }6\mathrm{cm},
\quad
\text{再走 }50\mathrm{ms}\text{ 又前进 }6\mathrm{cm}.
\]

同样经过 `100 ms`，一个答案是 `10 cm`，另一个答案是 `12 cm`。只把 Phase 1 的整张路线图按时间插值执行时，这个矛盾不一定暴露；当模型被递归用作状态转移、改变查询 horizon，或需要比较一步预测与滚动预测时，它才会暴露。

所以 Phase 2 只解决这个更窄的矛盾：

> **同一份计划经过相同的物理时间，不管一次算完还是拆成多步执行，都应该得到同一个全身状态。**

#### 1. 模型不再分别猜每个位置，而是生成一条“运动规律”

一次 VLA 推理先把图像、语言和机器人状态压成一个上下文 `z_t`。可以把 `z_t` 理解成模型对当前任务的短期判断，例如“盒子在右前方，现在先移动并抬起右臂”。

然后 action expert 生成一条全身运动规律：

\[
X=(Q,h),
\qquad
\frac{dX}{d\tau}=F_\theta(X;z_t).
\]

这里：

- `Q` 是外面看得见的机器人全身位置；
- `h` 是模型内部随运动一起传递的小记忆，用来区分“正在靠近”“正在伸手”等阶段；
- `\tau` 是真实物理时间，单位是秒；
- `F_\theta` 回答“从当前状态继续过一小会儿，状态应该怎样变化”。

从起点沿这条规律运行 `\tau` 秒，得到：

\[
X_{t+\tau}=\Phi_{z_t}^{\tau}(X_t),
\qquad
Q_{t+\tau}^{ref}=\Pi_Q X_{t+\tau}.
\]

VLA 每次只需要生成一次这条运动规律或它的参数。底层控制器可以高频查询它，不需要每个控制 tick 都重新运行大模型。

这里必须避免一个名词陷阱：GR00T 原有的 **flow matching** 描述的是“模型怎样把噪声变成一份 action sample”，其中的 flow time 是生成算法内部变量；Phase 2 的 **state flow** 描述的是“机器人经过多少真实秒后应该在哪里”。这两种 flow 不是同一件事，只有后者直接对应部署频率。

#### 2. “一步等于拆步”的公式是什么

Phase 2 要满足两个规则：

\[
\Phi^0(X)=X,
\]

表示时间没有经过时，机器人不能凭空移动；以及：

\[
\boxed{
\Phi^{\tau_1+\tau_2}(X)
=
\Phi^{\tau_2}\!\left(\Phi^{\tau_1}(X)\right)
}
\]

表示先走 `\tau_1`、再走 `\tau_2`，必须等于一次走完 `\tau_1+\tau_2`。

例如：

\[
\Phi^{100\mathrm{ms}}(X)
=
\Phi^{50\mathrm{ms}}\!\left(\Phi^{50\mathrm{ms}}(X)\right)
=
\Phi^{34\mathrm{ms}}\!\left(
\Phi^{33\mathrm{ms}}\!\left(\Phi^{33\mathrm{ms}}(X)\right)
\right).
\]

因此，只要还是同一次 VLA 推理生成的同一份计划，`100 ms` 被控制器切成 1 步、2 步还是 3 步，理论终点都相同。内部记忆 `h` 必须跟着每个短步传下去；如果每一步都把 `h` 重置，公式就不成立。

#### 3. 怎么训练这个性质

训练首先仍要模仿示教中的真实未来位置：

\[
\mathcal L_{state}
=d_{\mathcal Q}\!\left(
\Pi_Q\Phi_{z_t}^{\tau}(X_t),
Q_{t+\tau}
\right)^2.
\]

这保证模型没有为了满足漂亮公式而忘记任务。然后，对同一个起点、同一张图、同一个 `z_t` 和同一份生成噪声，同时计算“一步答案”和“拆步答案”：

\[
\mathcal L_{partition}
=d_{\mathcal X}\!\left(
\Phi^{\tau_1+\tau_2}(X_t),
\Phi^{\tau_2}(\Phi^{\tau_1}(X_t))
\right)^2.
\]

训练让 `\mathcal L_{partition}` 尽量接近 0。固定图像、上下文和生成噪声很重要，否则比较的会是“两次随机生成为什么不同”，而不是“时间为什么不能拆分”。

如果使用数学上天然满足组合律的自主动力系统，公式主要由模型结构保证，`\mathcal L_{partition}` 用来检查数值求解和实现误差；如果使用普通的时间条件网络，组合律并不天然成立，这个 loss 就是主要约束。

#### 4. Phase 2 能解决什么，不能解决什么

Phase 2 能保证的是：

> 固定同一次推理得到的 `z_t` 和同一条状态流，只改变控制器怎样切分已经过去的物理时间，不应该改变结果。

它还不能保证 `10 Hz` 和 `30 Hz` 的 VLA 闭环一定相同。因为 `30 Hz` 会更早看到新图像、生成新的 `z`，还会重新采样一条新计划；这已经不是“同一条状态流被怎样切分”的问题。这个缺口留给 Phase 3。

因此，如果 Phase 1 的带时间位置路线已经通过 Gate N，Phase 2 应立即停止。不能为了获得一个漂亮的半群公式，继续增加对真实问题没有收益的模型复杂度。

#### 5. 怎样判断 Phase 2 是否有效

先固定图像、`z_t` 和随机噪声，比较总时长相同的：

\[
[100],\quad[50,50],\quad[33,33,34]\ \mathrm{ms},
\]

以及更多随机拆分。报告底盘 SE(2)、上肢关节和内部状态的 partition error。

然后再做闭环测试。M2 必须同时满足：

1. partition error 显著小于 M1、连续样条和普通时间条件网络；
2. 默认频率成功率不能下降超过 `10` 个百分点；
3. Arena Gate N 和随机频率闭环表现优于 M1，证明公式上的一致真的改善了机器人行为。

如果 partition error 下降但闭环结果没有改善，说明这个数学性质不是当前频率失败的主要原因，Phase 2 的核心假设被否定。如果已有 Neural Dynamic Policy 基线达到相同结果，则不能把动力系统或半群结构本身写成 novelty。

### Phase 3：让机器人在随机频率下反复重新规划也不崩

Phase 2 只管**同一张短路线图内部**。实际 VLA 推理频率改变时，机器人使用的并不是同一张路线图：它会在不同时间重新拍照、重新理解场景、重新随机生成计划。

所以 Phase 3 才是直接对应 Gate N 的阶段。

#### 1. 为什么 Phase 2 之后仍然有问题

假设观察同一个 `100 ms` 时间段：

| 推理频率 | 实际发生的事情 |
|---|---|
| `10 Hz` | `t=0` 看一次图，生成一张计划并连续执行约 `100 ms` |
| `30 Hz` | `t=0、33、67 ms` 分别看图，前后生成三张不同计划 |

Phase 2 可以保证第一行中的一张计划怎样拆步都一致，却不能保证第二行中的三张新计划组合后仍与第一行相同。用公式写，`10 Hz` 大致是：

\[
Q_{100}^{10\mathrm{Hz}}
=\Pi_Q\Phi_{z_0}^{100\mathrm{ms}}(X_0),
\]

而 `30 Hz` 是三个不同 observation 产生的重新规划算子 `\mathcal R` 连续组合：

\[
Q_{100}^{30\mathrm{Hz}}
=
\mathcal R_{o_{67}}^{34\mathrm{ms}}
\circ
\mathcal R_{o_{33}}^{33\mathrm{ms}}
\circ
\mathcal R_{o_0}^{33\mathrm{ms}}(Q_0).
\]

这里 `o_0、o_{33}、o_{67}` 是三个时刻的新观察。因为三个 `\mathcal R` 不相同，Phase 2 的半群公式不能直接约束它们。

如果用 `\mathcal R_\theta(o,Q,\tau)` 表示“看到 observation `o`、从实测状态 `Q` 重新规划并执行 `\tau` 秒后的状态”，真正对应推理频率的模型级目标是：

\[
\boxed{
\mathcal R_\theta(o_t,Q_t,\tau_1+\tau_2)
\approx
\mathcal R_\theta\!\left(
o_{t+\tau_1},
\mathcal R_\theta(o_t,Q_t,\tau_1),
\tau_2
\right)
}
\]

左边表示低频：现在看一次图，然后连续执行 `\tau_1+\tau_2`。右边表示高频：先执行 `\tau_1`，中途重新看图、读取实测状态，再执行 `\tau_2`。这叫**闭环时间分割一致性**，它和 Phase 2 固定 `z` 的同一计划组合律不是一回事。

这个等式只能在“环境没有出现新的外部变化，任务意图也没变”的配对片段上严格要求。如果中途物体滑落、有人挡路或发生新接触，右边看到了左边没有的新信息，改变行为才是正确答案。

#### 2. Phase 3 到底训练什么

Phase 3 不再只给模型看固定间隔的数据，而是让训练过程模拟实际设备的推理调度。先随机生成一次重规划时间表：

\[
\mathcal S=\{t_0,t_1,t_2,\ldots\},
\qquad
t_{i+1}-t_i\sim[1/30,1/10]\ \mathrm{s}.
\]

例如一条 schedule 可以是：

```text
0 ms → 41 ms → 78 ms → 144 ms → 180 ms → 279 ms
```

这代表模型有时接近 `30 Hz` 返回，有时只有 `10 Hz`，中间还会抖动。

对每个重规划时刻 `t_i`：

1. 给模型该时刻真实拍到的图像和实测机器人状态；
2. 从该实测状态生成一张新的短路线图；
3. 执行到 schedule 指定的下一次推理时刻；
4. 再从新的实测状态重新开始，而不是从上一张图预测的理想位置开始。

对离线示教，可以先按照许多随机 schedule 重新切片，让同一条示教产生 `10–30 Hz` 的不同训练版本。但这种训练只见过“机器人一直走在正确轨迹上”的状态；真实部署一旦走偏，模型可能不知道怎样回来。

因此还需要仿真闭环：让模型真的按随机 schedule 控制机器人，把它走偏后遇到的状态也保存下来。如果有仿真专家或 privileged controller，就让专家告诉模型怎样从这些状态恢复，这就是 DAgger；如果没有可靠专家，只能使用冻结的任务奖励做谨慎的小预算闭环微调，不能假装离线切片已经覆盖了闭环误差。

#### 3. 训练目标不是让不同频率逐帧输出相同动作

高频推理可能看到低频推理尚未看到的新障碍、接触或物体运动。这时改变路线是正确的，不能用 loss 强迫两者完全相同。

Phase 3 优化的是不同 schedule 下都应该保持的结果：

\[
\mathcal L_{M3}
=
\mathbb E_{\mathcal S}
\left[
\lambda_{task}\mathcal L_{task}
+\lambda_{closed}\mathcal L_{closed-partition}
+\lambda_{traj}\mathcal L_{trajectory}
+\lambda_{phase}\mathcal L_{base-arm}
+\lambda_{jump}\mathcal L_{command-jump}
+\lambda_{safe}\mathcal L_{safety}
\right].
\]

用人话说：

- `L_task`：任务最后是否成功；
- `L_closed-partition`：在没有新外部变化的配对片段中，低频一步执行和高频中途重规划是否到达相近状态；
- `L_trajectory`：没有新环境变化时，路线是否出现无理由的大幅漂移；
- `L_base-arm`：底盘和手臂是否仍处于互相配合的任务阶段；
- `L_command-jump`：新计划接管时，位置和速度命令是否突然跳变；
- `L_safety`：是否碰撞、掉落、速度越界或让控制器饱和。

成功率和安全必须优先于“轨迹长得完全一样”。不能为了让不同频率的曲线重合，逼机器人忽略新观察。

上面的式子是优化目标，不代表必须对 Isaac Sim 直接求梯度。若仿真器不可微，`L_task` 和 `L_closed-partition` 可以作为 rollout reward、样本筛选指标或专家纠正信号；可微的 state/command 项再直接反向传播。实现方式必须如实报告，不能把评测指标冒充训练 loss。

#### 4. 部署时 Phase 3 做不做事件触发

不做。外部系统仍然根据设备算力决定何时调用 VLA。Phase 3 不学习“什么时候值得推理”，只训练：

> **无论新推理在 `10–30 Hz` 中哪个时刻返回，Policy 都能从当时的实测状态安全接管。**

所以这仍然是频率鲁棒 Policy，不是事件触发 Policy，也没有扩大到通用可中断规划。

#### 5. 怎样判断 Phase 3 是否有效

至少测试：

- 固定 `10/15/20/30 Hz`；
- 每一步随机变化的 `10–30 Hz`；
- 从真实设备记录并重放的 latency trace；
- 训练没有见过的中间频率；
- 偶发一次或连续多次 deadline miss；
- 原 Arena `3.125/6.25/12.5 Hz` Gate N，保证没有换题。

M3 必须比 M2 更好地抵抗**反复重新规划**造成的漂移和成功率下降，同时守住默认频率能力。如果 M2 已经达到全部门槛，Phase 3 就没有必要；如果 M3 只对训练中列举的几个频率有效，对新 jitter 失效，就只能叫频率数据增强，不能声称连续频率鲁棒。

#### 三个 Phase 最简单的区别

| Phase | 用人话问的问题 | 它还没解决什么 |
|---|---|---|
| Phase 1 | 能不能不输出底盘速度，直接告诉全身未来在哪里？ | 一个长步和多个短步可能不一致 |
| Phase 2 | 同一张路线图无论怎样拆时间，能不能走到同一位置？ | 重新看图后会生成另一张路线图 |
| Phase 3 | 随机时间反复看图、换路线图时，机器人还能不能稳定完成任务？ | 不研究由模型自己决定何时推理 |

## 5. Phase、版本和对照关系

| 阶段 | 产物 | 这一步只增加什么 | 做完后问什么 |
|---|---|---|---|
| Gate -1 | 基础设施 | 不改方法，只锁定数据、环境和 evaluator | 实验链路可靠吗？ |
| Gate 0 | B0 | 冻结协议下运行官方 GR00T N1.5 | 原始闭环能力是多少？ |
| Gate N | B0+FT、B1、B2-I | 诊断频率问题，并测试 `delta_t` 与后处理 waypoint | 简单接口是否已经足够？ |
| Phase 1 | M1 | 多跨度全身位置状态 target 和统一 tracker | 学习 position reference 是否已经够用？ |
| Phase 2 | M2 | 模型内状态流、半群结构和 partition loss | 一步与拆步能否得到同一状态转移？ |
| Phase 3 | M3 | 随机重规划、实测状态重锚定和闭环时间分割损失 | 新观测反复重规划时还能否稳定？ |
| Gate S | 冻结后的 M3 | 不再改方法，只扩展仿真任务与 seed | 结论能否跨任务成立？ |
| Gate R | 冻结后的 M3 | 不再改方法，只做真机验证 | 对真实算力抖动是否安全有效？ |

现有 `phase_0` 和 `phase_1` 是早期远程部署批次名，不代表这里的研究 Phase。已有脚本和产物不重命名，以免失去追溯关系。后续实验同时记录部署批次、研究阶段和方法版本，例如：

```text
deployment_batch: phase_1
research_stage: phase_2
method_id: m2_semigroup_whole_body_flow
```

## 6. 必须打败哪些简单方案

新方法不能只和原始 velocity Policy 比；否则无法知道收益来自位置控制、连续插值还是半群机制。

| 版本 | 用人话解释 | 它排除的替代解释 |
|---|---|---|
| B0 | 原始 Policy：底盘 velocity、上肢 position，按固定 chunk 执行 | 当前系统起点 |
| B0+FT | 相同预算普通微调，不输入时钟 | 也许任何额外训练都会改善结果 |
| B1 | 输入真实 `delta_t`，但仍预测原固定频率 action target | 也许告诉模型下一次推理时间就够了 |
| B2-I | 不重训，把原底盘 twist 后处理为 SE(2) waypoint | 也许外部位置接口已经够用 |
| B2-WB | 重训离散 whole-body position waypoint，固定原生数据频率的 horizon | 也许只改全身 action representation 就够了 |
| B3 | Spline/NIAF 式连续全身轨迹，可按任意 `Δt` 查询，但不做 partition 训练 | 也许连续插值本身就够了 |
| B4 | NDP/Neural ODE 式动力学 action head，不做随机频率闭环训练 | 也许已有神经动力学结构已经足够 |
| M1 | 多跨度全身 position-state transition，无半群约束 | 检验 target 改造本身 |
| M2 | M1 加显式状态流与时间分割一致性 | 检验模型级 semigroup mechanism |
| M3 | M2 加随机 `10–30 Hz` 重规划和闭环时间分割训练 | 检验反复重锚定与新观测下的部署鲁棒性 |

所有版本必须使用相同视觉语言 backbone、数据量、优化步数、推理采样次数、tracker、限幅和控制频率。B3/B4 与 M2 参数量无法完全相同时，必须同时报告参数量、FLOPs、wall-clock latency 和 tracker 增益；不能让某个方法暗中获得更强控制器。

## 7. 怎么评价，而不是只看成功率

### 7.1 明确分开三只钟

- 数据频率：示教记录频率，目标设备数据为 `30 Hz`，当前 Arena 代理数据为理想化 `50 Hz`；
- 推理/重规划频率：VLA 产生新状态流的频率，主要测试 `10–30 Hz`；
- 控制频率：tracker/WBC 查询状态流并下发命令的频率，第一轮保持固定。

主实验只随机化推理频率。随后分别扰动数据采样与控制频率做归因，不能把三者统称为“frequency”。每条 rollout 必须记录真实 inference start/end、reference 生效时刻和相邻 inference 的 wall-clock `Δt`，不能用名义频率代替实测时间。

### 7.2 先做固定 latent 的时间分割单元测试

固定 observation、当前状态、语义 latent 和 flow-matching 随机噪声，对相同总时长 `T` 比较：

- 一步：`[100 ms]`；
- 均匀拆分：`[50,50] ms`、`[33,33,34] ms`；
- 随机拆分：总和仍为 `100 ms` 的多组 jitter partition。

报告底盘 SE(2) geodesic error、关节 RMSE、夹爪误差和导数误差，并画误差随拆分数量和总时长的曲线。这一测试隔离 action representation；它不能替代闭环成功率。

### 7.3 再做真实随机重规划测试

闭环测试至少包含：

- 已有 Arena `3.125/6.25/12.5 Hz`，保证新路线确实回到 Gate N 原问题；
- 固定 `10/15/20/30 Hz`；
- 每一步独立或按真实设备 latency trace 采样的 `10–30 Hz` jitter；
- 训练见过的频率端点与未见过的中间频率分开汇报；
- 偶发一次或连续数次 deadline miss；
- 相同 seeds、initial states、控制频率和任务预算下的配对 rollout。

主要指标是任务成功率、root XY/yaw 轨迹偏差、base–arm phase error、重规划边界的 position/velocity jump、累计底盘漂移和 safety violation。默认频率成功率仍是硬门槛，不能用“三个频率一起变差”换取表面一致。

### 7.4 区分表示、机制和闭环训练的贡献

至少做以下消融：

1. velocity-only、position-only、position + derivative feedforward；
2. M1 对 M2：相同 state target，只移除半群结构/`L_partition`；
3. M2 对 M3：移除随机重规划 schedule 和 `L_closed-partition`，并分别单独消融二者；
4. `L_partition` 置零、打乱 `Δt`、用 frame index 代替真实秒数；
5. 只做底盘状态流，对比完整 whole-body 状态流；
6. 固定旧 latent 持续查询，对比每次用新 observation 重规划。

第 6 项尤其重要：若固定 latent 时稳定、每次新推理时仍崩溃，说明 action 流内部的时间分割已解决，但 VLA 的闭环重锚定还没有解决，不能把 M2 写成完整频率鲁棒。

### 7.5 统计与外推边界

当前 Arena 10-seed 结果只负责快速证伪。正式主实验至少使用 3 个训练 seed，rollout 数量按成功率差异做 power analysis，并报告配对差值和置信区间。方法冻结后再扩展对象、自由空间移动、移动中操作和接触任务，最后回到 BEHAVIOR 或真机 latency trace 做压力测试。

## 8. 什么时候应该停止或收缩这条路线

以下条件必须预注册：

1. B2-WB 或 M1 已同时守住默认能力和随机频率门槛：停止声称半群机制是必要贡献，把工作收缩为全身位置 action representation。
2. B3 连续样条/NIAF 与 M2 表现相当：不能声称时间分割训练带来主要收益。
3. B4 已有 Neural Dynamic Policy 在公平预算下达到 M2/M3：novelty 只能落在 whole-body VLA benchmark、训练协议或系统验证，不能落在动力系统结构。
4. M2 的 partition error 没有显著低于 M1：模型没有学会所宣称的组合律，核心机制失败。
5. partition error 明显下降，但真实随机频率的成功率和漂移不改善：说明半群性质不是当前闭环失败的主要机制。
6. M3 只在训练列举的 `10/15/20/30 Hz` 有效，对未见连续频率或 jitter 失效：不能声称连续频率泛化。
7. 默认频率成功率相对公平参考下降超过 `10` 个百分点：即使跨频率曲线更平也判定失败。
8. 只做底盘即可获得全部收益：删除“全身联合状态流改善 base–arm 协同”的主张。
9. 换 3 个训练 seed、对象或任务后不能复现：结论降级为 Arena 单任务系统改进。
10. `L_closed-partition` 降低但随机重规划成功率没有改善：闭环组合公式不是有效机制，只能把收益归因于 schedule 数据增强或其他训练项。

## 9. 实际开发顺序

| 顺序 | 要完成的事情 | 通过标准 |
|---|---|---|
| 0 | 冻结 Gate 0、Gate N-A/N-B 和全部 telemetry | 已完成，原始证据可复核 |
| 1 | 分别处理目标 `30 Hz` 数据与 Arena `50 Hz` 代理数据 | Arena 命令积分 target 经专家 replay root telemetry 验证，目标数据保留真实 timestamp/odometry |
| 2 | 构造覆盖约 `33–100 ms` 的多跨度 target，并冻结 train/test 频率集合 | 无未来泄漏；插值标签单列，未见 jitter 留给仿真测试 |
| 3 | 固定统一 position + velocity-feedforward tracker | B2-I、B2-WB、B3、M1–M3 使用完全同一参数 |
| 4 | 训练 B2-WB 与 M1 | 判断 learned whole-body position target 是否已经够用 |
| 5 | 实现连续样条/NIAF 与 NDP 强基线 | 排除连续曲线或已有动力系统即可解决 |
| 6 | 实现 M2 状态流、identity 和 partition loss | 一步/拆步离线误差越过预注册门槛 |
| 7 | 在 Arena Gate N 与目标 `10/15/20/30 Hz` 做配对闭环 | M2 优于 M1/B3/B4，且默认能力不回退 |
| 8 | 训练 M3 随机 schedule 与闭环时间分割损失，并重放真实 latency trace | 未见频率、jitter 和 deadline miss 下仍稳定 |
| 9 | 扩展对象、技能和 3 个训练 seed | 结论不是 checkpoint 或 seed 特例 |
| 10 | 从低速、低频、安全区域开始真机验证 | 无安全回归，仿真结论可复现 |

Arena 主线已经把旧 Gate N 与新 Gate N 分开：旧版示教速度诊断于 2026-08-15 归档在 [Gate N v0](arena-g1-gate-n.md)；新版 [Gate N frequency v1](arena-g1-gate-n-frequency.md) 已完成 120 条主对照闭环，另有 10 条同频审计，证明部署频率问题存在且三条简单基线没有完整解决。现阶段不再继续扩展旧 progress/clock 路线，下一项实际工作是 whole-body state target 与统一 tracker 的数据/执行闭环。

截至 2026-08-16，Phase 1 的机器契约、数据重建、SE(2) target、统一 tracker、训练、checkpoint 合并和 80 条正式闭环均已执行。M1 因默认能力从 matched LoRA 的 10/10 降至 0/10 而失败；跨频率配对又暴露出分叉前物理状态不一致，不能作干净因果解释。主线现在停在 Phase 1 故障分类，不进入顺序 5–10；允许的下一项工作仅是时钟域、target/statistics 与 tracker 接线的最小 ablation。

## 10. 与已有工作的关系，以及 novelty 在哪里

现有工作已经覆盖了许多组件，因此不能把“连续”“位置目标”或“神经 ODE”单独写成创新：

- [Isaac-GR00T 数据格式](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_preparation.md)、[data config](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_config.md) 和 [Policy API](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/policy.md) 已给出 LeRobot/modality、`delta_indices`、action horizon 与物理单位接口；[Arena G1 官方闭环教程](https://isaac-sim.github.io/IsaacLab-Arena/release/0.1.0/pages/example_workflows/locomanipulation/step_4_evaluation.html) 已给出 16-row feedback/WBC 评测链。因此本项目复用 `16×32` 不是新结构，贡献只能来自 target 语义、配对 benchmark 与实证。
- [DMP](https://pubmed.ncbi.nlm.nih.gov/23148415/) 和 [Neural Dynamic Policies](https://arxiv.org/abs/2012.02788) 已经把动力系统式动作表示用于机器人 Policy；[Autonomous Neural Dynamic Policies](https://arxiv.org/abs/2305.12886) 进一步研究了自主、稳定的端到端神经动力系统。自主流的半群数学性质本身不是本文发明。
- [AWE](https://arxiv.org/abs/2307.14326) 和 [HYDRA](https://openreview.net/forum?id=_A15qsPswaK) 已研究 waypoint 模仿学习，所以“速度改位置”不是创新。
- [Spline Policy](https://arxiv.org/abs/2606.07386)、[B-spline Policy](https://arxiv.org/abs/2607.09648) 与 [NIAF](https://arxiv.org/abs/2603.01766) 已能输出连续动作函数、以不同控制频率查询并提供解析导数。连续曲线本身也不是创新。
- [ABPolicy](https://arxiv.org/abs/2602.23901) 和 [ChunkFlow](https://arxiv.org/abs/2607.12992) 处理异步 replanning、chunk 接缝与跨 chunk 连续性。这些工作更接近时间原点/重叠一致性，不应被错误地说成不存在。
- [RTR 的官方开源实现](https://github.com/tars-robotics/RTR)、[LeRobot RTC 示例](https://github.com/huggingface/lerobot/blob/main/examples/rtc/eval_with_real_robot.py) 和 [PACE](https://arxiv.org/abs/2606.00537) 进一步说明高频 action、异步 chunk seam 与 execution horizon 本身都能改变结果。因此所有方法必须报告 plan age、边界 jump 和执行规则，不能把接口收益全部记到 action model 名下。
- [TempoVLA](https://arxiv.org/abs/2606.06491) 已研究显式速度条件；[ISR](https://arxiv.org/abs/2606.22907) 已研究轨迹重采样；它们分别约束时间条件和数据处理方面的 claim。
- [DreamTrajectory](https://arxiv.org/abs/2608.01381) 已使用轨迹引导移动操作的全身动作，因此“移动操作 + 轨迹”也不能单独构成 novelty。

当前真正可检验的 novelty hypothesis 是以下组合，而不是其中任意一个名词：

1. 把移动全身 VLA 的实际部署失败明确表述为**混合 velocity–position action 在随机重规划时间分割下的状态转移不一致**，并建立区分数据、推理和控制频率的配对 benchmark；
2. 让 flow-matching VLA 的 action expert 端到端生成作用于 `SE(2) × R^n` 的全身状态流，而不是在输出端只修补底盘；
3. 区分同一计划内部的半群组合律与重新观察后的闭环时间分割一致性，后者直接比较低频长步和高频中途重规划；
4. 分层验证内部 partition error、closed-partition error 和真实随机频率闭环结果，明确每个公式能覆盖什么、不能覆盖什么。

这套组合是否足够发表，仍取决于 M2/M3 能否显著打败 NIAF/spline、NDP 和 learned whole-body waypoint。最有价值的结果不是给已有连续轨迹换名字，而是证明：**只有把可组合的真实时间状态转移直接放进 whole-body VLA，才能在不牺牲默认能力的前提下抵抗实际算力造成的 `10–30 Hz` 调度变化。** 如果强基线也能做到，就必须缩小 novelty。

## 11. 论文最后可以说什么，不能说什么

如果 Gate、强基线和消融全部成立，论文才可以说：

- 我们量化了 mixed-action whole-body VLA 对重规划频率的敏感性，并把数据、推理和控制频率分开；
- 普通同预算微调、原 target 上追加真实 `delta_t`、以及 post-hoc SE(2) waypoint 没有同时守住默认能力与频率鲁棒性；
- 我们提出并实现了一个按真实 `Δt` 查询的全身状态转移 action expert；
- 同一计划的一步/拆步约束与重新观察后的闭环时间分割约束分别降低了对应误差，并且这些变化转化成未见频率和 jitter 下的闭环收益；
- 位置参考与同一状态流的导数前馈兼顾了抗积分漂移和动态跟踪性能；
- 理论保证只覆盖固定语义条件下的时间分割，随机重规划下的结果来自另外的闭环训练和实验验证。

论文不能说：

- 时间原点等变性本身保证推理频率鲁棒；
- 把一个 `delta_t` 数字输入原 Policy 就等价于学习连续时间动力系统；
- absolute position 对任何频率和任何新观测完全不敏感；
- velocity 乘真实 `Δt` 产生不同位移违反物理规律；
- waypoint、连续样条、Neural ODE 或半群数学本身是首次提出；
- 固定 latent 的组合律自动保证不同重规划频率看见新观测时动作完全相同；
- 一个任务、一个 checkpoint 或 10 seeds 证明了通用移动机器人频率鲁棒性。

## 附录 A：用公式精确定义核心对象

### A.1 全身状态和局部 SE(2) 目标

机器人全身状态写成：

\[
Q(t)=\big(T_b(t),q_a(t),g(t)\big),
\qquad Q\in\mathcal Q=SE(2)\times\mathbb R^n.
\]

其中 `T_b` 是底盘平面位姿，`q_a` 是躯干和双臂关节位置，`g` 是夹爪状态。训练 target 中的底盘位姿相对当前实测 frame 表示：

\[
\bar T_b(t+\Delta t)=T_b(t)^{-1}T_b(t+\Delta t).
\]

SE(2) 误差使用群对数而不是直接相减角度：

\[
d_{\mathcal Q}^2(Q_1,Q_2)=
\|\operatorname{Log}(T_{b,1}^{-1}T_{b,2})\|_{W_b}^2+
\|q_{a,1}-q_{a,2}\|_{W_q}^2+
\lambda_g\|g_1-g_2\|^2.
\]

### A.2 带辅助状态的自主状态流

为表达“先移动、再伸手”等仅靠瞬时 `Q` 不易区分的阶段，引入 action latent `h`，令增广状态 `X=(Q,h)`。视觉语言编码器产生 `z_t`，action expert 初始化 `h_t` 并定义：

\[
z_t=E_\theta(o_t,l,Q_t),\qquad
h_t=H_\theta(z_t,Q_t),\qquad
\dot X=F_\theta(X;z_t).
\]

固定 `z_t` 后，求解该自主系统得到流 `X_{t+\Delta t}=\Phi_{\theta,z_t}^{\Delta t}(X_t)`；物理 reference 是其 `Q` 分量。`h` 必须随拆步结果一起传递，不能在每个短步重置，否则组合律不成立。新 VLA 推理到达时才用最新观测重新生成 `z`、初始化新的 `h`，并把 `Q` 锚定到实测状态。

GR00T 原有 flow-matching action expert 可以改为生成 `h_t`、动力学参数或一组共享的状态流参数。随机 Policy 在比较一步/拆步时必须复用同一个生成噪声；否则测到的是采样差异，不是时间分割差异。

### A.3 半群性质为什么直接对应频率

状态流需要满足：

\[
\Phi^0(X)=X,
\qquad
\Phi^{t+s}(X)=\Phi^s\big(\Phi^t(X)\big).
\]

于是对任意时间分割 `\mathcal P=(\delta_1,\ldots,\delta_K)`，只要 `\sum_k\delta_k=T` 且 `z` 不变：

\[
\Phi^{\delta_K}\circ\cdots\circ\Phi^{\delta_1}(X)
=\Phi^T(X).
\]

所以同一 `100 ms` 被 10 Hz 调度看作一个长步，或被 30 Hz 调度切成三个短步，不会因为 action 的离散解释不同而改变终态。这一结论不覆盖中途重新编码新图像得到不同 `z` 的情况；那部分只能靠 M3 的随机闭环训练和实验验证。

### A.4 M1/M2 的监督目标

从真实 timestamp 构造多跨度监督集合；Arena 数据只有理想化 `50 Hz` timestamp，需把命令积分重建的 target 和未来真实 odometry 数据分开标记：

\[
\mathcal D_t=
\{(\Delta t_k,Q_{t+\Delta t_k})\}_{k=1}^{K},
\qquad \Delta t_k\in[1/30,1/10]\text{ s}.
\]

状态预测损失为：

\[
\mathcal L_{state}=
\mathbb E_{t,k}
d_{\mathcal Q}\!\left(
\Pi_Q\Phi_{\theta,z_t}^{\Delta t_k}(X_t),
Q_{t+\Delta t_k}
\right)^2.
\]

M2 另采样 `\delta_1,\delta_2>0`，并保持 `z_t`、初始 latent 和生成噪声相同：

\[
\mathcal L_{partition}=
d_{\mathcal X}\!\left(
\Phi^{\delta_1+\delta_2}(X_t),
\Phi^{\delta_2}(\Phi^{\delta_1}(X_t))
\right)^2,
\]

\[
\mathcal L_{identity}=d_{\mathcal X}(\Phi^0(X_t),X_t)^2.
\]

若选用数学上严格满足半群性质的求解器/结构，`L_partition` 主要用于控制数值误差并审计实现；若使用近似 transition decoder，它就是必要的学习约束。总损失还包含原有 flow-matching/BC、状态导数、平滑和安全正则：

\[
\mathcal L_{M2}=\mathcal L_{FM/BC}
+\lambda_s\mathcal L_{state}
+\lambda_p\mathcal L_{partition}
+\lambda_0\mathcal L_{identity}
+\lambda_v\mathcal L_{derivative}.
\]

### A.5 M3 的闭环目标

先在示教上采样重规划时刻 `t_0<t_1<\cdots`，令 `t_{i+1}-t_i` 覆盖 `1/30–1/10 s`；再以相同分布做仿真闭环 rollout。每次从实测 `Q_{t_i}` 和新 observation 生成新流。M3 优化的不是不同 schedule 下逐 action 相等，而是对 schedule 分布的期望任务/轨迹代价：

在没有新外部变化的配对片段上，闭环时间分割误差定义为：

\[
\mathcal L_{closed-partition}=m_{stable}\,
d_{\mathcal Q}\!\left(
\mathcal R_\theta(o_t,Q_t,\tau_1+\tau_2),
\mathcal R_\theta\!\left(
o_{t+\tau_1},
\mathcal R_\theta(o_t,Q_t,\tau_1),
\tau_2
\right)
\right)^2.
\]

`m_stable=1` 表示这段时间没有新的外部事件，可以比较低频一步与高频中途重规划；否则为 0，不强迫模型忽略新信息。这个 mask 只用于训练和评测配对，不是部署时的事件触发器。

若仿真器不可微，`L_closed-partition` 作为配对 rollout reward、数据筛选指标或专家纠正信号使用，而不是假装可以直接对整个 Isaac Sim 闭环反向传播。

\[
\mathcal L_{M3}=\mathcal L_{M2}+
\mathbb E_{\mathcal S\sim p(10\text{--}30\,\mathrm{Hz})}
\left[
\lambda_{task}\mathcal L_{task}^{\mathcal S}+
\lambda_{closed}\mathcal L_{closed-partition}^{\mathcal S}+
\lambda_\tau\mathcal L_{trajectory}^{\mathcal S}+
\lambda_{phase}\mathcal L_{base-arm}^{\mathcal S}+
\lambda_j\mathcal L_{command-jump}^{\mathcal S}+
\lambda_{safe}\mathcal L_{safety}^{\mathcal S}
\right].
\]

其中 `\mathcal S` 是一次随机重规划 schedule。轨迹监督来自同一示教的真实未来状态或仿真专家纠正；任务与安全项来自 Arena evaluator。若新 observation 显示障碍、接触或任务阶段发生变化，模型可以改变后续路线；这里惩罚的是物理命令跳变和任务退化，不是语义上合理的重新规划。

## 附录 B：GR00T 实现边界

第一版继续使用 Arena 官方任务调优 `GR00T_N1_5` checkpoint 作为 frozen B0。视觉语言 backbone 暂不修改，主要变化限定在 timestamp-aware dataset、action expert 和所有方法共用的 tracker。除非 Arena 官方提供兼容的新 adapter/checkpoint，或完成同环境严格迁移验证，否则不升级底座。

建议的数据字段：

```text
timestamp_ns                    # 真实采集时间；Arena 当前仅有理想化 50 Hz 时间
state.proprio
state.base_pose_se2             # 目标数据来自 odometry；Arena parquet 中缺失
state.base_twist
state.arm_qpos
state.arm_qvel
state.gripper
state.contact_or_risk

query.delta_t_s              # 以真实秒表示的状态转移时长
target.base_relative_se2     # odometry 真值，或明确标记的 command-integration proxy
target.arm_qpos
target.gripper
target.base_twist            # 导数/前馈监督，可选
target.arm_qvel              # 导数/前馈监督，可选

partition.delta_t_parts      # 总时长的一种随机拆分
partition.mask
schedule.replan_delta_t_s    # M3：闭环实测/采样的推理间隔
schedule.stable_pair_mask    # 只在无新外部变化时比较低频/高频闭环分割
```

代码应拆成以下独立模块：

```text
whole_body_state_builder     用 timestamp 对齐 SE(2)、关节和夹爪；Arena 用命令积分代理
multi_horizon_target_builder 从 30/50 Hz 数据生成覆盖 33–100 ms 的 target
se2_ops                      exp/log、composition、relative pose 和角度处理
state_flow_action_head       生成可按真实 delta_t 查询的全身状态流
partition_sampler            生成总时长相同的一步/拆步配对
semigroup_loss               计算 identity 和 partition consistency
closed_partition_loss        比较低频长步和高频中途重规划后的状态
whole_body_tracker           共用 position reference + derivative feedforward tracker
random_replan_scheduler      生成 10–30 Hz jitter 与 latency trace
temporal_benchmark           分别扰动数据、推理和控制频率
```

最低限度的自动化测试包括：

- SE(2) 平移、旋转、composition、inverse 和 `-π/π` 跨越正确；
- 30/50 Hz frame index 正确转换为秒，多跨度 target 不读取未来 observation 作为输入；
- Arena command-integration proxy 与 expert replay 的实测 root telemetry 误差通过预注册 Gate；
- `Δt=0` 严格返回当前增广状态；
- 一步 `100 ms` 与多种总和为 `100 ms` 的 partition 在解析测试系统上相同；
- 拆步时 action latent `h` 被传递而不是错误重置；
- 固定 observation、latent 和随机噪声后，partition test 可重复；
- `stable_pair_mask=0` 的动态变化片段不进入 closed-partition loss；
- 底盘和上肢 reference 使用同一个真实时间查询；
- 新推理到达时从实测 `Q_t` 重锚定，不从旧预测状态偷偷起步；
- position-only、derivative feedforward 和所有新方法使用相同 tracker 参数；
- telemetry 同时记录数据时间、inference 时间、reference 生效时间和 control 时间；
- train frequency、unseen frequency、jitter 和 deadline-miss manifest 在看结果前冻结。
