# Arena G1 Gate N 探究记录

更新日期：2026-08-15

> 历史状态：这是 Gate N v0 的“示教速度/label ambiguity”诊断。该问题已被否定并归档，不再是项目主 Gate。重新定义后的主问题与新闭环测试见 [Gate N frequency v1](arena-g1-gate-n-frequency.md)。

这份文档只记录 Gate N：先确认“示教快慢造成训练标签冲突”是否真实存在，再判断 `Δt`、waypoint、进度采样、ISR 和 spline 等简单方案是否已经足够。它不是新方法的结果页，也不把离线数据整理指标冒充闭环成功率。

## 当前一句话结论

**Gate N 离线阶段未通过，按预注册规则停止 B1–B5 训练。** Arena 的 100 条示教确实有轻微、方向一致的时间标签效应，但进度对齐只把 speed-contrast pair 的未来标签中位冲突降低 `8.0%`，没有达到预注册的 `20%`；可靠 speed-contrast pair 也只有 `92` 个，没有达到 `200` 个。与此同时，whole-body progress 和预算匹配 ISR 都漏掉了超过容许范围的转向或底盘移动片段。

这不是说时间表示永远没用，而是说当前这个单任务、固定 50 Hz、同质化很强的 Arena 数据不能支撑“时间标签歧义是主要瓶颈”这一主张。现在继续训练五个模型，得到的差异很可能主要来自输出接口和数据删减，而不是我们想研究的问题。

## 第 0 步：连接和资源审计

### 做了什么

使用用户给出的远程地址连接：

```bash
ssh -p 51384 -L 8080:localhost:8080 root@47.186.21.5
```

连接本身成功，并完整阅读了远端 `/etc/vast-agents-guide.md`。随后检查磁盘、内存、GPU、Supervisor、训练日志、checkpoint 和 Gate 0 产物。

### 看到了什么

- 本机 `8080` 已被其他进程占用，所以 SSH 会话成功，但 `-L 8080:...` 转发没有建立；需要 Jupyter 时应改用本地 `8083`。
- 远端是 RTX 5090 32 GB；审计时 GPU 空闲。
- 根文件系统 256 GB，已使用约 250 GB，只剩约 7 GB。
- `/workspace` 不是持久卷；实例 recycle 或 destroy 会丢失全部实验产物。
- Arena 的 2-epoch action-head LoRA 已完整训练到 `47,468` step，训练时长约 3.05 小时，最终训练 loss 为 `0.00405`。
- 该 LoRA 在 seed 0 闭环仍是 `1/1` 成功。
- 冻结官方 GR00T N1.5 的 Gate 0 正式结果是 `44/50 = 88.0%`，Wilson 95% CI 为 `[76.2%, 94.4%]`。

### 用人话说，这说明什么

远端现在不是“环境还没装好”，而是已经有可信 B0 和可训练链路，可以开始 Gate N。真正的工程风险是磁盘和持久性：不能粗暴复制五份完整模型，也不能把唯一结果只留在远端。后续优先保存很小的 LoRA adapter、JSON、CSV 和图；长任务必须交给 Supervisor。

## 第 1 步：确认数据里到底有哪些时间和状态信息

### 做了什么

直接读取冻结 revision 的 100 个 LeRobot parquet、`info.json`、`modality.json`，并检查一个完整 episode 的每个时间戳和字段。

### 看到了什么

| 项目 | 结果 |
|---|---|
| Episodes | 100 |
| Frames | 94,936 |
| 标称频率 | 50 Hz |
| 所有相邻时间戳 | **全部为 0.02 s** |
| 单条长度 | 大约 880–1,418 frames |
| 原始机器人 state/action | 43 维关节位置 |
| GR00T 实际输入 state | 双臂、双手、腰，共 31 维 |
| GR00T 实际输出 action | 双臂、双手、base height、`vx/vy/yaw_rate`，共 32 维 |
| 额外可用数据 | 双腕 observation/action pose、导航命令、base height、RGB |
| 缺失数据 | 实测 root pose、物体 pose、接触状态、接触力 |

parquet 的 `timestamp` 是数据转换器用 `np.arange(length) / fps` 重新生成的，不是带采集抖动的原始墙钟。不同示教有不同总长度，但每一帧都挂在同一只理想 50 Hz 时钟上。

### 用人话说，这说明什么

数据里有“有人做得快、有人做得慢”的可能性，但没有“这一帧实际隔了多久”的自然变化。模型无法从原始 `Δt` 区分快慢，因为每帧看到的都是 0.02 秒。

这也限制了我们能证明什么：可以用双腕轨迹、关节动作和导航命令检查转向、停顿和精细运动是否被删掉，但不能仅凭 parquet 宣称“接触事件被完整保留”。接触保留最终必须靠 simulator replay 或补录 contact telemetry 验证。

## 第 2 步：把强基线的原义钉死

### 做了什么

核对了四篇原论文的方法部分，而不是只按论文名猜实现：

- [TempoVLA](https://arxiv.org/abs/2606.06491)
- [ISR](https://arxiv.org/abs/2606.22907)
- [Spline Policy](https://arxiv.org/abs/2606.07386)
- [B-spline Policy](https://arxiv.org/abs/2607.09648)

### 看到了什么

- TempoVLA 不是“追加一个 `Δt` 数字”这么简单。它把目标速度作为显式条件，并用 VSTA 对动作做可逆的合并/拆分；分段时保留方向反转和 gripper 事件边界。
- ISR 原论文只对单个 3D 末端位置做离线重采样。它用速度项保留空间间距，用加速度项保留转弯和接触前减速；默认 `D_target=0.05`、`lambda_vel=1.0`，接触丰富任务的 `lambda_acc=0.01`。论文自己也把“尚未扩到 joint space”列为限制。
- Spline Policy 真正改变的是模型输出对象：16 个离散 action 被更少的 spline 参数替代；论文的 matched comparison 使用 8 个 spline 参数对 16-step chunk。
- B-spline Policy 使用三次 B-spline，自适应拟合 knot，并让模型同时预测 knot 和 control point；它不是在模型输出后简单套一个平滑滤波器。

### 用人话说，这说明什么

如果我们只是把现有 16-step action chunk 平滑一下，就不能把结果叫作 B5。公平 B5 必须重新制作 spline 监督并重新训练输出头。类似地，原版 ISR 对固定底座机械臂合理，但 Arena G1 会一边走一边操作；只看机器人坐标系里的手腕，很可能把“手不动、整个人在走”的片段误当成低信息。Gate N 必须同时报告：

1. 按论文原义把左右腕位置拼成 6D 的 ISR；
2. 面向 whole-body 的扩展版本；
3. 两者是否漏掉底盘转向和移动片段。

扩展版本只能叫项目适配，不能伪装成论文原版 ISR。

## 第 3 步：离线问题诊断

### 已冻结的做法

代码：[`scripts/gate_n/analyze_arena_gate_n.py`](../scripts/gate_n/analyze_arena_gate_n.py)

机器可读预注册：[`configs/arena-g1-gate-n-v0.yaml`](../configs/arena-g1-gate-n-v0.yaml)

远端产物目录：

```text
/workspace/time-invariant-whole-body-policy/artifacts/validation/arena-g1-gate-n/offline-diagnostics-v0/
```

分析分成两层：

1. 每隔 5 帧取一个候选锚点，用机器人当前状态、积分得到的底盘位姿代理和双腕 pose 找跨 episode 近邻；再用 `16×12` 的低分辨率 RGB 描述子排除明显不是同一场景的帧。
2. 对“当前状态相似、后续运动方向相同、瞬时速度至少相差 1.5 倍”的 pair，比较固定 0.32 秒标签、固定 whole-body 运动进度标签和 ISR 信息坐标标签的未来不一致度。

采样整理同时比较：

- 固定每 3 帧取 1 帧；
- 在相同样本数下按 whole-body 运动距离均匀取点；
- 论文默认参数 ISR；
- 与 1/3 样本预算匹配的 ISR。

除重建误差外，还分别检查高双腕加速度、底盘转向和底盘平移附近是否仍有采样点。这里的“高双腕加速度”明确只是接触/精细操作代理，不是真接触标签。

### 实际结果

远端 Supervisor 程序 `arena_g1_gate_n_analysis` 已正常完成，无 stderr。纯函数有 5 个单元测试，覆盖底盘积分、端点保留、静止轨迹、进度插值的速度不变性和 ISR 索引单调性；独立 gate evaluator 另有 1 个停止条件测试。

| 指标 | 结果 | 预注册门槛 | 判定 |
|---|---:|---:|---|
| 高相似跨 episode pair | 3,000 | ≥1,000 | 通过 |
| 同方向且速度差 ≥1.5× 的 pair | **92** | ≥200 | **失败** |
| speed-contrast pair 的进度对齐中位改善 | **7.95%** | ≥20% | **失败** |
| 进度对齐优于时间对齐的 pair 比例 | 81.5% | ≥60% | 通过 |

在全部 3,000 个 matched pair 上，进度对齐的中位改善为 `8.23%`，bootstrap 95% 区间为 `[7.58%, 8.82%]`。在预注册的 92 个 speed-contrast pair 上是 `7.95%`，bootstrap 95% 区间为 `[2.90%, 10.71%]`。这说明改善方向很稳定，但量级小，不是一个被少量噪声遮住的 20% 大效应。

作为只用于排错、不改变正式判定的事后敏感性分析，我们把速度比和方向余弦阈值放宽或收紧。除一个只有 13 个 pair 的极小子集外，样本数为 27–377 的各组中位改善都在约 `4.7%–8.5%`。所以失败不是某一个阈值刚好卡住造成的。

自动生成的 12 组 top-pair 图也做了人工检查：箱子位置、双手构型和相机视角总体对齐，没有发现“把抓取前和放置后当成同一状态”这类成批错配。匹配器仍不是语义模型，所以这个检查只能排除明显错误，不能把 RGB 小图距离升级成严格场景等价证明。

### 这一步的判定方式

- 少于 1,000 个可靠 matched pair，或少于 200 个同方向 speed-contrast pair，说明这个单任务数据不足以支撑 label ambiguity 主张；不能靠调阈值制造问题。
- 按进度后未来标签中位不一致度至少降低 20%，并且至少 60% pair 比按时间更一致，才把核心问题记为通过。
- 如果 ISR 降低不一致度却明显漏掉转向/移动/高加速度片段，说明它只是在更激进地删数据，不能视为问题已解决。
- 整理方法的关键事件 coverage 不得比 3× 固定时间采样低超过 5%，几何重建 RMS 不得高于它的 1.25 倍。
- 只有离线问题成立且关键片段没有被整理方法破坏，才值得支付 B1–B5 的训练和 50-seed 闭环成本。

### 用人话说，这一步说明什么

“按进度看未来”大多数时候确实比“死等 0.32 秒看未来”更一致，但只好了一点点。当前数据更像是 100 条节奏相近、流程统一的示教，而不是不同操作者忽快忽慢造成的大量互相矛盾标签。它可以作为小的数据标准化效应继续观察，却不足以支撑重做 action expert 的主要研究动机。

## 第 4 步：比较 time、whole-body progress 和 ISR

### 实际结果

以下均为 100 个 episode 的中位数。`coverage` 表示关键帧前后 1 帧内至少保留一个采样点。

| 整理方式 | 保留比例 | 几何重建 RMS | 运动步长 CV | 高加速度 coverage | 转向 coverage | 底盘移动 coverage |
|---|---:|---:|---:|---:|---:|---:|
| 固定时间 3× | 33.44% | 0.01896 | 0.328 | 100% | 100% | 100% |
| Whole-body progress | 33.22% | **0.01778** | **0.190** | 100% | **88.67%** | 99.18% |
| ISR 论文默认参数 | **100%** | 0 | 0.437 | 100% | 100% | 100% |
| ISR 匹配 1/3 预算 | 33.33% | 0.01813 | 0.262 | 100% | **92.22%** | **91.81%** |

Whole-body progress 的好处是真的：在相同样本数下，几何重建略好，运动步长不均匀程度从 `0.328` 降到 `0.190`。但它漏掉了约 `11.3%` 的转向关键帧，超过预注册允许的 5%。这说明当前进度距离里 `0.5 m/rad` 的 yaw 权重仍不足以保护转向；现在可以把它列为下一版设计问题，但不能看完结果后调大权重再宣布本 gate 通过。

ISR 暴露了更根本的迁移问题。论文默认 `D_target=0.05` 在这批 50 Hz 仿真双腕数据上保留了每一帧，等于完全没有重采样。要把中位保留率压到 1/3，数据集全局 `D_target` 必须调到 `1.326`，约为论文默认值的 26.5 倍。预算匹配后重建误差不错，但转向和底盘移动 coverage 分别下降约 `7.8%` 和 `8.2%`，仍未过 gate。

### 用人话说，这一步说明什么

- 固定时间采样很笨，但在规则的 50 Hz 数据里相当稳，不会偏心某种动作。
- Whole-body progress 更会“把点花在有运动的地方”，却还不够懂转弯的重要性。
- 原版 ISR 只看手腕，对会走路的 G1 不够；直接搬论文参数时一个点也删不掉，强行调到同样预算后又会忽略身体在移动和转向。

因此目前没有一种进度/ISR 整理方式同时满足“更一致”和“不伤关键动作”。N2 失败。

## Gate N 正式判定

机器判定文件：

```text
/workspace/time-invariant-whole-body-policy/artifacts/validation/arena-g1-gate-n/offline-diagnostics-v0/gate-evaluation.json
```

远端 `/workspace` 没有持久卷，所以关键证据也已同步到本地：

- [完整离线报告](../artifacts/validation/arena-g1-gate-n/offline-diagnostics-v0/report.json)
- [机器门槛判定](../artifacts/validation/arena-g1-gate-n/offline-diagnostics-v0/gate-evaluation.json)
- [speed-contrast pair 预览图](../artifacts/validation/arena-g1-gate-n/offline-diagnostics-v0/speed-contrast-pairs-preview.jpg)
- [匹配 pair 明细](../artifacts/validation/arena-g1-gate-n/offline-diagnostics-v0/matched-pairs.csv)
- [逐 episode 重采样明细](../artifacts/validation/arena-g1-gate-n/offline-diagnostics-v0/resampling-per-episode.csv)

这些验证产物受 `.gitignore` 管理，不会把大批实验输出误提交到 Git，但当前工作区保留了可复核副本。

判定为：

```text
offline_gate_passed: false
decision: stop_model_work_and_reassess_dataset_or_claim
```

失败原因有四条：

1. speed-contrast pair 数量 `92 < 200`；
2. 进度对齐中位改善 `7.95% < 20%`；
3. whole-body progress 的转向 coverage 降幅 `11.33% > 5%`；
4. 预算匹配 ISR 的转向/底盘移动 coverage 降幅分别为 `7.78% / 8.19% > 5%`。

所以本轮不启动 B1–B5 训练，也不启动 waypoint tracker replay。这是 Gate N 的正常停止条件，不是工程没做完。

如果还要继续这条研究路线，下一份数据至少应提供：多操作者或受控的快慢示教、真实采集时间戳、实测 root/object pose、接触 telemetry，以及对快慢重放是否保持任务结果的 simulator 验证。否则最多只能研究“合成速度控制”，不能声称解决了自然示教里的主要 label ambiguity。

## 后续 Gate N 队列

| 顺序 | 工作 | 当前状态 | 进入下一步的条件 |
|---|---|---|---|
| N0 | 数据与论文语义审计 | 完成 | 已确认可测边界 |
| N1 | 相似状态 label ambiguity | **失败** | pair 数和效应量均未过门槛 |
| N2 | time / progress / ISR 整理比较 | **失败** | progress/ISR 均漏掉过多关键动作 |
| N3 | B2 waypoint 与统一 tracker replay | 按 gate 停止 | 只有更换数据或主张后才重开 |
| N4 | 公平训练 B0–B5 | 按 gate 停止 | 不为未成立的问题支付训练预算 |
| N5 | 相同 seed 闭环比较 | 按 gate 停止 | N4 未进入 |

当前不能跳过 N1/N2 直接宣称要训练五个模型。Gate N 的价值正是允许便宜、可证伪的结果提前叫停不值得的路线。
