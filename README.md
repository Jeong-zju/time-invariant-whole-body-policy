# Semigroup-Consistent Whole-Body Policy

面向移动操作机器人的时间分割一致全身策略研究。项目当前聚焦一个具体部署问题：同一个 whole-body action chunk 同时包含底盘速度和上肢位置，若每个 action slot 的物理时刻是隐式的，改变重规划、动作消费或控制频率可能造成底盘积分漂移和 base–arm phase mismatch。

项目先用冻结策略判断频率问题是否真实存在，再比较普通同预算 LoRA、把真实重规划 `delta_t` 输入 Policy 的条件 LoRA，以及无需重训的相对 SE(2) waypoint tracker。三条简单路线都不足后，研究 Phase 1 先让 action expert 学习带真实时间刻度的 whole-body position 路线图；只有它仍不够，Phase 2/3 才依次增加时间分割一致性和随机重规划训练。项目不假设真实物理对任意速度不变。

## 研究目标

- 分开量化数据、重规划、动作消费和底层控制频率的影响；
- 检查混合 velocity–position action 在不同频率下是否保持相同物理时间语义；
- 比较原始 Policy、普通同预算微调、真实 `delta_t` 条件 Policy 和相对 SE(2) waypoint tracker；
- 记录底盘轨迹、base–arm 同步、任务成功和真实推理延迟；
- 只有简单执行接口仍不能解决频率问题时，才考虑修改 VLA action expert。

主开发基准已冻结为 NVIDIA Isaac Lab Arena 的 **G1 Loco-Manipulation — Box Pick-and-Place**。它原生提供 G1 全身移动操作环境、demonstrations、GR00T adapter 和任务调优 checkpoint；官方冻结 GR00T N1.5 已在远端完成 50-seed 闭环，成功率为 `44/50`。MS-HAB 与 BEHAVIOR 结果保留为 legacy 诊断。当前部署状态见 [Arena G1 BM-0 部署记录](docs/arena-g1-bm0-deployment.md)，重新定义的 Gate N 见 [频率一致性测试](docs/arena-g1-gate-n-frequency.md)，旧 Gate 结果见 [归档记录](docs/arena-g1-gate-n.md)，技术路线见 [研究方案](docs/time-invariant-whole-body-policy.md)。

## 仓库结构

```text
configs/       可版本化的训练与评测配置
docs/          研究设计、决策记录和部署记录
experiments/   实验清单、结果索引和分析脚本
paper/         论文大纲、图表与 LaTeX 源码
src/           数据表示、模型适配和控制器实现
tests/         单元测试与不变性回归测试
```

原始数据、模型权重和运行产物不得提交到 Git。每项可报告实验必须记录配置、随机种子、代码 commit、数据 revision 和评测环境。

## 路线图

准备和证伪使用 Gate，只有模型或目标函数升级使用 Phase；每个 Phase 只新增一个可消融变量：

1. Legacy Gate -1/0：保留已完成的 BEHAVIOR 链路、telemetry 和 B0 诊断结果。
2. BM-1/BM-0：部署 Arena G1，复现官方 GR00T 闭环，并在相同 demonstrations 上建立固定时间 GR00T B0。
3. Gate N v0：示教速度/label ambiguity 离线诊断未过门槛，结论归档。
4. Gate N frequency v1：固定 checkpoint 与 50 Hz control，比较 3.125/6.25/12.5 Hz 闭环重规划。
5. Gate N-A：原始 Policy 的跨频率成功率与轨迹变化已通过实际问题准入；同频重复继续限制因果解释。
6. Gate N-B：普通同预算 LoRA、真实 `delta_t` Policy 和 SE(2) waypoint 均未通过冻结门槛，允许进入新移动 VLA 策略研究。
7. Phase 1，M1：实现多跨度 whole-body position target、带时间刻度的短路线图与统一 tracker，并与 Gate N-B 强基线比较。
8. Phase 2，M1→M2：增加模型内 state flow、identity 与时间分割一致性。
9. Phase 3，M2→M3：增加随机 `10–30 Hz` 重规划、实测状态重锚定与闭环时间分割训练。
10. Gate S/R：冻结方法后回到 BEHAVIOR 多任务仿真并做真机验收。

## 协作

稳定分支为 `main`，不直接在其上开发。工作分支统一使用：

```text
<type>/<owner>/<topic>
```

例如 `exp/ranpeng/time-warp-ablation`、`feat/jeong/progress-sampler`、`paper/ranpeng/method-section`。完整类型、负责人别名和 PR 规则见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 当前状态

Arena G1 官方冻结 GR00T N1.5 已完成 50-seed、每条最多 1,200 step 的闭环评测，结果为 `44/50` 成功（Wilson 95% CI `[0.762, 0.944]`）。100 episodes / 94,936 frames 的 2-epoch action-head LoRA 也已训练到 `47,468` step；seed 0 闭环为 `1/1`，该单 episode 只证明训练链路没有退化，不作为成功率估计。详见 [Arena G1 BM-0 部署记录](docs/arena-g1-bm0-deployment.md)。

Gate N v0 离线诊断已经否定“示教速度造成主要 label ambiguity”的主张，并作为历史结果保留。重新定义的 Gate N-A 已用原始 Policy 的 30 条跨频率闭环确认实际问题：成功率从 `8/10` 降到 `5/10` 和 `4/10`，XY RMS 为 `0.543/0.632 m`。同频复现的 XY 中位噪声 `0.123 m` 仍限制严格因果归属，但不再把部署退化判成不存在。Gate N-B 又完成 90 条闭环：普通 LoRA 为 `10/2/1`，真实 `delta_t` Policy 为 `9/1/4`，SE(2) waypoint 为 `5/6/8`；三者都未同时守住默认能力和跨频率门槛。Gate N 最终通过，允许进入新移动 VLA 策略研究。逐步协议与人话解释见 [Gate N frequency v1](docs/arena-g1-gate-n-frequency.md)，完整技术结论见 [研究方案](docs/time-invariant-whole-body-policy.md)，旧结果见 [Gate N v0](docs/arena-g1-gate-n.md)。

研究 Phase 1 的本地实现已建立：保持 GR00T `16×32` 输出外形，把底盘后三维改为带真实时间刻度的相对 SE(2) 位置，配套多跨度 target、wall-clock 路线查询、统一 tracker、数据代理 gate 与单元测试。机器契约见 [M1 配置](configs/arena-g1-phase-1-m1-v1.yaml)，给远端执行 AI 的逐项规格见 [Phase 1 AI 实施文稿](docs/phase-1-ai-implementation.md)。正式训练仍由 expert replay root-odometry 数据 Gate 阻挡；不能把“代码就绪”写成“实验已经完成”。

此前 BEHAVIOR B0 的 0/10 和 MS-HAB BM-0 的单轨迹失败均保留用于追溯；MS-HAB 正式 904-episode 训练没有启动，不再是当前主线。
