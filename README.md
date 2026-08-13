# Time-Invariant Whole-Body Policy

面向移动操作机器人的时间重参数化不变全身策略研究。项目聚焦一个具体矛盾：底盘通常输出速度，而机械臂通常输出位置，导致全身动作表示与采集频率、推理频率和控制频率耦合。

本项目尝试把底盘与机械臂统一表示为相对当前机器人坐标系的几何路径，并将“怎么走”与“多快走”解耦。这里的“不变”特指**时间重参数化不变性**，不是经典 LTI 系统意义下的时不变，也不意味着机器人动力学与执行速度无关。

## 研究目标

- 构造局部 SE(2) 底盘位姿与机械臂关节位置组成的全身几何路径；
- 按运动进度而非固定时间索引采样动作序列；
- 验证空间原点不变性、time-warp 不变性和频率泛化；
- 在动态约束下通过独立时间标定器安全执行预测路径；
- 形成可复现实验、消融研究和最终论文。

主基准暂定为 BEHAVIOR 2026，第一阶段使用 `turning_on_radio`、`bringing_water` 和 `sweeping_garage` 三个任务。详细技术路线与实验设计见 [研究方案](docs/time-invariant-whole-body-policy.md)，当前基础设施状态见 [Phase -1 部署记录](docs/phase-minus-1-deployment.md)。

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

1. 完成官方 baseline 的闭环准入与指标采集。
2. 建立原始 velocity action 与局部 SE(2) 路径 baseline。
3. 实现 progress-based sampler 和 time-warp/frequency benchmark。
4. 实现带动态约束的在线时间标定器。
5. 完成多任务、多频率、多 time warp 的主实验与消融。
6. 冻结实验协议，整理图表并撰写论文。

## 协作

稳定分支为 `main`，不直接在其上开发。工作分支统一使用：

```text
<type>/<owner>/<topic>
```

例如 `exp/ranpeng/time-warp-ablation`、`feat/jeong/progress-sampler`、`paper/ranpeng/method-section`。完整类型、负责人别名和 PR 规则见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 当前状态

Phase -1 的数据、checkpoint 和 GPU 环境检查已完成；闭环 rollout 仍依赖 gated 模型授权与仿真许可确认。尚未进入正式模型实现阶段。
