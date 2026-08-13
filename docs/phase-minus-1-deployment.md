# Phase -1 远程部署记录

更新日期：2026-08-13

## 结论

新实例已经从官方源完成全量部署，并通过实际闭环运行验收。它不是从旧实例复制出来的环境：BEHAVIOR-1K、Isaac-GR00T、数据集、Isaac Sim 资产和 checkpoint 都在新实例上重新安装或下载。

新实例使用 RTX 5090 和 NVIDIA 580.126.09。Isaac Sim 5.1 在原生驱动下成功进入 `[app ready]`、载入 `turning_on_radio` public instance 301、初始化 R1Pro 与相机，并分别完成：

- local zero-action policy 的 1-step 仿真 smoke，结果 JSON 正常写出；
- 官方 GR00T checkpoint 的 WebSocket 闭环 smoke，连续执行 51 个 simulator steps，结果 JSON 和 51 帧 H.264 视频正常写出。

因此旧实例遇到的 595.84 驱动崩溃没有在新实例复现。Phase -1 的“数据—模型—WebSocket—仿真—动作—结果”准入链路已通过，可以在新实例继续 Phase 0。完整的时间戳、Policy 延迟、底盘和关节轨迹记录器仍应作为 Phase 0 的第一个工程任务补齐；它不再是基础设施阻塞项。

## 远程入口与存储

```text
SSH         ssh -p 51384 root@47.186.21.5
Jupyter     https://127.0.0.1:8083
远程根目录  /workspace/time-invariant-whole-body-policy
```

用户原命令中的本地 `8080` 已被本机其他服务占用，因此当前隧道使用本地 `8083`：

```bash
ssh -N -p 51384 -L 8083:localhost:8080 root@47.186.21.5
```

Jupyter 使用自签名 TLS，浏览器可能需要手动接受证书。该实例的 `/workspace` **不是持久卷**；停止后重启通常保留，但销毁或回收会丢失数据。开始训练前应挂载持久卷或定期同步 checkpoint、配置与实验结果。

## 锁定的软件与硬件

| 项目 | 版本或路径 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090，32 GB，compute capability 12.0 |
| Host driver / system toolkit | `580.126.09` / CUDA `13.0` |
| GR00T PyTorch | `2.7.1+cu128`，GPU 矩阵乘通过 |
| BEHAVIOR PyTorch | `2.7.0+cu128`，GPU 矩阵乘通过 |
| OmniGibson / Isaac Sim | `3.9.1` / `5.1.0.0` |
| BEHAVIOR-1K | tag `v3.9.1`，commit `26f2c7ef7b9cf96bd0414f81e1e751e493762779` |
| Isaac-GR00T | branch `behavior`，commit `ace36d935b376fbf25cd56371e23877b95407c40` |
| 数据集 | `behavior-1k/2026-challenge-demos` |
| 数据 revision | `4f50b44796641a4d526a19d9aeadc8aa51e2f2c2` |
| BEHAVIOR env | `/workspace/miniconda3/envs/behavior` |
| GR00T env | `/workspace/time-invariant-whole-body-policy/repos/Isaac-GR00T/.venv` |

系统 Toolkit 是 CUDA 13.0，但两个项目环境都使用官方 CUDA 12.8 PyTorch wheel。驱动向后兼容这套用户态运行时，无需且不应在 Vast 容器内更换宿主机驱动。

## 数据、授权与 checkpoint

已下载 `turning_on_radio`、`bringing_water`、`sweeping_garage` 三个任务，共 600 条示范。数据目录包含 204 个非缓存文件和 181 个 MP4；所有视频均通过 `ffprobe`，无损坏或零字节文件。三项低维校验均通过：`fps=30`、state 61 维、action 23 维、timestamp 单调。

| task | episodes | rows | timestamp 最大误差 | 可视化 episode | 时长 | 积分终点 `[x,y,yaw]` |
|---|---:|---:|---:|---:|---:|---|
| `turning_on_radio` / 0 | 200 | 429,928 | `8.14e-6 s` | 0 | 65.17 s | `[0.516,-0.549,0.649]` |
| `bringing_water` / 17 | 200 | 1,887,709 | `2.24e-5 s` | 3400 | 238.97 s | `[0.919,2.223,-1.182]` |
| `sweeping_garage` / 57 | 200 | 894,606 | `2.24e-5 s` | 11400 | 125.63 s | `[3.504,-0.484,0.545]` |

固定种子 `20260813` 的 episode split 已生成，每项任务 180 train / 20 validation。验证报告、轨迹图与 split 位于：

```text
/workspace/time-invariant-whole-body-policy/artifacts/validation/dataset
```

Hugging Face gated access 已通过真实下载 `nvidia/Cosmos-Reason2-2B/config.json` 验证。官方 `turning_on_radio` GR00T checkpoint 也通过 ZIP 完整性和 SHA256 校验：

```text
压缩包  /workspace/time-invariant-whole-body-policy/checkpoints/turning_on_radio_GR00T-checkpoint-150000.zip
大小    6,917,651,266 bytes
SHA256  74eab585f4cf0f92326c9d4d74732343f6d789b05e945d3e71afe46bac69cb46
模型目录 /workspace/time-invariant-whole-body-policy/checkpoints/turning_on_radio_GR00T-checkpoint-150000
```

访问凭据只保存在服务器的 Hugging Face credential store，没有写入仓库或本文档。

## 推理服务与闭环验收

GR00T Policy Server 由 Supervisor 管理，绑定 loopback，不对公网暴露：

```text
manager  supervisor:groot_policy_server
endpoint http://127.0.0.1:8000/healthz -> OK
status   RUNNING
GPU RAM  约 6.8 GB
log      /workspace/time-invariant-whole-body-policy/logs/groot-policy-server-supervisor.log
```

原生 local-policy smoke 的结果：

```text
instance  301
steps     2（max-steps=1 时 evaluator 的计数语义）
exit      0
result    /workspace/time-invariant-whole-body-policy/artifacts/validation/simulator-smoke/json/turning_on_radio_301_0.json
log       /workspace/time-invariant-whole-body-policy/logs/behavior-simulator-smoke.log
```

GR00T WebSocket 闭环 smoke 的结果：

```text
instance  301
steps     51（max-steps=50 时 evaluator 的计数语义）
success   false；只做链路验收，不以 50 步任务成功为目标
q_score   0.0
exit      0
JSON      /workspace/time-invariant-whole-body-policy/artifacts/validation/closed-loop-smoke/json/turning_on_radio_301_0.json
video     /workspace/time-invariant-whole-body-policy/artifacts/validation/closed-loop-smoke/videos/turning_on_radio_301_0.mp4
video     H.264，672×448，30 fps，51 frames，1.7 s
log       /workspace/time-invariant-whole-body-policy/logs/behavior-closed-loop-smoke.log
```

截至验收完成时磁盘使用约 135/256 GB，剩余约 122 GB；足够进行 Phase 0 的单任务 baseline，但训练产生大量 checkpoint 前仍需配置清理和外部备份策略。

## 与旧实例故障的关系

旧实例的 595.84 驱动会让 Isaac Sim 5.1 在 `[app ready]` 后崩溃。新实例的 580.126.09 属于 R580 系列，且高于 NVIDIA 对 Isaac Sim 5.1 列出的 Linux 测试驱动 580.65.06；实际 local 与 WebSocket 两条运行结果进一步证明该组合可用。

参考：

- [Isaac Sim 5.1 system requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
- [NVIDIA Isaac Sim issue #568](https://github.com/isaac-sim/IsaacSim/issues/568)
- [NVIDIA Isaac Sim discussion #648](https://github.com/isaac-sim/IsaacSim/discussions/648)

旧实例上的 `isaacsim_rtx_compat.py` 只保留为历史诊断工具；新实例没有加载或使用该 workaround。

## Phase 0 起点

基础设施已经足够进入 Phase 0，执行顺序固定为：

1. 在官方 evaluator 外增加逐步 telemetry：观测时间戳、动作请求/返回时间、deadline miss、底盘 state/action 和关节 state/action。
2. 用相同任务、instance、seed 和 evaluator 配置复现官方 velocity-action B0。
3. 锁定 B0 的成功率、q-score、time、distance、延迟和轨迹日志，再开始 B1/B2 表示改造。

不要用本次 50-step smoke 的 `success=false` 评估 checkpoint 的任务能力；它只证明部署闭环能够正确运行。
