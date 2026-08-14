# Arena G1 Box Pick-and-Place：Codex 安装、部署与运行手册

文档版本：1
冻结日期：2026-08-14
协议 ID：`arena-g1-box-pick-place-v0`
主要读者：Codex 或其他负责远程部署、训练和评测的编程 Agent

## 1. 本文的角色

本文是可执行 runbook，不是结果宣传页。Agent 必须按照这里的版本、路径、gate 和失败条件部署：

**NVIDIA Isaac Lab Arena G1 Loco-Manipulation — Box Pick-and-Place**。

机器可读协议是 [`configs/arena-g1-box-pick-place-v0.yaml`](../configs/arena-g1-box-pick-place-v0.yaml)。若本文与 YAML 冲突，以 YAML 中的冻结版本和 action contract 为准；如果上游接口变化导致两者无法同时满足，停止执行并报告，不得静默升级。

本协议的目标是建立一个原生支持 GR00T、能够获得非零闭环成功率的移动操作基线。它不证明新方法有效，也不允许用单 episode 成功率声称统计显著性。

需要把任务交给另一个 Codex 实例时，使用配套的 [`docs/arena-g1-codex-prompt.md`](arena-g1-codex-prompt.md)。

## 2. 不可修改的 benchmark contract

### 2.1 上游版本

| 组件 | 冻结值 |
|---|---|
| Isaac Lab Arena | `f479817431f6a02a6e40e4ea76d89203b59146d4`，release 0.1.1 |
| Isaac Lab | `3c6e67bb5c7ada942a6d1884ab69338f57596f77`，v2.3.0 |
| Arena 内 Isaac-GR00T | `3bce5530b3af0ce3619d5fda041385f9b732dca8` |
| Isaac Sim | 5.1.0 |
| Dataset | `nvidia/Arena-G1-Loco-Manipulation-Task` @ `97f7b5a4135e4e6a206d394c9b6ec0253f1558a7` |
| Official checkpoint | `nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation` @ `629479fedb1cf97c2f11ddc49eed951c5b750139` |

官方 checkpoint 的 `config.json` 架构是 `GR00T_N1_5`。Agent 必须把它写成 **GR00T N1.5**；不得因为文件名含 `GN1x`，或用户允许使用 N1.6，就将它标成 N1.6/N1.7。

上游资料：

- [Isaac Lab Arena release 0.1.1](https://isaac-sim.github.io/IsaacLab-Arena/release/0.1.1/index.html)
- [G1 loco-manipulation workflow](https://isaac-sim.github.io/IsaacLab-Arena/release/0.1.1/example_workflows/locomanipulation/index.html)
- [官方数据集](https://huggingface.co/datasets/nvidia/Arena-G1-Loco-Manipulation-Task)
- [官方任务 checkpoint](https://huggingface.co/nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation)

### 2.2 环境与任务

```text
environment id  galileo_g1_locomanip_pick_and_place
object          brown_box
embodiment      g1_wbc_joint（策略闭环）
replay body     g1_wbc_pink（专家 HDF5 回放）
physics         CPU；数据由 CPU physics 生成
control         50 Hz；PhysX 200 Hz / decimation 4
camera          robot_head_cam_rgb，640×480
instruction     Pick up the brown box from the shelf, and place it into the
                blue bin on the table located at the right of the shelf.
episode budget  1,200 environment steps
metric          success_rate、num_episodes
```

### 2.3 模型输入与输出

GR00T 每次读取一帧头部 RGB、语言指令，以及双臂、双手和腰部关节状态。模型一次输出未来 16 步，每步 32 维：

```text
left arm absolute joint positions    7
right arm absolute joint positions   7
left hand joint positions            7
right hand joint positions           7
base height command                  1
navigate command [vx, vy, yaw_rate]  3
                                      --
model action                         32
model chunk shape                 16×32
```

Arena adapter 将每步扩展为 50 维：

```text
[43 joint slots, vx, vy, yaw_rate, base_height, torso_roll, torso_pitch, torso_yaw]
```

只有双臂和双手的 28 个关节槽位来自 GR00T。12 个腿关节和 3 个腰关节由 Homie v2 lower-body policy/WBC 生成；3 维 torso RPY 在当前 adapter 中固定填零。最终仿真器执行 43 维关节位置目标。

因此不得把 B0 描述成“GR00T 端到端预测 43 个全身关节”。准确表述是：**GR00T 预测上身关节位置、底盘速度和基座高度，外部 WBC 生成下身关节动作。**

### 2.4 数据、gate 与报告套件

训练数据必须是冻结 revision 的 LeRobot v2.1 数据：100 episodes、94,936 frames、50 fps。小型 HDF5 只用于专家回放。

| 层级 | Seeds | 用途 | 通过/报告规则 |
|---|---|---|---|
| 安装 gate | 专家 episode 0 | 验证环境、资产、WBC 和 action replay | 完整运行，无异常退出 |
| 闭环 smoke | `0` | 验证 checkpoint、adapter 和 evaluator | `success_rate > 0`；只算 pipeline gate |
| 开发套件 | `0..9` | 日常模型比较 | 按 episode 聚合，报告 Wilson 95% CI |
| 最终套件 | `0..49` | 冻结模型后的正式报告 | 不允许据结果删 seed；报告逐 seed JSON |

同一比较中的模型必须使用相同 seed 集、1,200 steps、CPU physics、指令、相机、action chunk 和 WBC。单次 `1/1` 不能替代开发或最终套件。

## 3. 资源与安全前提

推荐至少：

- Linux x86_64；
- NVIDIA GPU，官方完整训练建议 48 GB 级别；本项目的 action-head LoRA 已在 RTX 5090 32 GB 上验证；
- NVIDIA driver 能运行 CUDA 12.8 PyTorch；
- Python 3.11、`uv`、Git、Git LFS、FFmpeg；
- 安装环境与 Isaac Sim 预留约 35 GB；官方推理权重约 7.6 GB；Omniverse/Arena 资产 cache 首次约 4–5 GB；数据约 0.5 GB；
- 训练 checkpoint 另行预留空间。

Agent 在远端执行前必须：

1. 阅读宿主机自己的 agent/运维指南，例如 `/etc/vast-agents-guide.md`；
2. 运行 `df -h`、`free -h`、`nvidia-smi` 和 `git status --short`；
3. 确认数据盘是否持久；当前 Vast 类实例的 `/workspace` 可能不是持久卷；
4. 保留用户已有数据、模型、日志和未提交代码；不得为了腾空间删除无关 checkpoint；
5. 训练、长评测和下载使用 Supervisor 或宿主机提供的持久进程管理器；
6. 不在训练占满 GPU 时并发启动未知显存需求的任务。

## 4. 标准目录

所有脚本默认以下布局；使用其他根目录时，直接运行脚本可设置 `ARENA_G1_ROOT`，但仓库提供的 Supervisor 配置冻结为默认路径。

```text
/workspace/time-invariant-whole-body-policy
├── configs/
│   ├── arena-g1-box-pick-place-v0.yaml
│   ├── arena-g1-gr00t-closedloop.yaml
│   └── arena-g1-gr00t-lora-closedloop.yaml
├── repos/
│   ├── IsaacLab
│   └── IsaacLab-Arena
│       └── submodules/Isaac-GR00T
├── venvs/arena
├── datasets/arena_g1
├── checkpoints/arena_g1
├── artifacts/deployment/arena_g1
├── artifacts/validation/arena-g1-bm0
└── logs/arena_g1_bm0
```

默认官方模型放在 `/dev/shm/arena_g1_checkpoint-20000`，因为已验证实例的系统盘空间不足。`/dev/shm` 会在重启后丢失。若机器有持久空间，可设置 `ARENA_G1_BASE_MODEL` 下载到持久目录，同时必须让 policy YAML 的 `model_path` 指向同一路径；不得出现“下载成功但评测读取另一个 checkpoint”的情况。

## 5. 安装路径选择

### 5.1 官方支持路径：Docker

Arena 0.1.1 的官方安装方式是 Docker。宿主机允许 Docker、GPU runtime 和 NGC 拉取时，优先按照上游 release 0.1.1 文档构建 Base + GR00T 镜像。仍必须 checkout 本文的 commit、使用本文的 dataset/model revision，并执行第 8 节的本项目验证。

### 5.2 本项目验证路径：源码 venv

无 Docker 权限或位于 RTX 50/Blackwell 时，使用：

```bash
cd /workspace/time-invariant-whole-body-policy
ARENA_G1_ROOT=/workspace/time-invariant-whole-body-policy \
  scripts/arena_g1/bootstrap-arena-g1.sh
```

该脚本会：

1. checkout 本文冻结的 Arena、Isaac Lab 和 Isaac-GR00T commit；
2. 创建 Python 3.11 venv；
3. 安装 Isaac Sim 5.1；
4. 安装 PyTorch 2.7.0 + CUDA 12.8；
5. editable-install Isaac Lab、Arena 与 GR00T；
6. 在 Blackwell 上安装匹配 torch 2.7/cp311 的 FlashAttention 2.7.4.post1 wheel；
7. 强制 `warp-lang==1.8.1`。更高版本曾导致 Isaac Sim 调用缺少 `warp.types.array`；
8. 做 CUDA matrix multiply 和核心 import smoke。

这条路径是项目实测兼容方案，不应伪装成 NVIDIA 官方对 Blackwell 的支持承诺。已有环境只检查、不安装：

```bash
ARENA_G1_VERIFY_ONLY=1 scripts/arena_g1/bootstrap-arena-g1.sh
```

## 6. 下载冻结数据和模型

公开仓库通常无需 HF token；遇到 rate limit 时运行 venv 中的 `hf auth login`，不得把 token 写入仓库或日志。

```bash
cd /workspace/time-invariant-whole-body-policy
scripts/arena_g1/prepare-arena-g1-data-model.sh
```

脚本只下载推理所需的五个模型文件，不下载 optimizer state。默认产物：

```text
datasets/arena_g1/arena_g1_loco_manipulation_dataset_generated_small.hdf5
datasets/arena_g1/arena_g1_loco_manipulation_dataset_generated/lerobot
/dev/shm/arena_g1_checkpoint-20000
artifacts/validation/arena-g1-bm0/install-manifest.json
```

下载结束必须由 `verify_arena_g1.py` 检查 commit、模型文件精确大小、`GR00T_N1_5` 架构，以及数据的 100 episodes / 94,936 frames / 50 fps。不要仅以目录存在判断完成。

## 7. 部署脚本与 Supervisor

在所有 `arena_g1_*` 程序停止时执行：

```bash
cd /workspace/time-invariant-whole-body-policy
scripts/arena_g1/deploy-arena-g1.sh
supervisorctl status | grep arena_g1
```

部署器将版本化脚本复制到 `artifacts/deployment/arena_g1`，并安装以下 Supervisor 程序：

| 程序 | 作用 |
|---|---|
| `arena_g1_eval` | 单 seed 闭环评测 |
| `arena_g1_suite` | 可断点续跑的多 seed 套件 |
| `arena_g1_train` | 2-epoch action-head LoRA |
| `arena_g1_finalize` | 等待训练，核验、合并并闭环复验 |
| `arena_g1_video` | 保存策略头部相机 MP4 |

部署器默认拒绝覆盖正在运行的 Arena 程序。确有理由热更新时，Agent 必须先解释风险，再显式设置 `ARENA_G1_ALLOW_RUNNING_UPDATE=1`。

没有 Supervisor 时可设置 `ARENA_G1_INSTALL_SUPERVISOR=0`，随后直接运行脚本；但不要用普通 SSH shell 承载数小时训练。

## 8. 按 gate 顺序运行

### 8.1 安装与 artifact 验证

```bash
/workspace/time-invariant-whole-body-policy/venvs/arena/bin/python \
  scripts/arena_g1/verify_arena_g1.py \
  --project-root /workspace/time-invariant-whole-body-policy \
  --dataset-root /workspace/time-invariant-whole-body-policy/datasets/arena_g1 \
  --model-root /dev/shm/arena_g1_checkpoint-20000 \
  --output artifacts/validation/arena-g1-bm0/install-manifest.json
```

只有 JSON 的 `status` 为 `pass` 才能继续。

### 8.2 专家回放

```bash
scripts/arena_g1/run-arena-g1-replay.sh
```

该步骤使用 `g1_wbc_pink` 和 CPU physics。它验证场景、资产、控制器和官方 demonstration，不验证 GR00T。

### 8.3 官方冻结 checkpoint 闭环 smoke

直接运行：

```bash
ARENA_G1_POLICY_CONFIG=$PWD/configs/arena-g1-gr00t-closedloop.yaml \
ARENA_G1_SEED=0 \
ARENA_G1_NUM_STEPS=1200 \
  scripts/arena_g1/run-arena-g1-eval.sh
```

或使用 Supervisor：

```bash
supervisorctl start arena_g1_eval
tail -f logs/arena_g1_bm0/supervisor-eval.log
```

日志必须出现类似：

```text
Metrics: {'success_rate': 1.0, 'num_episodes': 1}
```

通过条件是 `success_rate > 0`，不是必须等于 1。失败时检查 traceback、checkpoint 路径、资产下载、action adapter 和模型架构；不得直接启动训练来掩盖安装问题。

### 8.4 开发多 seed 套件

```bash
supervisorctl start arena_g1_suite
tail -f logs/arena_g1_bm0/suite-dev-seeds-0-9.log
```

脚本默认执行 seeds 0–9，并按 seed 保存 JSON；重启会跳过已完成结果。输出：

```text
artifacts/validation/arena-g1-bm0/dev-seeds-0-9/runs/seed-*.json
artifacts/validation/arena-g1-bm0/dev-seeds-0-9/summary.json
```

运行最终 50-seed 套件时，不修改源码：

```bash
ARENA_G1_SUITE_ID=final-seeds-0-49 \
ARENA_G1_SEEDS="$(seq -s ' ' 0 49)" \
  artifacts/deployment/arena_g1/run-arena-g1-suite.sh
```

### 8.5 录像

```bash
supervisorctl start arena_g1_video
```

默认视频是官方冻结 checkpoint、seed 0 的 G1 头部相机，写入：

```text
artifacts/validation/arena-g1-bm0/videos/frozen-official-seed-0-headcam.mp4
```

录像 runner 与官方 policy loop 相同，只增加逐帧 FFmpeg 编码。视频不是成功判据，最终仍以 evaluator metrics 为准。

## 9. 可选：本项目 2-epoch LoRA 适配

冻结 B0 通过后才允许训练。本项目的单卡 profile 从官方 task-tuned N1.5 继续训练 action head：

```text
dataset               100 episodes / 94,936 frames
batch                 4
steps per epoch       23,734
epochs                2
max steps             47,468
LoRA rank/alpha       32 / 32
LoRA dropout          0.1
learning rate         1e-4
trainable parameters  6,553,600（总参数 0.2400%）
frozen                LLM、vision backbone
tuned                  action projector、diffusion/DiT
```

启动前检查输出目录不存在或为空，且 `/dev/shm/arena_g1_lora_merged` 不存在。脚本会拒绝覆盖非空合并目录。

```bash
supervisorctl start arena_g1_train
# 确认训练进入 RUNNING 后再启动等待器
supervisorctl start arena_g1_finalize
supervisorctl status arena_g1_train arena_g1_finalize
```

finalizer 的 fail-closed 条件：

1. 训练进程状态为 `EXITED`；
2. 根目录存在 `trainer_state.json` 与 `adapter_model.safetensors`；
3. `global_step == 47468`；
4. LoRA `safe_merge` 成功；
5. 合并模型闭环 `success_rate > 0`。

任何条件失败时不得发布候选为 B0。训练 adapter、日志和错误必须保留用于诊断。

## 10. 监控与完成时间

```bash
supervisorctl status arena_g1_train arena_g1_finalize arena_g1_eval arena_g1_suite arena_g1_video
grep -aoE '[0-9]+/47468' logs/arena_g1_bm0/train-2ep.log | tail -1
grep -E 'Metrics:|Traceback|RuntimeError|CUDA out of memory|No space left' \
  logs/arena_g1_bm0/*.log | tail -50
nvidia-smi
df -h /workspace /dev/shm
```

RTX 5090 实测 action-head LoRA 稳态约 4.3 step/s；47,468 steps 纯训练约 3 小时。保存、合并和闭环评测另留 15–30 分钟。Agent 应使用实时 step/elapsed 重新计算 ETA，不要一直复述启动前估计。

## 11. 常见故障

### `no kernel image`、FlashAttention 或 Blackwell 失败

确认 PyTorch 是 `2.7.0+cu128`、GPU capability 是 `12.0`，FlashAttention wheel 与 torch 2.7/cp311 匹配。不要安装 GR00T `[base]` 后保留其默认 torch 2.5.1；它不能作为本项目 RTX 5090 环境。

### `warp.types.array` 不存在

检查 `warp-lang`，冻结版本是 `1.8.1`。Arena/Isaac Sim import smoke 通过前不要运行环境。

### checkpoint 下载后磁盘爆满

官方推理文件约 7.6 GB；不要下载 optimizer。空间不足时使用 `/dev/shm`，并明确它在重启后不可恢复。只删除本次生成且可重建的临时合并权重，不删除用户旧 checkpoint。

### 首次启动很慢

Arena 会从 Omniverse 拉取 Galileo/G1 资产，cache 可能超过 4 GB。区分“仍在下载/编译 shader”和进程死锁；查看日志、网络和 cache 增长后再判断。

### 模型成功但看起来不是直接控制走路

这是预期 action contract。GR00T 输出 `vx, vy, yaw_rate`，Homie/WBC 生成腿部动作；不要把外部 locomotion controller 的行为归到 GR00T 的直接关节预测能力。

### Supervisor 显示 `EXITED`

一次性程序正常完成也会显示 `EXITED`。必须结合日志、metrics 和期望 artifact 判断；不要只凭状态字符串认定失败。

## 12. 参考验证证据

2026-08-14 在 RTX 5090 / driver 580.126 上已得到：

- 官方小型 HDF5 episode 0：853 actions 完整回放；
- 官方冻结 GR00T N1.5，seed 0，1,200 steps：`1/1` 成功；
- 100-step action-head LoRA smoke：`train_loss=0.003976`、`3.852 step/s`；
- 100-step LoRA 合并后，seed 0，1,200 steps：`1/1` 成功；
- 官方冻结 checkpoint 录像复验：`1/1` 成功。

这些结果只说明链路能工作。开发和最终报告仍必须运行冻结的 multi-seed suite。

## 13. 版本化脚本清单

```text
bootstrap-arena-g1.sh               源码/Blackwell 环境安装
prepare-arena-g1-data-model.sh      固定 revision 下载和 manifest
verify_arena_g1.py                  环境、commit、数据、模型、action contract gate
deploy-arena-g1.sh                  部署脚本和 Supervisor 配置
run-arena-g1-replay.sh              专家 HDF5 回放
run-arena-g1-eval.sh                单 seed 官方闭环
run-arena-g1-suite.sh               多 seed 可续跑评测
summarize_arena_g1.py               单次结果 JSON
aggregate_arena_g1.py               suite 聚合与 Wilson CI
run-arena-g1-video.sh               带录像的闭环
policy_runner_video.py              官方 loop + POV 编码
run-arena-g1-train.sh               单卡 action-head LoRA
merge_arena_g1_lora.py              安全合并 LoRA
finalize-arena-g1-train.sh          训练完成 gate 与自动复验
arena-g1-*.conf                     Supervisor 程序
```

## 14. 变更控制

以下任一变化都必须创建新的协议 ID，不得覆盖 `arena-g1-box-pick-place-v0`：

- Arena、Isaac Lab、GR00T、dataset 或 checkpoint revision 变化；
- N1.5 升级到 N1.6/N1.7；
- task object、initial pose、instruction、WBC 或 success condition 变化；
- CPU physics 改 GPU physics；
- action horizon/chunk、模型输入或输出 contract 变化；
- seed manifest 或 1,200-step budget 变化。

只改变日志位置、checkpoint 存储目录或进程管理方式，不改变实验语义时，可以保持协议 ID，但必须记录实际路径和宿主机信息。
