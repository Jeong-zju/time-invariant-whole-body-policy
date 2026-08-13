# Phase -1 远程部署记录

更新日期：2026-08-13

## 结论

Phase -1 的远程数据与模型基础设施已经部署到服务器，三个任务的数据均通过低维轨迹、视频、官方 LeRobot loader 和 GPU 环境检查；官方 `turning_on_radio` GR00T checkpoint 也已下载并校验。

目前还不能宣布 Phase -1 完成：官方闭环 rollout 需要访问 gated 模型 `nvidia/Cosmos-Reason2-2B`，OmniGibson/BEHAVIOR 安装还需要操作者明确接受相应许可条款。完成这两项后，才能执行闭环 websocket rollout，并生成第 3 和第 5 项准入证据。

## 远程入口与目录

```text
SSH         ssh -p 16867 root@146.115.17.138
Jupyter     https://localhost:8081
远程根目录  /workspace/time-invariant-whole-body-policy
```

用户指定的本地 `8080` 已被现有 Docker 服务占用，因此没有终止或覆盖该服务，而是把 SSH 隧道改到本地 `8081`：

```bash
ssh -N -p 16867 \
  -L 8081:localhost:8080 \
  root@146.115.17.138
```

远端 Jupyter 使用 TLS，自签名证书场景下浏览器可能需要手动继续访问。

该 Vast.ai 实例的 `/workspace` **不是持久卷**。停止再启动通常仍会保留文件，但销毁或回收实例会丢失全部远程数据。开始大规模训练前，应先挂载持久卷或把数据、checkpoint、实验配置和结果同步到外部存储。

## 已锁定的软件与数据

| 项目 | 版本或路径 |
|---|---|
| GPU | NVIDIA RTX 5090，32 GB，compute capability 12.0 |
| Driver / 系统 CUDA | 595.84 / 13.2 |
| PyTorch | `2.7.1+cu128`，CUDA 矩阵乘测试通过 |
| BEHAVIOR-1K | tag `v3.9.1`，commit `26f2c7ef7b9cf96bd0414f81e1e751e493762779` |
| Isaac-GR00T challenge fork | branch `behavior`，commit `ace36d935b376fbf25cd56371e23877b95407c40` |
| 数据集 | `behavior-1k/2026-challenge-demos` |
| 数据 revision | `4f50b44796641a4d526a19d9aeadc8aa51e2f2c2` |
| 数据目录 | `/workspace/time-invariant-whole-body-policy/data/2026-challenge-demos` |
| 数据规模 | 35,123,422,375 bytes；204 个非缓存文件；181 个 MP4 |
| GR00T 环境 | `/workspace/time-invariant-whole-body-policy/repos/Isaac-GR00T/.venv` |
| 数据检查环境 | `/workspace/time-invariant-whole-body-policy/venvs/data` |

官方 dGPU 安装脚本执行后，GR00T 源码文件保持不变，但 `uv.lock` 被安装流程更新：加入 `websockets==16.0`，同时重新解析部分平台 marker 和 CUDA wheel 来源。因此上表中的 commit 是源码基线，当前部署态还包含这份未提交的 lockfile 差异。

## 数据准入结果

已下载并校验 `turning_on_radio`、`bringing_water` 和 `sweeping_garage` 三个任务。所有 181 个 MP4 均通过 `ffprobe`，没有零字节文件。官方 `LeRobotEpisodeLoader` 能读取数据，并成功解码三路 RGB；schema 为 61 维 state、23 维 action，底盘 state/action 均为 3 维。

| task | episodes | rows | timestamp 最大误差 | 可视化 episode | 时长 | 积分终点 `[x,y,yaw]` |
|---|---:|---:|---:|---:|---:|---|
| `turning_on_radio` / 0 | 200 | 429,928 | `8.14e-6 s` | 0 | 65.17 s | `[0.516,-0.549,0.649]` |
| `bringing_water` / 17 | 200 | 1,887,709 | `2.24e-5 s` | 3400 | 238.97 s | `[0.919,2.223,-1.182]` |
| `sweeping_garage` / 57 | 200 | 894,606 | `2.24e-5 s` | 11400 | 125.63 s | `[3.504,-0.484,0.545]` |

检查输出位于：

```text
/workspace/time-invariant-whole-body-policy/artifacts/validation
/workspace/time-invariant-whole-body-policy/artifacts/episode-splits.json
/workspace/time-invariant-whole-body-policy/artifacts/phase-minus-1-lock.json
```

episode split 使用固定种子 `20260813`，每个任务为 180 train / 20 validation，同一 episode 不跨 split。闭环配置固定为 public instance `0–9` 主报告、`10–19` 冻结补充，`20–39` 不参与调参，训练随机种子为 `0/1/2`。

## Checkpoint 与推理检查

官方 `turning_on_radio` checkpoint 已下载并通过 ZIP 完整性检查：

```text
压缩包  /workspace/time-invariant-whole-body-policy/checkpoints/turning_on_radio_GR00T-checkpoint-150000.zip
大小    6,917,651,266 bytes
SHA256  74eab585f4cf0f92326c9d4d74732343f6d789b05e945d3e71afe46bac69cb46
解压后  /workspace/time-invariant-whole-body-policy/checkpoints/turning_on_radio_GR00T-checkpoint-150000
```

GR00T 的 R1Pro modality 能正常加载，RTX 5090 上的 PyTorch CUDA 运算已通过。Policy Server 已运行到加载 backbone 的阶段，随后因 Hugging Face 对 `nvidia/Cosmos-Reason2-2B` 返回 `401 GatedRepoError` 而停止。这说明本地 checkpoint 和启动参数已经走通，剩余问题是访问授权，不是 CUDA 或 checkpoint 损坏。

## 尚未完成的两项操作

### 1. 登录 Hugging Face gated 模型

先在浏览器申请并接受 [`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B) 的访问条件，然后在远程服务器执行：

```bash
/workspace/time-invariant-whole-body-policy/venvs/data/bin/hf auth login
```

Token 只应输入远程终端，不要粘贴到聊天或写入项目文件。登录完成后可重新启动 `scripts/b1k/serve_b1k.py`。

### 2. 明确接受仿真安装条款

BEHAVIOR 官方安装脚本要求明确接受：

- [Conda Terms of Service](https://legal.anaconda.com/policies/en/)；
- [NVIDIA Isaac Sim EULA](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement)；
- `BEHAVIOR-1K/setup.sh` 中的 BEHAVIOR Data Bundle EULA。该数据许可限定非商业学术研究用途，并包含禁止提取、逆向和再分发加密数据等条件。

没有操作者确认前，不应自动代为传入 `--accept-conda-tos`、`--accept-nvidia-eula` 和 `--accept-dataset-tos`。确认后再安装 OmniGibson、BDDL、JoyLo、dataset 和 evaluator。

## 完成 Phase -1 的最后验收

获得授权后只剩一条主链：启动 GR00T Policy Server，启动 BEHAVIOR evaluator，在 `turning_on_radio` public instance 上完成至少一次 websocket rollout，并保存 JSON metrics、视频、真实 action timestamp、Policy 返回时间、底盘轨迹和关节轨迹。届时再把 `Phase -1` 标为完成，开始 Phase 0 的原始 velocity-action baseline。
