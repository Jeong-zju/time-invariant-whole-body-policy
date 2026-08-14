# 给人类用户的 Arena G1 Codex Prompt

更新日期：2026-08-14

## 使用方法

把下面 prompt 中尖括号字段替换为实际值，然后完整复制给 Codex。不要把密码、私钥或 Hugging Face token 写入 prompt；SSH 应使用已经配置的 key/agent，HF token 应在远端通过环境或交互登录提供。

推荐第一次只执行到官方 checkpoint 的闭环 smoke。确认非零成功率后，再要求运行 multi-seed suite 或训练。

## 可复制 Prompt

```text
请在远程服务器上部署并运行本仓库冻结的 benchmark：

NVIDIA Isaac Lab Arena G1 Loco-Manipulation — Box Pick-and-Place

连接信息：
- SSH：ssh -p <SSH_PORT> <SSH_USER>@<SSH_HOST>
- 项目根目录：<PROJECT_ROOT，推荐 /workspace/time-invariant-whole-body-policy>
- 模型存储：<MODEL_ROOT；空间不足时填 /dev/shm/arena_g1_checkpoint-20000>
- 本次目标：<只做官方闭环 smoke / 做 seeds 0-9 开发套件 / 做 2-epoch LoRA 并复验 / 做最终 seeds 0-49>

执行前先完整阅读：
1. docs/arena-g1-bm0-deployment.md
2. configs/arena-g1-box-pick-place-v0.yaml
3. 远端宿主机自己的 Agent/运维指南；若存在 /etc/vast-agents-guide.md，必须先读完。

必须遵守以下要求：

1. 严格使用协议 arena-g1-box-pick-place-v0 中冻结的 commit、dataset revision、checkpoint revision、CPU physics、1,200-step budget 和 seed manifest；不要自动升级上游版本。
2. 官方 checkpoint 的真实架构是 GR00T N1.5，不得写成 N1.6 或 N1.7。
3. 先检查 nvidia-smi、磁盘、内存、权限、现有 Supervisor 状态和 git status。保留所有用户数据、旧 checkpoint、日志和未提交修改，不得为了腾空间删除无关文件。
4. 优先使用官方 Docker 路径；若远端没有 Docker 权限或使用 RTX 50/Blackwell，则按 runbook 使用 scripts/arena_g1/bootstrap-arena-g1.sh 的源码 venv 路径，并明确这是项目验证过的兼容方案。
5. 下载后必须运行 verify_arena_g1.py。只有 commit、模型文件、GR00T_N1_5 架构、100 episodes / 94,936 frames / 50 fps 和 CUDA 检查全部通过，才进入仿真。
6. 严格按 gate 执行：安装验证 → 专家 HDF5 回放 → 官方冻结 checkpoint seed 0 闭环。官方 seed 0 的 success_rate 必须大于 0；否则停止，不要启动训练。
7. 长下载、训练和多 seed 评测使用 Supervisor 或远端规定的持久进程管理器。每隔合理时间汇报 step、速度、GPU/磁盘状态和按实时吞吐计算的 ETA；不要用普通 SSH shell 承载数小时任务。
8. 如果本次目标包含训练，使用冻结的 action-head LoRA profile：batch 4、rank/alpha 32/32、LR 1e-4、47,468 steps（2 epochs），冻结 LLM 和 vision。训练完成必须由 finalizer 检查 global_step、合并权重并再次闭环；最终 success_rate 为 0 时不得称为通过。
9. 正确描述 action contract：GR00T 输出 16×32 chunk，即双臂/双手 28 维关节位置、3 维 [vx, vy, yaw_rate] 和 1 维 base height。Arena adapter 扩展到 50 维，Homie/WBC 生成腿和腰部动作。不要声称 GR00T 直接预测完整 43 关节全身动作。
10. 单次 1/1 只报告为 pipeline gate。开发套件运行 seeds 0-9，最终套件运行 seeds 0-49，并按 episode 聚合 success rate、Wilson 95% CI 和逐 seed JSON。
11. 每次执行保存命令、日志、机器可读 JSON、版本、路径、seed 和 checkpoint 标识。远端非持久卷上的重要结果在实例回收前同步到持久存储；未经我授权不要上传公开仓库或发送外部消息。

请自主完成所有安全、范围内且可逆的步骤，不要在每个普通命令前询问。遇到版本冲突、空间不足、缺少权限、官方 gate 为零或可能覆盖已有数据时停止并报告具体证据。

最终向我汇报：
- 安装/部署是否完成；
- 实际使用的所有 commit/revision 和模型架构；
- 专家回放结果；
- 每个已运行评测的 seed、episodes、success rate 和聚合 CI；
- 若训练，当前/最终 step、loss、吞吐、checkpoint、合并与复验结果；
- 日志、JSON、checkpoint 和视频的绝对路径；
- 仍存在的风险，包括 /dev/shm 或非持久 /workspace；
- 与冻结协议的任何偏差。没有偏差时明确写“无偏差”。
```

## 当前服务器示例

只展示字段形式，不建议把临时 IP 固化进仓库：

```text
SSH_PORT=<端口>
SSH_USER=root
SSH_HOST=<服务器地址>
PROJECT_ROOT=/workspace/time-invariant-whole-body-policy
MODEL_ROOT=/dev/shm/arena_g1_checkpoint-20000
本次目标=只做官方闭环 smoke
```

后续继续训练时，只需把“本次目标”改成：

```text
做 2-epoch action-head LoRA，训练结束自动合并，并用相同 seed 0 闭环复验；通过后运行 seeds 0-9 开发套件。
```
