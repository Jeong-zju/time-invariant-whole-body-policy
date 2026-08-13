# 协作开发规范

## 分支模型

`main` 始终表示已评审、可复现的稳定状态。所有实现、实验和论文修改都从最新 `main` 创建短生命周期分支，并通过 Pull Request 合并；不设置长期个人分支或长期 `develop` 分支。

分支名格式为：

```text
<type>/<owner>/<topic>
```

### Type

| 类型 | 用途 | 示例 |
|---|---|---|
| `feat` | 新功能或算法实现 | `feat/ranpeng/progress-sampler` |
| `exp` | 单次实验、消融或分析 | `exp/jeong/frequency-sweep` |
| `eval` | benchmark 与评测指标 | `eval/ranpeng/time-warp-metrics` |
| `data` | 数据转换、切分与校验 | `data/jeong/local-se2-cache` |
| `fix` | 缺陷修复 | `fix/ranpeng/yaw-wraparound` |
| `docs` | 研究文档与决策记录 | `docs/jeong/benchmark-protocol` |
| `paper` | 论文正文、图表与附录 | `paper/ranpeng/method-section` |
| `chore` | CI、依赖和仓库维护 | `chore/jeong/test-workflow` |

### Owner

| 负责人 | 分支别名 |
|---|---|
| Jeong Li | `jeong` |
| 邱然鹏 | `ranpeng` |

`topic` 使用简短英文 kebab-case，描述单一任务；不要使用 `test`、`update`、姓名缩写或日期作为唯一主题。一个分支只解决一个可独立评审的问题。

## 标准流程

```bash
git switch main
git pull --ff-only
git switch -c exp/ranpeng/time-warp-ablation
```

开发期间尽早创建 Draft PR。准备合并前同步最新 `main`、完成测试，并由另一位合作者评审。默认采用 squash merge，使 `main` 上每个提交对应一个完整变更。

## Commit 与 PR

Commit message 使用英文 Conventional Commits 风格，例如：

```text
feat: add progress-based action sampler
exp: evaluate nonlinear time warps
paper: describe path representation
```

PR 必须说明：

- 研究问题或工程问题；
- 实际变更与未变更的边界；
- 验证方法和关键结果；
- 相关 issue、实验配置或结果路径；
- 是否影响论文中的方法、指标或结论。

## 实验可复现性

进入主结果表的实验必须固定并记录：

- 当前 Git commit；
- 数据集名称与 revision；
- 训练、验证和评测实例划分；
- 完整配置和随机种子；
- 模型 checkpoint 来源或哈希；
- 软件版本、硬件与控制频率；
- 原始 metrics 和生成图表的脚本。

实验结论不得只保存在聊天、终端日志或本地 notebook 中。设计决策写入 `docs/`，实验配置与索引写入 `experiments/`，论文相关产物写入 `paper/`。

## 数据与安全

不要提交原始数据、checkpoint、访问令牌、服务器凭据、个人密钥或受许可限制的资产。提交前运行：

```bash
git status --short
git diff --cached
```

若发现秘密信息，停止推送并先从提交历史中清理、轮换对应凭据。
