# 路线二成果仓库移交

2026-10-05，远端已设为用户创建的 [prophetricker/rlinf-musa-lab](https://github.com/prophetricker/rlinf-musa-lab)。独立本地成果仓库位于 `artifacts/route2-github/`；研究根仓库保存全部路线，成果仓库只导出路线二与复现所需的共用 runner。GitHub CLI 已完成 `prophetricker` 账号认证；已推送并核对远端 `main` 与本地 commit 一致。首个学习节点发布提交为 [`6a99f21`](https://github.com/prophetricker/rlinf-musa-lab/commit/6a99f2128424c19be22f97b1badab103017395e2)，对应研究提交 `c22ba8b`。

## 固定材料

| 材料 | 来源 |
|---|---|
| 上游 | RLinf `c70606f08cdca259b8dec03d4430926b5b8fac9d` |
| Worker 学习适配基线 | `3e7329c629293f1b0f9c331b5bb24eccc3e3f142`，tree `b78ed3694bcd1d69afd887e6ce11b780d7528ecf` |
| FSDP1 实验增量 | `f9b74d95f97ad4234311580ab32c7b586e1dc7af`，tree `1a4eef5dd1f4984652168ebecce1d3cc194ec598` |
| 系统栈 | Driver 2.7.0 / MUSA 3.1.0 / Torch 2.2.0 / Torch-MUSA 1.3.0 |
| 新学习结果 | Pendulum 与 HalfCheetah 各三种子，全部 102,400 transitions 与独立 20 局 test；见 [学习报告](route2-learning.md) |
| 训练后端 | 真实 RLinf FSDP1 单卡更新，9/9 CPU 回归；尚无 FSDP checkpoint/官方 Actor |
| 模型合约 | synthetic attention 数值/梯度/mask/内存证据；未加载完整 GR00T |

## 导出与检查

研究根仓库的 `scripts/export_route2.py` 按清单生成新目录，拒绝覆盖已有输出。共用 `routes/route1/mujoco_ppo_worker.py` 原样导出为 `probes/mujoco_ppo_worker.py`。下载的 audit 源码只导出 URL/commit/hash manifest。

导出不包含其他路线结果、上游完整 Git 历史、环境、权重、checkpoint 或认证材料。`EXPORT_MANIFEST.json` 保存各文件 SHA256、字节数及研究来源 commit。既有成果 Git 的 `.git` 保留，刷新时只同步清单中的材料；不强推。

本轮离线恢复已验证 Worker 基线与 FSDP1 实验的两个源码 tree 完全一致，导出 manifest 和共用 runner 一致性通过，所有相对 Markdown 链接有效，Python/Bash 语法通过；学习语义/评估 CPU 测试 10/10。首个节点包含 84 个材料文件加 manifest，总材料约 3.28 MB。FSDP1 从基线加实验补丁重建，保持基线与实验边界。

提交使用本地显式身份 `Codex <codex@localhost>`，不修改全局 Git 身份。登录令牌由 GitHub CLI 管理，不写入研究记录。
