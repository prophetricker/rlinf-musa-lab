# 来源与导出范围

上游项目为 [RLinf/RLinf](https://github.com/RLinf/RLinf)，固定 commit `c70606f08cdca259b8dec03d4430926b5b8fac9d`，上游许可证为 Apache-2.0，其原文随包保存在 [UPSTREAM_LICENSE.txt](UPSTREAM_LICENSE.txt)。许可证副本说明上游代码来源，不为新研究材料另行选择许可证。

研究补丁修改的文件与原因见 `routes/route2/patches/current-minimal-compat.patch` 的各 diff：DTensor 兼容、MUSA 探测回退、Ray CPU Channel 接口及对应测试/文档。本地适配提交为 `3e7329c629293f1b0f9c331b5bb24eccc3e3f142`，由本地 `Codex <codex@localhost>` 身份提交并 Signed-off-by；未代表厂商或上游提交。

| 导出文件 | 本地研究仓库中的来源 |
|---|---|
| `README.md`、`ROADMAP.md`、`REPRODUCE.md`、本文件、`.gitignore`、`scripts/run_probes.sh`、`scripts/run_learning.sh` | `routes/route2/publishing/` 的发布模板 |
| `routes/route2/README.md`、`patches/`、`probes/`、`evidence/` | `routes/route2/` 对应文件，原样复制 |
| `routes/route2/learning/`、`planning/`、`model_probes/`、`fsdp_probes/` | 存在时复制路线二新增材料；audit_sources 只复制 URL/hash manifest，不复制下载源码缓存 |
| `probes/mujoco_ppo_worker.py` | `routes/route1/mujoco_ppo_worker.py`，共用 runner，原样复制；入口显式传路线二 commit/label |
| `locks/sources.json` | `locks/sources.json`，只选择 RLinf 路线二并记录预期源码 tree |
| `locks/environment.md`、`locks/route2-environment.txt` | 同路径原始环境记录；文档涉及路线一的描述保留作历史背景 |
| `scripts/restore_sources.py`、`scripts/summarize_log.py` | 同路径共用脚本 |
| `reports/route2-learning.md`、`reports/route2-handoff.md` | 同路径路线二报告 |
| `UPSTREAM_LICENSE.txt` | 固定上游 commit 的 `LICENSE` |

`EXPORT_MANIFEST.json` 记录每个导出文件的 SHA256、路径、字节数与来源 commit，记录复制后的包内容。共用 MuJoCo runner 的 SHA256 与研究仓库原文件完全相同，保留首轮可验证性。

包不包含上游完整 Git 历史、其他路线的补丁/结果、虚拟环境、模型权重或 checkpoint。历史 server paths 只是证据产物位置；SSH 连接信息和密钥没有复制到包中。

新增Eagle补丁来自 [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T/tree/4af2b622892f7dcb5aae5a3fb70bcb02dc217b96) 的固定源码；上游Apache-2.0原文保存在 [UPSTREAM_GR00T_LICENSE.txt](UPSTREAM_GR00T_LICENSE.txt)。完整归档169文件Git blob匹配上游tree；导出只保存补丁/config/manifest，使用 `scripts/restore_eagle_sources.py` 恢复。公开Spatial index/headers/LFS metadata固定HF revision；权重不提交。真实结构审计和forward/strict-load范围由 [报告](reports/route2-spatial-eagle.md) 分别说明。

官方 MUSA 支持的参考：[#1464](https://github.com/RLinf/RLinf/pull/1464)、[#1578](https://github.com/RLinf/RLinf/pull/1578)、[GR00T 配方](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/gr00t.html)。这些上游记录不替代本项目 S4000/旧栈的具体实测。

FSDP1 增量对应本地提交 `f9b74d95f97ad4234311580ab32c7b586e1dc7af`，独立 patch SHA256 `f7a791d89fbc9973d4d7a95f0e4a22b14e8467631c214e74716b9f899c5791f4`，不覆盖 Worker 学习基线。实际复测版本由 source/probe hash 关联；历史失败补丁保存在 `fsdp_probes/history/`。

新增跨进程checkpoint探针保留旧helper与生产补丁，结果由四份save/restore JSONL及独立run-validation关联。骨干组件使用Transformers4.51.3固定源码和公开Spatial配置；downloaded source只按manifest恢复，probe的实例kernel基于Apache-2.0的Transformers合约并保留来源注释。旧v1源码与失败证据保留在versions/evidence中。新增环境准备与导出验证脚本、三个环境锁分别来自研究仓库同路径。
