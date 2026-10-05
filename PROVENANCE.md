# 来源与导出范围

上游项目为 [RLinf/RLinf](https://github.com/RLinf/RLinf)，固定 commit `c70606f08cdca259b8dec03d4430926b5b8fac9d`，上游许可证为 Apache-2.0，其原文随包保存在 [UPSTREAM_LICENSE.txt](UPSTREAM_LICENSE.txt)。许可证副本说明上游代码来源，不为新研究材料另行选择许可证。

研究补丁修改的文件与原因见 `routes/route2/patches/current-minimal-compat.patch` 的各 diff：DTensor 兼容、MUSA 探测回退、Ray CPU Channel 接口及对应测试/文档。本地适配提交为 `3e7329c629293f1b0f9c331b5bb24eccc3e3f142`，由本地 `Codex <codex@localhost>` 身份提交并 Signed-off-by；未代表厂商或上游提交。

| 导出文件 | 本地研究仓库中的来源 |
|---|---|
| `README.md`、`ROADMAP.md`、`REPRODUCE.md`、本文件、`.gitignore`、`scripts/run_probes.sh` | `routes/route2/publishing/` 的发布模板 |
| `routes/route2/README.md`、`patches/`、`probes/`、`evidence/` | `routes/route2/` 对应文件，原样复制 |
| `probes/mujoco_ppo_worker.py` | `routes/route1/mujoco_ppo_worker.py`，共用 runner，原样复制；入口显式传路线二 commit/label |
| `locks/sources.json` | `locks/sources.json`，只选择 RLinf 路线二并记录预期源码 tree |
| `locks/environment.md`、`locks/route2-environment.txt` | 同路径原始环境记录；文档涉及路线一的描述保留作历史背景 |
| `scripts/restore_sources.py`、`scripts/summarize_log.py` | 同路径共用脚本 |
| `UPSTREAM_LICENSE.txt` | 固定上游 commit 的 `LICENSE` |

`EXPORT_MANIFEST.json` 记录每个导出文件的 SHA256、路径、字节数与来源 commit，记录复制后的包内容。共用 MuJoCo runner 的 SHA256 与研究仓库原文件完全相同，保留首轮可验证性。

包不包含上游完整 Git 历史、其他路线的补丁/结果、虚拟环境、模型权重或 checkpoint。历史 server paths 只是证据产物位置；SSH 连接信息和密钥没有复制到包中。

官方 MUSA 支持的参考：[#1464](https://github.com/RLinf/RLinf/pull/1464)、[#1578](https://github.com/RLinf/RLinf/pull/1578)、[GR00T 配方](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/gr00t.html)。这些上游记录不替代本项目 S4000/旧栈的具体实测。
