# S4000 首轮实验环境

2026-10-04 在本会话服务器实测：单卡 MTT S4000，49152 MiB，15 CPUs；`mthreads-gmi` 显示 Driver 2.7.0，工具版本 1.14.0。

| 组件 | 实测版本 |
|---|---|
| Python | 3.10.8 |
| Toolkit / MCC / runtime | 3.1.0，`/usr/local/musa` → `/usr/local/musa-3.1.0` |
| Torch distribution metadata | `2.2.0a0+git8ac9b20` |
| `torch.__version__` | `2.2.0` |
| Torch-MUSA distribution metadata | `1.3.0` |
| `torch_musa.__version__` | `1.3.0+81caf0a` |
| muDNN / MCCL / muBLAS | 2.7.0 / 2.11.4 / 1.6.0 |
| MuJoCo / Gymnasium | 3.3.7 / 1.1.1 |
| Ray / OmegaConf / Accelerate | 2.59.0 / 2.3.1 / 1.15.0 |
| 新增 regex / pytest | 2024.11.6 / 8.3.5 |

原匹配 Torch-MUSA 位于系统 `/root/miniconda3`。早先探针环境 `/root/autodl-tmp/rlinf-s4000` 提供 Gymnasium、MuJoCo、Ray 等已安装包；本轮 `envs/route1` 与 `envs/route2` 通过 `--system-site-packages` 和 `research-base.pth` 只读继承它们。隔离的是各路线的新增包、源码加载和进程；并非复制了两套独立的系统动态库。

`scripts/prepare_envs.sh` 给各 venv 用 `--no-deps` 安装明确版本的轻量依赖，不替换 Torch。完整包清单为 `route1-environment.txt` 和 `route2-environment.txt`；包清单中的 `rlinf==0.3.0` 来自先前安装包，实际测试代码由各路线 `PYTHONPATH` 的固定源树提供，以 `sources.json` 与结果里的 `source_commit` 为准。

Toolkit 的完整组件 commit 见 `routes/route4/results/system_versions.json`。公开 Torch-MUSA v1.3.0 tag 和当前 `+81caf0a` 构建不是同一个已确认的 Git commit，不能混作源码配套证据。
