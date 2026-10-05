# Spatial action-layout probe

运行入口为 [spatial_action_layout_probe.py](spatial_action_layout_probe.py)。固定字节与外部文件清单见 [source-manifest.json](source-manifest.json)，配置、mask边界、完整CPU/MUSA命令及验收门槛见 [spatial-action-layout.md](../../planning/spatial-action-layout.md)。

完整 argv、导出文件与预期行数见 [run-plan.json](run-plan.json)，采用协调者新建的 `route2-integration` 隔离解释器。

正式宽度：action H32/D48、inner1536、cross2048、Q49/K570，真实DiT两层输出1024；独立VL Attention H32/D64、inner2048、Q/K570。所有执行是随机共享权重、eval/dropout0；不加载checkpoint，不做optimizer/PPO/simulator。

固定DiT忽略encoder_attention_mask，基线保留no-mask；rightpad与4D mask合同使用独立真实Diffusers Attention。严格gate与analytic-zero semantic gate分开；exit0仅表示严格gate与全部合同通过。已有toy/action/backbone文件原字节不变。

本地仅静态验证，见 [static-validation.json](static-validation.json)。实机数值由协调者执行，不以静态或semantic结果声称strict通过。
