# Spatial layout v1 原字节快照

保存 runner、source-manifest、run-plan、static-validation 与 route1 review 的原字节，供核对首次CPU实测使用的来源。runner SHA256 `0a0966b2fac705a55b916ee7145ad808553decac7e1c8c166f59aa97cfdf99ae`，manifest SHA256 `a965d31996791a66ddf720cea63da7d06bc0b313d1fbf932139f441f24a72fa6`。

协调者报告首次CPU作业到真实DiT阶段的证据写入时失败：`Object of type dtype is not JSON serializable`。此前持久化的partial JSON有前9个Attention cases的18 rows，命令exit1；不能将partial结果当成完整all范围通过。实际JSON与stdout/stderr由协调者保留，记录来源不是本代理执行。

原因是DiT `effective_config`保留真实 `torch.dtype`。v2仅对证据树递归把该类型转成字符串，不改变模型、权重、输入、forward/VJP或strict/semantic门槛。旧快照不覆盖；旧run-plan反映当时实际预备路径，不指向当前v2源码的可复现版本。重现旧版需选择单独历史目录，并显式指定原固定support/config/DiT路径与本快照manifest。
