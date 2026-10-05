# Backbone interface v1 exact-source snapshot

These two files preserve the exact bytes used by the first CPU/MUSA runs:

- `gr00t_backbone_interface_probe.py`: SHA256 `e136cc33b556d1e20bb09051d4126add048cfc68ca3bec6bdc97bf7d950bb514`
- `fp32_backbone_attention.py`: SHA256 `ee6caf6745a92cf9b339a70c71067fb562730f4fe14fea80cfed2450d6ceef01`

Both remain together so the runner imports its original sibling kernel. Since
the config files are not duplicated into this snapshot, pass their paths
explicitly when reproducing from the research repository root:

```sh
envs/route2-backbone/bin/python routes/route2/model_probes/versions/backbone-interface-v1/gr00t_backbone_interface_probe.py --device cpu --eagle-config routes/route2/model_probes/audit_sources/Isaac-GR00T/gr00t/model/backbone/eagle2_hg_model/config.json --checkpoint-config routes/route2/model_probes/backbone_sources/checkpoint-config.json --output /tmp/backbone-interface-v1-reproduction.json
```

Original CPU evidence: `artifacts/run-20261005-process-backbone/backbone-cpu.json`,
SHA256 `a96bc0febc985116fd9e02ea05003aa11f34e09de771ff8f0c3c8549b69ebde2`;
8/8 rows passed.

Original MUSA evidence: `artifacts/run-20261005-process-backbone/backbone-musa.json`,
SHA256 `10092a9d2caf55fc9a10fcd711e8d42a0c234f324ea67f880afcd17df63d40be`;
11/12 rows passed. The Qwen3 BF16 MUSA row raised at a diagnostic BF16
`weights.masked_select`, unsupported by Torch-MUSA. It cannot be counted as a
passed attention row. The v2 kernel only casts the diagnostic read to FP32;
attention results/VJP and numeric thresholds are unchanged. The v2 runner also
gates binding invariants and checks global registry entry identities.
