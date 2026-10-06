"""Construct the small fixed upstream configuration, without Hydra or GPU use."""
from pathlib import Path


def make_config(source: Path, model_path: Path, *, steps: int = 8):
    from omegaconf import OmegaConf

    defaults = source / "examples/embodiment/config"
    model = OmegaConf.load(defaults / "model/gr00t.yaml")
    model.model_path = str(model_path.resolve())
    model.num_action_chunks = 1
    model.musa_eager_attention = True
    model.rl_head_config.add_value_head = True
    fsdp = OmegaConf.load(defaults / "hybrid_engines/fsdp.yaml")
    fsdp.sharding_strategy = "no_shard"
    fsdp.use_orig_params = True
    fsdp.limit_all_gathers = True
    fsdp.save_full_model_weights = False
    fsdp.save_runtime_state = True
    fsdp.mixed_precision.param_dtype = None
    fsdp.mixed_precision.reduce_dtype = None
    fsdp.mixed_precision.buffer_dtype = None
    # Default GR00T wrap classes separate the frozen BF16 Eagle from FP32 head.
    env = OmegaConf.load(defaults / "env/libero_spatial.yaml")
    env.total_num_envs = 1
    env.rollout_epoch = 1
    env.specific_reset_id = 0
    env.max_steps_per_rollout_epoch = steps
    env.video_cfg.save_video = False
    env.video_cfg.video_base_dir = "/root/autodl-tmp/s4000-research/route2/results/video"
    return OmegaConf.create({
        "cluster": {"num_nodes": 1, "component_placement": {"actor,env,rollout": "all"}},
        "runner": {"task_type": "embodied", "only_eval": False,
                   "val_check_interval": -1, "ckpt_path": None},
        "actor": {"group_name": "ActorGroup", "training_backend": "fsdp",
                  "model": model, "fsdp_config": fsdp, "seed": 1234,
                  "enable_offload": False, "micro_batch_size": 1,
                  "global_batch_size": steps,
                  "optim": {"lr": 1e-8, "value_lr": 1e-8, "adam_beta1": 0.9,
                            "adam_beta2": 0.95, "adam_eps": 1e-8,
                            "weight_decay": 0.01, "clip_grad": 1.0,
                            "critic_warmup_steps": 0, "lr_scheduler": "constant"}},
        "rollout": {"group_name": "RolloutGroup", "backend": "huggingface",
                    "pipeline_stage_num": 1, "enable_offload": True,
                    "model": {"model_path": str(model_path.resolve()), "precision": "bf16"}},
        "env": {"group_name": "EnvGroup", "train": env, "eval": env},
        "algorithm": {"adv_type": "gae", "loss_type": "actor_critic",
                      "reward_type": "chunk_level", "logprob_type": "chunk_level",
                      "entropy_type": "chunk_level", "group_size": 1,
                      "normalize_advantages": True, "loss_agg_func": "token-mean",
                      "update_epoch": 1, "gamma": 0.99, "gae_lambda": 0.95,
                      "clip_ratio_high": 0.2, "clip_ratio_low": 0.2,
                      "value_clip": 0.2, "huber_delta": 10.0, "entropy_bonus": 0.0},
        "weight_syncer": {"type": "bucket", "transport": "ray_cpu", "bucket": {
            "bucket_size": 32 * 1024 * 1024, "bucket_dtype": None,
            "bucket_device": "cpu", "load_instant": True, "is_agent": False}},
    })
