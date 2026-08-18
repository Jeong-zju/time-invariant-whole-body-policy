"""Launch a matched multi-task B0/B1/B2 GR00T N1.6 fine-tune."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

from gr00t.configs.base_config import get_default_config
from gr00t.data.dataset import factory as dataset_factory
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment import experiment as experiment_module

from lpwb_dataset import selected_dataset_class


def load_module(path: str) -> None:
    spec = importlib.util.spec_from_file_location("lpwb_modality_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--dataset-path", action="append", required=True)
    parser.add_argument("--modality-config-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--method", choices=["b0", "b1", "b2"], required=True)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=25)
    parser.add_argument("--save-only-model", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--num-shards-per-epoch", type=int, default=100000)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--use-wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.save_only_model:
        training_arguments_class = experiment_module.TrainingArguments
        original_init = training_arguments_class.__init__

        def initialize_save_only(self, *training_args, **training_kwargs):
            training_kwargs["save_only_model"] = True
            original_init(self, *training_args, **training_kwargs)

        # Keep the original Transformers class identity so training_args.bin is
        # pickleable; only inject the missing constructor argument.
        training_arguments_class.__init__ = initialize_save_only
    os.environ["LPWB_METHOD"] = args.method
    load_module(args.modality_config_path)
    dataset_factory.ShardedSingleStepDataset = selected_dataset_class()
    # All LPWB action groups are explicitly absolute; relative statistics are unused.
    dataset_factory.generate_rel_stats = lambda *unused_args, **unused_kwargs: None
    embodiment = EmbodimentTag.NEW_EMBODIMENT.value
    datasets = [
        {"dataset_paths": [path], "mix_ratio": 1.0, "embodiment_tag": embodiment}
        for path in args.dataset_path
    ]
    config = get_default_config().load_dict(
        {"data": {"download_cache": False, "datasets": datasets}}
    )
    config.load_config_path = None
    config.model.tune_llm = False
    config.model.tune_visual = False
    config.model.tune_projector = True
    config.model.tune_diffusion_model = True
    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.eagle_collator = True
    config.model.model_name = "nvidia/Eagle-Block2A-2B-v2"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True
    # The downloaded checkpoint head is 50 steps. Data uses a 32-step mask inside
    # that existing head; this avoids resizing or randomly initializing weights.
    config.model.action_horizon = 50

    training = config.training
    training.experiment_name = args.experiment_name
    training.start_from_checkpoint = args.base_model_path
    training.optim = "adamw_torch"
    if args.global_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError("effective global batch must be divisible by accumulation steps")
    forward_global_batch = args.global_batch_size // args.gradient_accumulation_steps
    if forward_global_batch % args.num_gpus != 0:
        raise ValueError(
            "per-forward global batch must be divisible by the number of GPUs"
        )
    # GR00T's field is the per-forward global micro batch; TrainingArguments then
    # applies accumulation. For example, 32 / 8 = 4 globally = 2 samples/GPU
    # with two GPUs, while 32 / 16 = 2 globally = 1 sample/GPU.
    training.global_batch_size = forward_global_batch
    training.dataloader_num_workers = args.dataloader_num_workers
    training.learning_rate = args.learning_rate
    training.gradient_accumulation_steps = args.gradient_accumulation_steps
    training.output_dir = args.output_dir
    training.save_steps = args.save_steps
    training.save_total_limit = args.save_total_limit
    training.num_gpus = args.num_gpus
    training.use_wandb = args.use_wandb
    training.max_steps = args.max_steps
    training.weight_decay = args.weight_decay
    training.warmup_ratio = args.warmup_ratio

    config.data.shard_size = args.shard_size
    config.data.episode_sampling_rate = 1.0
    config.data.num_shards_per_epoch = args.num_shards_per_epoch
    config.data.ds_weights_alpha = None
    config.data.override_pretraining_statistics = True
    run_dir = Path(args.output_dir) / args.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "lpwb_launch.json").open("w") as file:
        launch_record = vars(args) | {
            "forward_global_batch_size": forward_global_batch,
            "per_gpu_micro_batch_size": forward_global_batch // args.num_gpus,
            "optimizer_samples": args.global_batch_size * args.max_steps,
        }
        json.dump(launch_record, file, indent=2)
    experiment_module.run(config)


if __name__ == "__main__":
    main()
