# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from transformers import TrainingArguments

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag, DatasetMetadata
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1 import GR00T_N1, GR00T_N1Config
from gr00t.utils.peft import get_lora_model
from dataset_loader import *


@dataclass
class Config:
    """Configuration for GR00T model fine-tuning."""

    # Dataset parameters
    dataset_path: str = 'demo_data/robot_sim.PickNPlace'
    """Path to the dataset directory."""

    output_dir: str = "logs"
    """Directory to save model checkpoints."""

    data_config: str = "hand_motion"
    """Data configuration name from DATA_CONFIG_MAP."""

    # Training parameters
    batch_size: int = 1
    """Batch size per GPU for training."""

    max_steps: int = 10000
    """Maximum number of training steps."""

    num_gpus: int = 1
    """Number of GPUs to use for training."""

    save_steps: int = 500
    """Number of steps between saving checkpoints."""

    # Model parameters
    model_cfg_path: str = "model_config"
    """Path or HuggingFace model ID for the base model."""

    tune_llm: bool = False
    """Whether to fine-tune the language model backbone."""

    tune_visual: bool = True
    """Whether to fine-tune the vision tower."""

    tune_projector: bool = True
    """Whether to fine-tune the projector."""

    tune_diffusion_model: bool = True
    """Whether to fine-tune the diffusion model."""

    resume: bool = False
    """Whether to resume from a checkpoint."""

    # Advanced training parameters
    learning_rate: float = 1e-4
    """Learning rate for training."""

    weight_decay: float = 1e-5
    """Weight decay for AdamW optimizer."""

    warmup_ratio: float = 0.05
    """Ratio of total training steps used for warmup."""

    lora_rank: int = 0
    """Rank for the LORA model."""

    lora_alpha: int = 16
    """Alpha value for the LORA model."""

    lora_dropout: float = 0.1
    """Dropout rate for the LORA model."""

    dataloader_num_workers: int = 8
    """Number of workers for data loading."""

    report_to: str = "tensorboard"
    """Where to report training metrics (e.g., 'wandb', 'tensorboard')."""

    # Data loading parameters
    embodiment_tag: str = "gr1"
    """Embodiment tag to use for training. e.g. 'new_embodiment', 'gr1'"""

    video_backend: str = "decord"
    """Video backend to use for training. [decord, torchvision_av]"""


#####################################################################################
# main training function
#####################################################################################


def main(config: Config):
    """Main training function."""
    # ------------ step 1: load dataset ------------
    data_config = DATA_CONFIG_MAP[config.data_config]
    modality_configs = data_config.modality_config()  # TODO
    '''
    train_dataset = LeRobotSingleDataset(
        dataset_path=config.dataset_path,
        modality_configs=modality_configs,
        transforms=transforms,
        video_backend=config.video_backend,
    )
    '''
    train_dataset_name_list = [
        "hoi4d",  # zhan
        # "oakink2",  # zhan_4t 
    ]
    train_dataset_list, eval_dataset_list = [], []
    for each in train_dataset_name_list:
        train_dataset_name = each.upper()
        train_dataset = eval(f"{train_dataset_name}Dataset")()
        train_dataset.initialize_data_storage(eval(f"{train_dataset_name}Storage"), dict(window_size=data_config.action_horizon, side=data_config.side), split=dict(train=0.95))
        train_dataset_list.append(train_dataset)
        eval_dataset = eval(f"{train_dataset_name}Dataset")()
        eval_dataset.initialize_data_storage(eval(f"{train_dataset_name}Storage"), dict(window_size=data_config.action_horizon, side=data_config.side), split=dict(eval=0.95))
        eval_dataset_list.append(eval_dataset)
    transform_meta = DatasetMetadata.model_validate(json.load(open('model_config/metadata.json', "r")))  # TODO
    train_transforms = data_config.transform(split='train')
    train_transforms.set_metadata(transform_meta)
    train_dataset = CascadedDatasetWithTransform(train_dataset_list, train_transforms)
    eval_transforms = data_config.transform(split='eval')
    eval_transforms.set_metadata(transform_meta)
    eval_dataset = CascadedDatasetWithTransform(eval_dataset_list, eval_transforms)
    # ------------ step 2: load model ------------
    with open(Path(config.model_cfg_path) / "config.json", "r") as f:
        model_config = json.load(f)
    model = GR00T_N1(GR00T_N1Config(**model_config),"")
    model.backbone.set_trainable_parameters(tune_visual=config.tune_visual, tune_llm=config.tune_llm)
    model.action_head.set_trainable_parameters(tune_projector=config.tune_projector, tune_diffusion_model=config.tune_diffusion_model)
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

    if config.lora_rank > 0:
        model = get_lora_model(
            model,
            rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        )

    # 2.1 modify training args
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=None,
        remove_unused_columns=False,
        deepspeed="",
        gradient_checkpointing=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=1,
        dataloader_pin_memory=True,
        # TODO
        dataloader_num_workers=0, #config.dataloader_num_workers,
        dataloader_persistent_workers=False,#True,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10.0,
        num_train_epochs=300,
        max_steps=config.max_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=8,
        report_to=config.report_to,
        seed=42,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
        # eval
        do_eval=True,
        evaluation_strategy="steps",   # epoch
        eval_steps=config.save_steps,  # when evaluation_strategy is "steps"
        per_device_eval_batch_size=8,
    )

    # 2.2 run experiment
    experiment = TrainRunner(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
    )

    # 2.3 run experiment
    experiment.train()


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(Config)

    # Print the tyro config
    print("\n" + "=" * 50)
    print("GR00T FINE-TUNING CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")


    # Validate GPU configuration
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    assert config.num_gpus <= available_gpus, f"Number of GPUs requested ({config.num_gpus}) is greater than the available GPUs ({available_gpus})"
    assert config.num_gpus > 0, "Number of GPUs must be greater than 0"
    print(f"Using {config.num_gpus} GPUs")

    if config.num_gpus == 1:
        # Single GPU mode - set CUDA_VISIBLE_DEVICES=0
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # Run the script normally
        main(config)
    else:
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(config)
        else:
            # Multi-GPU mode - use torchrun
            script_path = Path(__file__).absolute()
            # Remove any existing CUDA_VISIBLE_DEVICES from environment
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

            # Use subprocess.run instead of os.system
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                "--nnodes=1",  # default to 1 node for now
                str(script_path),
            ]

            # Convert config to command line arguments
            for key, value in vars(config).items():
                if isinstance(value, bool):
                    # For boolean values, use --flag or --no-flag format
                    if value:
                        cmd.append(f"--{key.replace('_', '-')}")
                    else:
                        cmd.append(f"--no-{key.replace('_', '-')}")
                else:
                    # For non-boolean values, use --key value format
                    cmd.append(f"--{key.replace('_', '-')}")
                    cmd.append(str(value))
            print("Running torchrun command: ", cmd)
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)
