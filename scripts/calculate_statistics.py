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
from dataclasses import dataclass, field
from pathlib import Path

import torch
import tyro
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from mhmp_vis.dataset_loader import *


@dataclass
class Config:
    # customize
    depthanything_path: str = "/mnt/vepfs/zbh/backup/pretrain/depth_anything_v2/"
    dataset_list: list = field(default_factory=list)
    exp_name: str = "runs"
    incorporate_depth: int = 0

    # Dataset parameters
    dataset_path: str = '/mnt/vepfs/zbh/MHMP/datasets/'
    """Path to the dataset directory."""

    output_dir: str = "logs"
    """Directory to save model checkpoints."""

    data_config: str = "hand_motion"
    """Data configuration name from DATA_CONFIG_MAP."""

    # Training parameters
    batch_size: int = 8
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
    data_config = DATA_CONFIG_MAP[config.data_config]
    
    statistics_dataset_list = []
    data_storage_kwargs = dict(window_size=data_config.action_horizon, side=data_config.side, is_render_depth=data_config.is_render_depth)
    for each in config.dataset_list:
        dataset_name = each.upper()
        dataset = eval(f"{dataset_name}Dataset")(os.path.join(config.dataset_path, dataset_name))
        # TODO: filter TACO task list
        dataset.initialize_data_storage(eval(f"{dataset_name}Storage"), data_storage_kwargs, split=dict(train=1.0))
        statistics_dataset_list.append(dataset)
    
    cascaded_dataset = CascadedDataset(statistics_dataset_list)
    statistics = cascaded_dataset.calculate_statistics(num_workers=8)

    metadata_json = 'model_config/metadata.json'
    with open(metadata_json, "r") as f:
        metadata = json.load(f)
    # update
    metadata['statistics'] = statistics
    with open(metadata_json, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Statistics calculated and saved to {metadata_json}")
    breakpoint()
    

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
