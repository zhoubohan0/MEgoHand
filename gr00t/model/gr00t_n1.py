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

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import torch
import tree
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
from gr00t.utils.util import download_from_hg
from .action_head.flow_matching_action_head import FlowmatchingActionHead, FlowmatchingActionHeadConfig
from .backbone import EagleBackbone

BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3


# config
@dataclass
class GR00T_N1Config(PretrainedConfig):
    model_type = "gr00t_n1"
    backbone_cfg: dict = field(init=False, metadata={"help": "Backbone configuration."})

    action_head_cfg: dict = field(init=False, metadata={"help": "Action head configuration."})

    action_horizon: int = field(init=False, metadata={"help": "Action horizon."})

    action_dim: int = field(init=False, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


# real model
class GR00T_N1(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = GR00T_N1Config
    """
    we expect the backbone output to have a key 'backbone_features' with shape (batch_size, n, hidden_size)
    here n is variable and can be e.g. time, 1 or user specified
    we expect the action head output to have a key 'action_pred' with shape (batch_size, time, action_dim) during inference time
    we expect these to have type BatchFeature, and they can of course have many other user specified keys too
    """

    def __init__(
        self,
        config: GR00T_N1Config,
        local_model_path: str,
    ):
        '''
        EagleBackbone(backbone_cfg)
        "backbone_cfg": {
            "allow_reshape_visual": true,
            "model_name": "$GR00T_BACKBONE_PATH/eagle2_hg_model",
            "processor_cfg": {
            "max_input_tiles": 1,
            "model_path": "$GR00T_BACKBONE_PATH/eagle2_hg_model",
            "model_spec": {
                "num_image_token": 64,
                "template": "qwen2-chat"
            }
            },
            "projector_dim": 2048,
            "reproject_vision": false,
            "scale_image_resolution": 1,
            "select_layer": 12,
            "tune_llm": false,
            "tune_visual": true
        }

        FlowmatchingActionHead(action_head_cfg)
        "action_head_cfg": {
            "action_dim": 32,
            "action_horizon": 16,
            "add_pos_embed": true,
            "diffusion_model_cfg": {
            "attention_head_dim": 48,
            "dropout": 0.2,
            "final_dropout": true,
            "interleave_self_attention": true,
            "norm_type": "ada_norm",
            "num_attention_heads": 32,
            "num_layers": 16,
            "output_dim": 1024,
            "positional_embeddings": null
            },
            "freeze_decode_layer": false,
            "hidden_size": 1024,
            "input_embedding_dim": 1536,
            "load_pretrained_det_decode_layer_path": null,
            "max_action_dim": 32,
            "max_state_dim": 64,
            "model_dtype": "float32",
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_inference_timesteps": 16,
            "num_timestep_buckets": 1000,
            "tune_diffusion_model": true,
            "tune_projector": true
        }
        '''
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.action_head_cfg, dict)
        super().__init__(config)
        self.local_model_path = local_model_path
        # customize
        config.backbone_cfg['depthanything_path'] = config.depthanything_path
        config.backbone_cfg['multimodal'] = config.multimodal
        self.backbone = EagleBackbone(**config.backbone_cfg)
        self.action_head = FlowmatchingActionHead(FlowmatchingActionHeadConfig(**config.action_head_cfg))
        # seem unnecessary
        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype

    def validate_inputs(self, inputs):
        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs:
            action = inputs["action"]
            type_ok = isinstance(action, torch.Tensor)
            shape_ok = (
                len(action.shape) == 3
                and action.shape[1] == self.action_horizon
                and action.shape[2] == self.action_dim
            )
            if not type_ok:
                error_msg += f"\n{action.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{action.shape=}"
                detected_error = True

        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f"\n{type(video)=}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\n{video.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{video.shape=}"
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):
        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature)
            or BACKBONE_FEATURE_KEY not in backbone_outputs
        )

        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(backbone_outputs, BatchFeature)=}"
            error_msg += f"\n{BACKBONE_FEATURE_KEY in backbone_outputs=}"
            error_msg += f"\n{backbone_outputs[BACKBONE_FEATURE_KEY].shape=}"
            raise ValueError(error_msg)

        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (
                LOSS_KEY in action_head_outputs and is_training
            )  # there might not be an action prediction during training
            or (
                ACTION_KEY in action_head_outputs
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == self.action_dim
            )
        )

        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(action_head_outputs, BatchFeature)=}"
            error_msg += f"\n{LOSS_KEY in action_head_outputs=}"
            error_msg += f"\n{action_head_outputs[ACTION_KEY].shape=}"
            error_msg += f"\n{self.action_horizon=}"
            error_msg += f"\n{self.action_dim=}"
            raise ValueError(error_msg)

    def forward(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head(backbone_outputs, action_inputs)
        action_head_outputs.update({k:v for k,v in backbone_outputs.items() if 'loss' in k})
        # self.validate_data(action_head_outputs, backbone_outputs, is_training=True)
        return action_head_outputs

    def get_action(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)  # repeat
        '''
        backbone_inputs:
        - pixel_values torch.bfloat16
        - input_ids torch.int64
        - attention_mask torch.int64
        - (video.depthanything torch.float32)

        action_inputs:
        - state torch.float32
        '''
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)  # action_head_outputs['action_pred']: (B, T, Da)
        # self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:
        self.validate_inputs(inputs)
        backbone_inputs = self.backbone.prepare_input(inputs)  
        '''
        backbone_inputs:
        - pixel_values torch.bfloat16
        - input_ids torch.int64
        - attention_mask torch.int64
        - (video.depthanything torch.float32)
        backbone_outputs:
        - backbone_outputs['backbone_features']: [1, 99, 1536]
        - backbone_outputs['backbone_attention_mask']: [1, 99]
        '''
        action_inputs = self.action_head.prepare_input(inputs) 
        '''
        action_inputs:
        - state torch.float32
        - action torch.float32
        - action_mask torch.bool
        '''

        # to device and dtype
        '''
        to_device_with_maybe_dtype = lambda x: x.to(self.device, dtype=self.action_head.dtype) if torch.is_floating_point(x) else x.to(self.device)
        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        '''
        return backbone_inputs, action_inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        tune_visual = kwargs.pop("tune_visual", True)
        tune_llm = kwargs.pop("tune_llm", False)
        tune_projector = kwargs.pop("tune_projector", True)
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", True)

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")

        # get the current model path being downloaded
        try:
            # NOTE(YL) This downloads the model to the local cache and returns the local path to the model saved in ~/.cache/huggingface/hub/
            local_model_path = download_from_hg(pretrained_model_name_or_path, repo_type="model")
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {pretrained_model_name_or_path}")
            local_model_path = pretrained_model_name_or_path
        pretrained_model = super().from_pretrained(local_model_path, local_model_path=local_model_path, **kwargs)
        pretrained_model.backbone.set_trainable_parameters(tune_visual=tune_visual, tune_llm=tune_llm)
        pretrained_model.action_head.set_trainable_parameters(tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model)
        return pretrained_model


# register
AutoConfig.register("gr00t_n1", GR00T_N1Config)
AutoModel.register(GR00T_N1Config, GR00T_N1)
