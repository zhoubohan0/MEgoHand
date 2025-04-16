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

import os
import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature
import torchvision.models as models
from einops import repeat, rearrange

import gr00t
from gr00t.utils.util import depth_to1, resize, mse, minmaxscale
from gr00t.model.transforms import EAGLE_KEYS
from .eagle2_hg_model.inference_eagle_repo import EagleProcessor, ModelSpecificValues, reshape_model_embeddings
from .depth_anything_v2.dpt import DepthAnythingV2
DEFAULT_EAGLE_MODEL_NAME = os.path.join(os.path.dirname(gr00t.__file__), "model", "backbone", "eagle2_hg_model")

class DepthEncoder(nn.Module):
    def __init__(self):
        super(DepthEncoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.net = nn.Sequential(
            *list(resnet.children())[:-2],
            nn.Flatten(-2)
        )
        self.mean, self.std = torch.Tensor([0.485, 0.456, 0.406]).cuda().view(1,3,1,1), torch.Tensor([0.229, 0.224, 0.225]).cuda().view(1,3,1,1)

    def forward(self, x):
        '''
        [B, 1, 256, 256] (0,1500) -> [B, 64, 2048]
        '''
        x = repeat(x, 'B 1 H W -> B 3 H W')
        # normalize
        x = (x - self.mean) / self.std
        return rearrange(self.net(x), 'B L D -> B D L')

def get_embeddings(
    self,
    reproject_vision: bool,
    depth_feature=None,
    pixel_values=None,
    input_ids=None,
    attention_mask=None,
    visual_features=None,
) -> torch.LongTensor:
    ## get visual embeddings from pixel, if provided then overwrite
    assert pixel_values is not None and self.img_context_token_id is not None
    pixel_values, input_ids, attention_mask = pixel_values.squeeze(1), input_ids.squeeze(1), attention_mask.squeeze(1)  # avoid additional batch dim
    vit_embeds = self.extract_feature(pixel_values.float())  # [B, 3, 224, 224] -> [B, 64, 2048]
    # pooling visual feature and depth feature
    visual_embeds = (depth_feature + vit_embeds) if depth_feature is not None else vit_embeds

    ## get language embeddings, then overwrite some with visual embeddings
    # 1. get input embeddings
    input_embeds = self.language_model.get_input_embeddings()(input_ids)
    B, N, C = input_embeds.shape
    input_embeds = input_embeds.reshape(B * N, C)
    # 2. get visual position
    selected = input_ids.reshape(B * N) == self.img_context_token_id
    assert selected.sum() != 0
    # 3. overwrite visual embeddings
    embeds_to_scatter = visual_embeds.reshape(-1, C).to(input_embeds.device, input_embeds.dtype)
    input_embeds[selected] = embeds_to_scatter
    input_embeds = input_embeds.reshape(B, N, C)

    ## vllm forward
    embeddings = self.language_model.forward(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
    ).hidden_states[-1]

    if reproject_vision:  # only use the output of language and overwrite the input visual embeddings
        embeddings = embeddings.reshape(B * N, C)
        embeddings[selected] = embeds_to_scatter
        embeddings = embeddings.reshape(B, N, C)
    return embeddings


class EagleBackbone(nn.Module):
    valid_input_keys = EAGLE_KEYS + ["video.depthanything"]
    def __init__(
        self,
        select_layer: int = 12,
        model_name: str = DEFAULT_EAGLE_MODEL_NAME,
        tune_llm: bool = False,
        tune_visual: bool = False,
        reproject_vision: bool = False,
        scale_image_resolution: int = 1,
        processor_cfg: dict = dict(),
        projector_dim: int = -1,
        allow_reshape_visual: bool = True,
        use_local_eagle_hg_model: bool = True,
        # customize
        depthanything_path: str = "",  
        incorporate_depth: bool = False,
    ):
        super().__init__()
        # load local eagle model and set property
        if use_local_eagle_hg_model:
            model_name = DEFAULT_EAGLE_MODEL_NAME
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_config(config, trust_remote_code=True)
        self.model.neftune_alpha = None
        self.reproject_vision = reproject_vision
        self.incorporate_depth = incorporate_depth

        # cutomize: depth-anything model
        if os.path.exists(depthanything_path):
            depth_anything_model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }
            depth_anything_encoder_type = 'vitl'
            self.depth_anything_model = DepthAnythingV2(**depth_anything_model_configs[depth_anything_encoder_type])
            self.depth_anything_model.load_state_dict(torch.load(os.path.join(depthanything_path, f'depth_anything_v2_{depth_anything_encoder_type}.pth'), map_location='cpu'))
            # TODO: finetune depth anything
            for param in self.depth_anything_model.parameters():
                param.requires_grad = False
        else:
            self.depth_anything_model = None

        # cutomize: depth encoder
        self.depth_encoder = DepthEncoder()

        # initialize processor
        processor = EagleProcessor(
            model_path=processor_cfg["model_path"],
            max_input_tiles=processor_cfg["max_input_tiles"],
            model_spec=ModelSpecificValues(**processor_cfg["model_spec"]),
        )
        self.model.img_context_token_id = processor.get_img_context_token()
        assert self.model.template == processor.model_spec.template
        assert self.model.num_image_token == processor.model_spec.num_image_token

        if allow_reshape_visual and scale_image_resolution != 1:
            reshape_model_embeddings(self.model, scale_image_resolution)

        # language model (LlamaForCausalLM): keep the first 12 layers of LLM and remove lm_head
        self.model.language_model.lm_head = torch.nn.Identity()
        while len(self.model.language_model.model.layers) > select_layer:  
            self.model.language_model.model.layers.pop(-1)

        # vision model (SiglipVisionModel): remove head (of two-layer vision_tower)
        if hasattr(self.model.vision_model, "vision_model") and hasattr(self.model.vision_model.vision_model, "head"):
            self.model.vision_model.vision_model.head = torch.nn.Identity()

        if (
            hasattr(self.model, "vision_model")
            and hasattr(self.model.vision_model, "vision_model")
            and hasattr(self.model.vision_model.vision_model, "vision_towers")
            and len(self.model.vision_model.vision_model.vision_towers) > 1
        ):
            vision_towers = self.model.vision_model.vision_model.vision_towers

            if (
                hasattr(vision_towers[0], "vision_tower")
                and hasattr(vision_towers[0].vision_tower, "vision_model")
                and hasattr(vision_towers[0].vision_tower.vision_model, "encoder")
            ):
                vision_towers[0].vision_tower.vision_model.encoder.gradient_checkpointing = False
                vision_towers[0].vision_tower.vision_model.head = torch.nn.Identity()

            if hasattr(vision_towers[1], "vision_tower"):
                vision_towers[1].vision_tower.head = torch.nn.Identity()

        # last linear mapping for visual language embeddings
        self.linear = torch.nn.Linear(projector_dim, 1536) if projector_dim != -1 else torch.nn.Identity()

        # set is_tune
        self.set_trainable_parameters(tune_llm, tune_visual)

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        if not tune_llm:
            self.model.language_model.requires_grad_(False)
        if not tune_visual:
            self.model.vision_model.requires_grad_(False)
            self.model.mlp1.requires_grad_(False)

    def set_frozen_modules_to_eval_mode(self):
        if self.training:
            if self.model.language_model and not self.tune_llm:
                self.model.language_model.eval()
            if self.model.vision_model and not self.tune_visual:
                self.model.vision_model.eval()
                self.model.mlp1.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        for key in self.valid_input_keys:
            if key not in batch:
                raise ValueError(f"Missing required input key: {key}")
        return BatchFeature(data=batch)

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        '''
        vl_input["pixel_values"]: [B, 3, H, W]
        vl_input["input_ids"]: [B, T1] language token ids
        vl_input["attention_mask"]: [B, T1] all 1
        '''
        # Set frozen module to eval
        self.set_frozen_modules_to_eval_mode()
        if self.incorporate_depth:
            depth_true = vl_input["depth.ego_view"].squeeze(1)  # (B, 1, H, W)
            depth_feature=self.depth_encoder(depth_true)        # (B, 64, 2048), or self.depth_encoder(depth_pred_rescale)
        else:
            depth_feature = None
        embeddings = get_embeddings(
            self.model,
            self.reproject_vision,
            **{k:v for k,v in vl_input.items() if k in EAGLE_KEYS},
            depth_feature=depth_feature,
        )
        vlm_output = {
            "backbone_features": self.linear(embeddings),           # [B, T2, hidden_size]
            "backbone_attention_mask": vl_input["attention_mask"],
        }
        
        if self.depth_anything_model is not None: # depth estimation
            depth_pred = self.depth_anything_model(vl_input["video.depthanything"].squeeze(1))
            depth_pred_rescale = resize(depth_to1(depth_pred))  # clip, to1 (TODO: reletive (x - x.min()) / (x.max() - x.min()))
            vlm_output["depth_loss"] = mse(minmaxscale(depth_pred_rescale), minmaxscale(depth_true))
            '''
            # visualization
            import matplotlib.pyplot as plt
            import torch
            torch.save([depth_pred_rescale, depth_true], "depth.pt")
            plt.imshow(torch.cat([depth_pred_rescale, depth_true], dim=-1).detach().squeeze().cpu().numpy(), cmap='viridis')
            plt.axis('off')
            plt.savefig("depth_cmp.png")
            breakpoint()
            '''
        
        return BatchFeature(data=vlm_output)  
