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
import torch
from torch import nn
from torch.cuda.amp import autocast
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature
import torchvision.models as models
from einops import repeat, rearrange
import gr00t
from gr00t.utils.util import clip_depth, resize, l1, smooth_l1, mse, minmaxscale
from gr00t.model.transforms import EAGLE_KEYS
from .eagle2_hg_model.inference_eagle_repo import EagleProcessor, ModelSpecificValues, reshape_model_embeddings
from .depth_anything_v2.dpt import DepthAnythingV2
from .unidepth.models import UniDepthV2
from .unidepth.ops.losses import SILog, EdgeGuidedLocalSSI, Confidence

DEFAULT_EAGLE_MODEL_NAME = os.path.join(os.path.dirname(gr00t.__file__), "model", "backbone", "eagle2_hg_model")


class UnidepthLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.silog_loss = SILog()
        # self.edge_loss = EdgeGuidedLocalSSI()
        self.conf_loss = Confidence()

    def forward(self, depth, depth_gt, **kwargs):
        l1_loss = l1(depth, depth_gt)
        depth_mask = torch.ones_like(depth, dtype=torch.bool)   
        silog_loss = self.silog_loss(depth, target=depth_gt, mask=depth_mask)
        # edge_loss = self.edge_loss(depth, target=depth_gt,mask=depth_mask)
        if "confidence" in kwargs:
            conf_loss = self.conf_loss(kwargs["confidence"], target_pred=depth, target_gt=depth_gt,mask=depth_mask)
        else:
            conf_loss = 0
        return l1_loss + silog_loss + conf_loss * 0.1  # + edge_loss

class DepthEncoder(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.net = nn.Sequential(
            *list(resnet.children())[:-2],
            nn.Flatten(-2)
        )
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

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
    multimodal='tv'
) -> torch.LongTensor:
    ## get visual embeddings from pixel, if provided then overwrite
    assert pixel_values is not None and self.img_context_token_id is not None
    pixel_values, input_ids, attention_mask = pixel_values.squeeze(1), input_ids.squeeze(1), attention_mask.squeeze(1)  # avoid additional batch dim
    
    if 'tv' in multimodal:
        # 1. get language input embeddings
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        # 2. overwrite visual embeddings
        with autocast(dtype=torch.bfloat16):
            vit_embeds = self.extract_feature(pixel_values)  # [B, 3, 224, 224] -> [B, 64, 2048]
        # pooling visual feature and depth feature
        visual_embeds = (depth_feature + vit_embeds) if depth_feature is not None else vit_embeds  # [B, 64, 2048]
        embeds_to_scatter = visual_embeds.reshape(-1, C).to(input_embeds.device, input_embeds.dtype)
        selected = input_ids.reshape(B * N) == self.img_context_token_id
        input_embeds[selected] = embeds_to_scatter
        input_embeds = input_embeds.reshape(B, N, C)
    elif 't' in multimodal:
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
    elif 'v' in multimodal:
        with autocast(dtype=torch.bfloat16):
            vit_embeds = self.extract_feature(pixel_values)  # [B, 3, 224, 224] -> [B, 64, 2048]
        input_embeds = (depth_feature + vit_embeds) if depth_feature is not None else vit_embeds  # [B, 64, 2048]
        attention_mask = torch.ones(input_embeds.shape[0], input_embeds.shape[1], device=input_embeds.device, dtype=torch.int64)
    else:
        raise ValueError(f"Invalid multimodal mode: {multimodal}")

    ## vllm forward
    with autocast(dtype=torch.bfloat16):
        embeddings = self.language_model.forward(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        ).hidden_states[-1]

    if reproject_vision:  # use the original visual embeddings instead of VLM output embeddings
        embeddings = embeddings.reshape(B * N, C)
        embeddings[selected] = embeds_to_scatter
        embeddings = embeddings.reshape(B, N, C)
    return embeddings


class EagleBackbone(nn.Module):
    valid_input_keys = EAGLE_KEYS + ["video.depthmodel"]
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
        depth_model_path: str = "",  
        depth_mode: str = "abs",
        multimodal: str = 'tv',
    ):
        super().__init__()
        # load local eagle model and set property
        if use_local_eagle_hg_model:
            model_name = DEFAULT_EAGLE_MODEL_NAME
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_config(config, trust_remote_code=True)
        self.model.neftune_alpha = None
        self.reproject_vision = reproject_vision
        self.incorporate_depth = os.path.exists(depth_model_path)
        self.multimodal = multimodal
        self.depth_mode = depth_mode
        # self.unidepth_loss = UnidepthLoss()
        self.emb_loss = mse
        
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        
        # cutomize: depth-anything model
        if self.incorporate_depth:
            if "depth_anything" in depth_model_path:
                depth_anything_model_configs = {
                    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
                }
                depth_anything_encoder_type = 'vits'
                self.depth_model = DepthAnythingV2(**depth_anything_model_configs[depth_anything_encoder_type])
                ckpt_path = os.path.join(depth_model_path, f'depth_anything_v2_{depth_anything_encoder_type}.pth')
            elif "unidepth" in depth_model_path:
                version = 'v2'
                backbone = 'vits14'
                name = f"unidepth-{version}-{backbone}"
                current_dir = os.path.dirname(os.path.abspath(__file__))
                with open(os.path.join(current_dir, "unidepth/configs", f"config_{version}_{backbone}.json")) as f:
                    config = json.load(f)
                self.depth_model = UniDepthV2(config)
                ckpt_path = os.path.join(depth_model_path, f'models--lpiccinelli--{name}/pytorch_model.bin')
            else:
                raise ValueError(f"Invalid depth model path: {depth_model_path}")
            # load pretrained checkpoint
            info = self.depth_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
            print(f"\t missing keys: {info.missing_keys}")
            print(f"\t additional keys: {info.unexpected_keys}")
            self.depth_model.to(device)
            # customize: freeze depth_model
            # for module in self.depth_model.modules():
            #     module.eval()
            # for param in self.depth_model.parameters():
            #     param.requires_grad = False
        else:
            self.depth_model = None

        # cutomize: depth encoder
        self.depth_encoder = DepthEncoder(device=device).to(device)

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

    def prepare_input(self, batch: dict):# -> BatchFeature
        for key in self.valid_input_keys:
            if key not in batch:
                raise ValueError(f"Missing required input key: {key}")
        # return BatchFeature(data=batch)
        return batch

    def forward(self, vl_input):#: BatchFeature -> BatchFeature
        '''
        vl_input["pixel_values"]: [B, 3, H, W]
        vl_input["input_ids"]: [B, T1] language token ids
        vl_input["attention_mask"]: [B, T1] all 1
        '''
        vlm_output = dict()
        # Set frozen module to eval
        self.set_frozen_modules_to_eval_mode()
        if self.incorporate_depth:
            depth_true = vl_input['depth.ego_view'].squeeze(1).detach()    # (B, 1, H, W)
            # depth estimation
            depth_pred = self.depth_model.forward_depth_model(vl_input["video.depthmodel"].squeeze(1)).detach() # (B, 1, H, W)
            vlm_output["depth_pred"] = depth_pred  # for test visualization
            if "abs" in self.depth_mode:  # clip + resize
                depth_pred = resize(clip_depth(depth_pred))
            elif "rel" in self.depth_mode:  # 2 * (x - x.min()) / (x.max() - x.min()) - 1
                depth_pred = minmaxscale(resize(depth_pred))
            else:
                raise ValueError(f"Invalid depth mode: {self.depth_mode}")
            '''
            # visualization
            import matplotlib.pyplot as plt
            import torch
            torch.save([depth_pred, depth_true], "depth.pt")
            plt.imshow(torch.cat([depth_pred, depth_true], dim=-1).detach().squeeze().cpu().numpy(), cmap='viridis')
            plt.axis('off')
            plt.savefig("depth_cmp.png")
            breakpoint()
            '''

            '''
            # filter absent / bad depth inputs            
            depth_mask = depth_true[...,0,0].view(-1) >= 0  # depth_mask: 0 indicates abnormal
            depth_feature = torch.empty(len(depth_mask), 64, 2048).to(device=depth_pred.device, dtype=torch.bfloat16)  # (B, 64, 2048)
            depth_feature[depth_mask] = self.depth_encoder(depth_true[depth_mask])
            depth_feature[~depth_mask] = self.depth_encoder(depth_pred[~depth_mask])
            vlm_output["depth_loss"] = l1(depth_pred[depth_mask], depth_true[depth_mask])
            '''
            depth_feature = self.depth_encoder(depth_pred)
            depth_feature_true = self.depth_encoder(depth_true)
            # vlm_output["depth_loss"] = self.unidepth_loss(depth_pred, depth_true)
            vlm_output["depth_feature_loss"] = self.emb_loss(depth_feature, depth_feature_true)  # mse
        else:
            depth_feature = None
        embeddings = get_embeddings(
            self.model,
            self.reproject_vision,
            **{k:v for k,v in vl_input.items() if k in EAGLE_KEYS},
            depth_feature=depth_feature,
            multimodal=self.multimodal,
        )
        vlm_output.update({
            "backbone_features": self.linear(embeddings),           # [B, T2, hidden_size]
            "backbone_attention_mask": vl_input["attention_mask"],
        })
        
        # return BatchFeature(data=vlm_output) 
        return vlm_output
