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

import argparse
import warnings
import numpy as np
import torch
# import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import MHMPolicy
from gr00t.utils.util import default_collate_fn, mse, glance_dict
from mhmp_vis.dataset_loader import *
from mhmp_vis.utils.rot import rot6d_to_rotmat, rotmat_dist

warnings.simplefilter("ignore", category=FutureWarning)

def cal_metrics(pred, true):
    assert all(isinstance(pred[key], type(true[key])) for key in pred.keys()), "The keys of pred and true must be the same."
    pred = {k: torch.as_tensor(v).cuda() for k, v in pred.items()}
    true = {k: torch.as_tensor(v).cuda() for k, v in true.items()}
    '''
    pred = {
        action.theta: torch.Size([B, T, 90])
        action.rot: torch.Size([B, T, 6])
        action.trans: torch.Size([B, T, 3])  
    }     
    gt = {
        action.theta: torch.Size([B, T, 90])
        action.rot: torch.Size([B, T, 6])
        action.trans: torch.Size([B, T, 3])  
    }
    '''
    # B, T = true['action.trans'].shape[:2]
    return {
        'action.theta': rotmat_dist(rot6d_to_rotmat(pred['action.theta'].reshape(-1,6)),rot6d_to_rotmat(true['action.theta'].reshape(-1,6))).mean(),
        'action.rot': rotmat_dist(rot6d_to_rotmat(pred['action.rot'].reshape(-1,6)),rot6d_to_rotmat(true['action.rot'].reshape(-1,6))).mean(),
        'action.trans': mse(pred['action.trans'],true['action.trans']),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_list", type=str, required=True, help="dataset string split by comma, e.g. 'oakink2,hoi4d'") 
    parser.add_argument("--dataset_path", type=str, default="/mnt/vepfs/zbh/MHMP/datasets/")
    parser.add_argument("--data_config", type=str, default="hand_motion", choices=list(DATA_CONFIG_MAP.keys()),help="data config name",)
    parser.add_argument("--split_portion", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--pretrain_ckpt", type=str, default=None, help="[Optional] Path to the model checkpoint directory, this will disable client server mode.",)
    parser.add_argument("--debug", type=int, default=0)
    args = parser.parse_args()
    # model
    data_config = DATA_CONFIG_MAP[args.data_config]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform(split='test')
    policy = MHMPolicy(
        model_cfg_path='model_config',
        modality_config=modality_config,
        modality_transform=modality_transform,
        pretrain_ckpt=args.pretrain_ckpt,
    )
    data_storage_kwargs = dict(window_size=data_config.action_horizon, side=data_config.side, is_render_depth=data_config.is_render_depth)
    cascaded_dataset = load_cascaded_dataset(
        dataset_path=args.dataset_path,
        dataset_list=args.dataset_list.split(','),
        split_portion=args.split_portion,
        data_config=data_config,
        data_storage_kwargs=data_storage_kwargs,
        test=True,
        debug=args.debug,
    )
    test_dataloader = DataLoader(cascaded_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False, collate_fn=default_collate_fn)
    
    avg_metrics = {
        'action.theta': [],
        'action.rot': [],
        'action.trans': [],
    }
    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Testing", unit="batch")
    for i, step_data in pbar:
        # inference
        gt_action = {k: v for k, v in step_data.items() if k.startswith("action.")}
        pred_action = policy.get_action(step_data)
        '''
        absolute_future_hand_rotmat, absolute_future_hand_trans = decode_absolute_pose_from_relative(
            torch.Tensor(self.storage["motion_sequences"]['rot'][frame_id:frame_id+1]),
            torch.Tensor(self.storage["motion_sequences"]['trans'][frame_id:frame_id+1]),
            future_hand_rotmat, future_hand_trans
        )
        assert torch.allclose(absolute_future_hand_rotmat, torch.Tensor(self.storage["motion_sequences"]['rot'][frame_id+1:frame_id+self.window_size+1]))
        assert torch.allclose(absolute_future_hand_trans, torch.Tensor(self.storage["motion_sequences"]['trans'][frame_id+1:frame_id+self.window_size+1]))
        '''
        batch_avg_metrics = cal_metrics(pred_action, gt_action)
        pbar.set_postfix(batch_avg_metrics)
        for k, v in batch_avg_metrics.items():
            avg_metrics[k].append(v.item())
        if i % 10 == 0:
            avg_mean_metrics = {k: np.mean(v) for k, v in avg_metrics.items()}
            print(f"{i+1} Batch Average: {avg_mean_metrics}")
