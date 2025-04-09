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

"""
python scripts/test.py
    --action_horizon 16
    --dataset_path demo_data/robot_sim.PickNPlace/
    --data_config hand_motion
provide --pretrain_ckpt to load up the model checkpoint in this script.
"""

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
    parser.add_argument("--data_config", type=str, default="hand_motion", choices=list(DATA_CONFIG_MAP.keys()),help="data config name",)
    parser.add_argument("--action_horizon", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--pretrain_ckpt", type=str, default=None, help="[Optional] Path to the model checkpoint directory, this will disable client server mode.",)
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
    # test dataset
    test_dataset_name_list = [
        "hoi4d",  # zhan
        # "oakink2",  # zhan_4t 
    ]
    test_dataset_list = []
    for each in test_dataset_name_list:
        dataset_name = each.upper()
        dataset = eval(f"{dataset_name}Dataset")()
        dataset.initialize_data_storage(eval(f"{dataset_name}Storage"), dict(window_size=data_config.action_horizon, side='right'), split=dict(train=1.0))
        test_dataset_list.append(dataset)
    cascaded_dataset = CascadedDataset(test_dataset_list)
    test_dataloader = DataLoader(cascaded_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False, collate_fn=default_collate_fn)
    print("Test dataset:", [str(each) for each in test_dataset_name_list])
    print("Total test samples:", len(cascaded_dataset))
    
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
        batch_avg_metrics = cal_metrics(pred_action, gt_action)
        pbar.set_postfix(batch_avg_metrics)
        for k, v in batch_avg_metrics.items():
            avg_metrics[k].append(v.item())
        if i % 10 == 0:
            avg_mean_metrics = {k: np.mean(v) for k, v in avg_metrics.items()}
            print(f"{i+1} Batch Average: {avg_mean_metrics}")
