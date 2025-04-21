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
from einops import repeat
from tqdm import tqdm
from torch.utils.data import DataLoader
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import MHMPolicy
from gr00t.utils.util import default_collate_fn, mse, glance_dict
from mhmp_vis.dataset_loader import *
from mhmp_vis.utils.rot import rot6d_to_rotmat, rotmat_dist, decode_absolute_pose_from_relative, batch_rot2aa, get_pose12d_from_camera_to_world, batch_rodrigues
from mhmp_vis.utils.util import mano_forward
from mhmp_vis.utils.pc import get_pc_from_world_to_camera, get_pc_from_camera_to_world
from mhmp_vis.dataset_visualizer.general_offline_visualizer import offline_render
# from mhmp_vis.dataset_visualizer.general_interactive_visualizer import GeneralInteractiveVisualizer

warnings.simplefilter("ignore", category=FutureWarning)

def cal_metrics(pred, true, return_pred=False):
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
    B, T = true['action.trans'].shape[:2]
    pred_theta_flatten_9d = rot6d_to_rotmat(pred['action.theta'].reshape(-1,6))  # (B*T*15, 3, 3)
    true_theta_flatten_9d = rot6d_to_rotmat(true['action.theta'].reshape(-1,6))  # (B*T*15, 3, 3)
    pred_rot_flatten_9d = rot6d_to_rotmat(pred['action.rot'].reshape(-1,6))    # (B*T, 3, 3)
    true_rot_flatten_9d = rot6d_to_rotmat(true['action.rot'].reshape(-1,6))    # (B*T, 3, 3)
    
    metrics = {
        'action.theta': rotmat_dist(pred_theta_flatten_9d, true_theta_flatten_9d).mean(),
        'action.rot': rotmat_dist(pred_rot_flatten_9d, true_rot_flatten_9d).mean(),
        'action.trans': mse(pred['action.trans'],true['action.trans']),
    }
    if return_pred:
        pred_theta = batch_rot2aa(pred_theta_flatten_9d).view(B, T, 45)    
        pred_theta9d = pred_theta_flatten_9d.reshape(B, T, 15, 3, 3)  
        pred_rot = batch_rot2aa(pred_rot_flatten_9d).view(B, T, 3)
        pred_rot9d = pred_rot_flatten_9d.view(B, T, 3, 3) 
        pred_mano_param = dict(
            theta=pred_theta,           # (B, T, 45)
            theta9d=pred_theta9d,       # (B, T, 15, 3, 3)
            rot=pred_rot,               # (B, T, 3)
            rot9d=pred_rot9d,           # (B, T, 3, 3)
            trans=pred['action.trans']  # (B, T, 3)
        )
        return metrics, pred_mano_param
    else:
        return metrics 

def conduct_testing(cascaded_dataset, policy, batch_size=8192):
    avg_metrics = {
        'action.theta': [],
        'action.rot': [],
        'action.trans': [],
    }
    test_dataloader = DataLoader(cascaded_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False, collate_fn=default_collate_fn)
    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Testing", unit="batch", leave=False, dynamic_ncols=True, position=0)
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


def conduct_visualization(cascaded_dataset, policy, per_vis=-1):
    def truncate_vis_data(vis_data, trunk_len):
        vis_data_copy = vis_data.copy()
        for k in ['image_filenames', 'cam_left_hand_verts', 'cam_right_hand_verts', 'world_left_hand_verts', 'world_right_hand_verts', 'extrinsic_matrix_sequence']:
            vis_data_copy[k] = vis_data_copy[k][:trunk_len]
        for view in ['cam', 'world']:
            for i in range(len(vis_data_copy[f'{view}_object_dict_list'])):
                vis_data_copy[f'{view}_object_dict_list'][i][f'{view}_object_verts'] = vis_data_copy[f'{view}_object_dict_list'][i][f'{view}_object_verts'][:trunk_len]
        # for k in vis_data_copy: print(k, len(vis_data_copy[k]) if vis_data_copy[k] is not None else '')
        return vis_data_copy
    window_size = cascaded_dataset.dataset_list[0].data_storage.window_size
    side = cascaded_dataset.dataset_list[0].data_storage.side
    # get all trajectories index in cascaded_dataset
    all_trajectory_index = dict()
    for dataset_idx in range(len(cascaded_dataset.dataset_list)):
         dataset = cascaded_dataset.dataset_list[dataset_idx]
         all_trajectory_index[str(dataset)] = np.array(dataset.data_storage.valid_trajectory_range) + cascaded_dataset.cumsum_length_list[dataset_idx]
    # Visualize K trajectories for each dataset 
    for dataset_name, trajectory_range in all_trajectory_index.items():
        vis_trajectory_range = trajectory_range[np.random.choice(len(trajectory_range), per_vis, replace=False)] if len(trajectory_range) > per_vis and per_vis!= -1 else trajectory_range
        for i_traj in range(len(vis_trajectory_range)):
            b, e = vis_trajectory_range[i_traj]
            T = e - b
            ## calculate trajectory loss
            step_data = default_collate_fn([cascaded_dataset[_] for _ in range(b,e)])
            gt_action = {k: v for k, v in step_data.items() if k.startswith("action.")}
            pred_action = policy.get_action(step_data)
            batch_avg_metrics, pred_mano = cal_metrics(pred_action, gt_action, return_pred=True)
            '''
            pred_mano
            - theta: (T, W, 45)
            - rot9d: (T, W, 3, 3)
            - trans: (T, W, 3)
            '''
            print(f"Dataset: {dataset_name} | Trajectory: {i_traj} | Frames: {T} | Metrics: {batch_avg_metrics}")
            ## visualize
            dataset_idx, sample_idx = cascaded_dataset.get_dataset_sample_id_by_idx(b)
            dataset = cascaded_dataset.dataset_list[dataset_idx]
            storage_data = default_collate_fn([dataset.index_for_vis(sample_idx+_) for _ in range(T)])
            assert len(np.unique(storage_data['task_name'])) == 1
            task_name = storage_data['task_name'][0][0]
            vis_data = dataset.task_example_for_vis(task_name)
            assert all([x in vis_data['save_video_file'] for x in task_name.split('/')])
            # save original video
            # offline_render(**vis_data)
            vis_data['save_video_file'] = f'_avg'.join(os.path.splitext(vis_data['save_video_file']))
            # overwrite 'cam_{side}_hand_verts', 'world_{side}_hand_verts'
            mano_theta9d = batch_rodrigues(torch.Tensor(storage_data['state.theta']).view(-1,3).cuda()).view(-1,15,3,3)
            mano_beta = torch.Tensor(storage_data['state.beta']).squeeze(1).cuda()
            init_rot = torch.Tensor(storage_data['state.rot']).squeeze(1).cuda()
            init_trans = torch.Tensor(storage_data['state.trans']).squeeze(1).cuda()
            extrinsic_matrix_sequence = torch.Tensor(vis_data['extrinsic_matrix_sequence']).cuda()
            
            # absolute world hand wrist pose
            ref_state_rot_world, ref_state_trans_world = get_pose12d_from_camera_to_world(
                torch.Tensor(storage_data['state.rot'].squeeze(1)).cuda(),
                torch.Tensor(storage_data['state.trans'].squeeze(1)).cuda(),
                extrinsic_matrix_sequence[:T]
            )
                            
            pred_mano_rot9d, pred_mano_trans = [], []
            for t in range(T):
                # relative pose -> absolute cam pose
                pred_mano_rot9d_t, pred_mano_trans_t = decode_absolute_pose_from_relative(  # (W, 3, 3), (W, 3)
                    init_rot[[t]],
                    init_trans[[t]],
                    pred_mano['rot9d'][t], 
                    pred_mano['trans'][t], 
                )
                # absolute cam pose -> absolute world pose
                pred_mano_rot9d_world, pred_mano_trans_world = get_pose12d_from_camera_to_world(  # (W, 3, 3), (W, 3)
                    pred_mano_rot9d_t, pred_mano_trans_t,
                    extrinsic_matrix_sequence[t+1:t+window_size+1],
                )
                pred_mano_rot9d.append(pred_mano_rot9d_world)
                pred_mano_trans.append(pred_mano_trans_world)
            pred_mano_rot9d, pred_mano_trans = torch.stack(pred_mano_rot9d), torch.stack(pred_mano_trans)  # (T, W, 3, 3), (T, W, 3)
            
            # rotation sequence smooth: temporal convolution, followed by an orthogonalisation
            indices = (torch.arange(T).unsqueeze(1) - torch.arange(window_size).unsqueeze(0) + (window_size - 1)).flip(1)
            pred_mano_theta_smooth, pred_mano_rot9d_smooth, pred_mano_trans_smooth = [], [], []
            L = T + window_size - 1
            for t in range(L):
                pred_mano_rot9d_chunk_t = pred_mano_rot9d[indices == t]  # at most (W, 3, 3)
                pred_mano_trans_chunk_t = pred_mano_trans[indices == t]  # at most (W, 3)
                pred_mano_theta_chunk_t = pred_mano['theta9d'][indices == t]  # at most (W, 15, 3, 3)
                chunk_size = len(pred_mano_rot9d_chunk_t)
                '''
                weights = torch.exp(-1.5 * torch.arange(chunk_size, dtype=torch.float32, device=pred_mano_trans_chunk_t.device))
                weights = weights / weights.sum() 
                '''
                weights = torch.ones(chunk_size, device=pred_mano_trans_chunk_t.device) / chunk_size
                U, S, Vh = torch.linalg.svd(torch.sum(pred_mano_rot9d_chunk_t * weights.view(-1,1,1), dim=0))
                pred_mano_rot9d_smooth.append(U @ Vh)  # (3, 3)
                U, S, Vh = torch.linalg.svd(torch.sum(pred_mano_theta_chunk_t * weights.view(-1,1,1,1), dim=0))
                pred_mano_theta_smooth.append(U @ Vh)  # (15, 3, 3)
                pred_mano_trans_smooth.append(torch.sum(pred_mano_trans_chunk_t * weights.view(-1,1), dim=0))  # (3)
                # print(f'[{t}/{L}]')
                # print(f'origin: rot:{mse(ref_state_rot_world[t],pred_mano_rot9d_chunk_t[0]):.6f} | trans:{mse(ref_state_trans_world[t], pred_mano_trans_chunk_t[0]):.6f} | theta:{mse(mano_theta9d[t],pred_mano_theta_chunk_t[0]):.6f}')
                # print(f'smooth: rot:{mse(ref_state_rot_world[t],pred_mano_rot9d_smooth[t]):.6f} | trans:{mse(ref_state_trans_world[t], pred_mano_trans_smooth[t]):.6f} | theta:{mse(mano_theta9d[t],pred_mano_theta_smooth[t]):.6f}')
            pred_mano_rot9d_smooth = batch_rot2aa(torch.stack(pred_mano_rot9d_smooth)).view(L,3)   # (L, 3, 3)
            pred_mano_trans_smooth = torch.stack(pred_mano_trans_smooth)  # (L, 3)
            pred_mano_theta_smooth = batch_rot2aa(torch.stack(pred_mano_theta_smooth).view(-1,3,3)).view(L,45)  # (L, 45)
            pred_mano_beta_smooth = repeat(mano_beta[0], 'L -> B L', B=L)  # (L, 10)
            pred_hand_vertices = mano_forward(pred_mano_theta_smooth, pred_mano_beta_smooth, pred_mano_rot9d_smooth, pred_mano_trans_smooth, side)[0]
            vis_data[f'world_{side}_hand_verts'][1:L+1] = pred_hand_vertices.cpu().numpy()
            # visualize
            offline_render(**vis_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_list", type=str, required=True, help="dataset string split by comma, e.g. 'oakink2,hoi4d'") 
    parser.add_argument("--dataset_path", type=str, default="/mnt/vepfs/zbh/MHMP/datasets/")
    parser.add_argument("--data_config", type=str, default="hand_motion", choices=list(DATA_CONFIG_MAP.keys()),help="data config name",)
    parser.add_argument("--split_portion", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--pretrain_ckpt", type=str, default=None, help="[Optional] Path to the model checkpoint directory, this will disable client server mode.",)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--mode", type=str, default='visualization', choices=['testing', 'visualization'], help="mode for testing")
    args = parser.parse_args()
    # policy
    data_config = DATA_CONFIG_MAP[args.data_config]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform(split='test')
    policy = MHMPolicy(
        model_cfg_path='model_config',
        modality_config=modality_config,
        modality_transform=modality_transform,
        pretrain_ckpt=args.pretrain_ckpt,
    )
    # dataset
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
    
    if 'vis' in args.mode:  # conduct testing for visualization
        conduct_visualization(cascaded_dataset, policy)
    else:  # conduct testing for reporting metrics
        conduct_testing(cascaded_dataset, policy, batch_size=args.batch_size)
