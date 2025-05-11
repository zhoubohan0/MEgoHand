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
import cv2, csv
import torch
from einops import repeat
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, Subset
from multiprocessing.pool import ThreadPool

from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import MHMPolicy
from gr00t.utils.util import default_collate_fn, mse, glance_dict, visualize_RGBD, resize, timer
from mhmp_vis.dataset_loader import *
from mhmp_vis.utils.rot import rot6d_to_rotmat, rotmat_dist, decode_absolute_pose_from_relative, batch_rot2aa, get_pose12d_from_camera_to_world, batch_rodrigues
from mhmp_vis.utils.util import mano_forward
from mhmp_vis.utils.pc import get_pc_from_world_to_camera, get_pc_from_camera_to_world
from mhmp_vis.dataset_visualizer.general_offline_visualizer import offline_render
# from mhmp_vis.dataset_visualizer.general_interactive_visualizer import GeneralInteractiveVisualizer

warnings.simplefilter("ignore", category=FutureWarning)

## metrics
@torch.jit.script
def procrustes_alignment(pred_joints: torch.Tensor, true_joints: torch.Tensor) -> torch.Tensor:
    """
    Computes a similarity transform (sR, t) in a batched way that takes
    a set of 3D points pred_joints (B, N, 3) closest to a set of 3D points true_joints (B, N, 3),
    where R is a 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    Args:
        pred_joints (torch.Tensor): First set of points of shape (B, N, 3).
        true_joints (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (torch.Tensor): The first set of points after applying the similarity transformation.
    """

    batch_size = pred_joints.shape[0]
    pred_joints = pred_joints.permute(0, 2, 1)
    true_joints = true_joints.permute(0, 2, 1)
    # 1. Remove mean.
    mu1 = pred_joints.mean(dim=2, keepdim=True)
    mu2 = true_joints.mean(dim=2, keepdim=True)
    X1 = pred_joints - mu1
    X2 = true_joints - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = (X1**2).sum(dim=(1,2))

    # 3. The outer product of X1 and X2.
    K = torch.matmul(X1, X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
    U, s, V = torch.svd(K)
    Vh = V.permute(0, 2, 1)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=U.device).unsqueeze(0).repeat(batch_size, 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.linalg.det(torch.matmul(U, Vh)))

    # Construct R.
    R = torch.matmul(torch.matmul(V, Z), U.permute(0, 2, 1))

    # 5. Recover scale.
    trace = torch.matmul(R, K).diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
    scale = (trace / var1).unsqueeze(dim=-1).unsqueeze(dim=-1)

    # 6. Recover translation.
    t = mu2 - scale*torch.matmul(R, mu1)

    # 7. Error:
    pred_joints_hat = scale*torch.matmul(R, pred_joints) + t

    return pred_joints_hat.permute(0, 2, 1)

@torch.jit.script   
def MPJPE_PA(pred_joints: torch.Tensor, true_joints: torch.Tensor) -> torch.Tensor:
    """
    Compute mean per joint position error in cm after performing Procrustes alignment.
    Args:
        pred_joints (torch.Tensor): Predicted 3D joints of shape (B, N, 3).
        true_joints (torch.Tensor): Ground truth 3D joints of shape (B, N, 3).
    """
    pred_joints_hat = procrustes_alignment(pred_joints, true_joints)
    # return 100 * torch.sqrt(((pred_joints_hat - true_joints)** 2).sum(dim=-1)).mean(dim=-1)
    return 100 * torch.norm(pred_joints_hat - true_joints, dim=-1).mean()

@torch.jit.script
def MPJPE(pred_joints, true_joints) -> torch.Tensor:
    """
    Compute mean per joint position error in cm. before Procrustes alignment.
    Args:
        pred_joints (torch.Tensor): Predicted 3D joints of shape (B, N, 3).
        true_joints (torch.Tensor): Ground truth 3D joints of shape (B, N, 3).
    """
    # return 100 * torch.sqrt(((pred_joints - true_joints) ** 2).sum(dim=-1)).mean(dim=-1)
    return 100 * torch.norm(pred_joints - true_joints, dim=-1).mean()

@torch.jit.script
def MPVE(pred_vertexes, gt_vertexes) -> torch.Tensor:
    """
    Compute mean per vertex error in cm.
    Args:
        pred_vertexes (torch.Tensor): Predicted 3D vertexes of shape (B, N', 3).
        gt_vertexes (torch.Tensor): Ground truth 3D vertexes of shape (B, N', 3).
    """
    # return 100 * torch.sqrt(((pred_vertexes - gt_vertexes) ** 2).sum(dim=-1)).mean(dim=-1)
    return 100 * torch.norm(pred_vertexes - gt_vertexes, dim=-1).mean()

@torch.jit.script
def MPVE_PA(pred_vertexes, gt_vertexes) -> torch.Tensor:
    """
    Compute mean per vertex error in cm. after Procrustes alignment.
    Args:
        pred_vertexes (torch.Tensor): Predicted 3D vertexes of shape (B, N', 3).
        gt_vertexes (torch.Tensor): Ground truth 3D vertexes of shape (B, N', 3).
    """
    pred_vertexes_hat = procrustes_alignment(pred_vertexes, gt_vertexes)
    return 100 * torch.norm(pred_vertexes_hat - gt_vertexes, dim=-1).mean()

@torch.jit.script
def MWTE(pred_trans, gt_trans) -> torch.Tensor:
    """
    Compute mean wrist translation error in cm.
    Args:
        pred_trans (torch.Tensor): Predicted 3D translation of shape (B, 1, 3).
        gt_trans (torch.Tensor): Ground truth 3D translation of shape (B, 1, 3).
    """
    return 100 * torch.norm(pred_trans - gt_trans, dim=-1).mean()

def MRE(pred_rots, gt_rots) -> torch.Tensor:
    """
    Compute mean rotation error (wrist + finger) in [0, pi]
    Args:
        pred_rots (torch.Tensor): Predicted 3D rotation of shape (B, 3, 3).
        gt_rots (torch.Tensor): Ground truth 3D rotation of shape (B, 3, 3).
    """
    return rotmat_dist(pred_rots, gt_rots).mean()


def decode_mano_para_6d(mano_dict):
    '''
    Args:
        mano_dict: dict
            action.theta: torch.Size([B, T, 90])
            action.rot: torch.Size([B, T, 6])
            action.trans: torch.Size([B, T, 3])  
    Returns:
        mano_dict: dict
            theta: torch.Size([B, T, 45])
            theta9d: torch.Size([B, T, 15, 3, 3])
            rot: torch.Size([B, T, 3])
            rot9d: torch.Size([B, T, 3, 3])
            trans: torch.Size([B, T, 3])  
    '''
    mano_dict = {k: torch.as_tensor(v).cuda() for k, v in mano_dict.items()}
    B, T = mano_dict['action.trans'].shape[:2]
    theta_flatten_9d = rot6d_to_rotmat(mano_dict['action.theta'].reshape(-1,6))  # (B*T*15, 3, 3)
    theta = batch_rot2aa(theta_flatten_9d).view(B, T, 45)    
    theta9d = theta_flatten_9d.reshape(B, T, 15, 3, 3)  
    rot_flatten_9d = rot6d_to_rotmat(mano_dict['action.rot'].reshape(-1,6))    # (B*T, 3, 3)
    rot = batch_rot2aa(rot_flatten_9d).view(B, T, 3)
    rot9d = rot_flatten_9d.view(B, T, 3, 3) 
    return dict(
        theta=theta,           # (B, T, 45)
        theta9d=theta9d,       # (B, T, 15, 3, 3)
        rot=rot,               # (B, T, 3)
        rot9d=rot9d,           # (B, T, 3, 3)
        trans=mano_dict['action.trans']  # (B, T, 3)
    )


def calculate_mano_outputs_metrics(pred_hand_vertices, pred_hand_joints, true_hand_vertices, true_hand_joints, mask):
    '''
    Args:
        pred_hand_vertices: torch.Size([L, 778, 3])
        pred_hand_joints: torch.Size([L, 21, 3])
        true_hand_vertices: torch.Size([L, 778, 3])
        true_hand_joints: torch.Size([L, 21, 3])
    mask: torch.Size([T])
        default to be True
    
    Returns:
        metrics: dict
            'MPJPE',
            'MPJPE-PA',
            'MPVE',
            'MPVE-PA',
    '''
    return {
        'MPJPE': MPJPE(pred_hand_joints[mask], true_hand_joints[mask]).item(),
        'MPJPE-PA': MPJPE_PA(pred_hand_joints[mask], true_hand_joints[mask]).item(),
        'MPVE': MPVE(pred_hand_vertices[mask], true_hand_vertices[mask]).item(),
        'MPVE-PA': MPVE_PA(pred_hand_vertices[mask], true_hand_vertices[mask]).item(),
    }

def calculate_mano_parameters_metrics(pred_mano, true_mano, mask):
    '''
    Args:
        pred_mano: dict
            theta9d: torch.Size([T, 15, 3, 3])
            rot9d: torch.Size([T, 3, 3])
            trans: torch.Size([T, 3])
        true_mano: dict
            theta9d: torch.Size([T, 15, 3, 3])
            rot9d: torch.Size([T, 3, 3])
            trans: torch.Size([T, 3])
        mask: torch.Size([T])
            default to be True
    
    Returns:
        metrics: dict
            'MWTE',
            'MRE'
    '''
    return {
        'MWTE': MWTE(pred_mano['trans'][mask], true_mano['trans'][mask]).item(),
        'MRE': MRE(
            torch.cat([pred_mano['rot9d'].unsqueeze(1), pred_mano['theta9d']], dim=1)[mask].view(-1, 3, 3), 
            torch.cat([true_mano['rot9d'].unsqueeze(1), true_mano['theta9d']], dim=1)[mask].view(-1, 3, 3)
        ).item(),
    }

def test_and_visualize(
    cascaded_dataset, policy,
    vis_dep_gen=False, vis_ren_ori=False, vis_ren_gen=False,
):
    def truncate_vis_data(vis_data, trunk_len):
        vis_data_copy = vis_data.copy()
        for k in ['image_filenames', 'cam_left_hand_verts', 'cam_right_hand_verts', 'world_left_hand_verts', 'world_right_hand_verts', 'extrinsic_matrix_sequence']:
            vis_data_copy[k] = vis_data_copy[k][:trunk_len]
        for view in ['cam', 'world']:
            for i in range(len(vis_data_copy[f'{view}_object_dict_list'])):
                vis_data_copy[f'{view}_object_dict_list'][i][f'{view}_object_verts'] = vis_data_copy[f'{view}_object_dict_list'][i][f'{view}_object_verts'][:trunk_len]
        # for k in vis_data_copy: print(k, len(vis_data_copy[k]) if vis_data_copy[k] is not None else '')
        return vis_data_copy
    
    # record metrics
    metrics_file = os.path.join(os.path.dirname(args.pretrain_ckpt), f'{cascaded_dataset.testset}_metrics.csv')
    fieldnames = ['Dataset', 'Trajectory', 'Frames', 'MPJPE', 'MPJPE-PA', 'MPVE', 'MPVE-PA', 'MWTE', 'MRE']

    # Read existing entries into a set for quick lookup
    metrics_backup = set()
    if os.path.isfile(metrics_file):
        with open(metrics_file, 'r') as f:
            reader = csv.DictReader(f)
            metrics_backup = {(row['Dataset'], int(row['Trajectory'])) for row in reader}
        writer_file = open(metrics_file, 'a', newline='')
        writer = csv.DictWriter(writer_file, fieldnames=fieldnames)
    else:
        writer_file = open(metrics_file, 'w', newline='')
        writer = csv.DictWriter(writer_file, fieldnames=fieldnames)
        writer.writeheader()
    
    window_size = cascaded_dataset.dataset_list[0].data_storage.window_size
    side = cascaded_dataset.dataset_list[0].data_storage.side
    # get all trajectories index in cascaded_dataset
    all_valid_trajectory_range = dict()
    for dataset_idx in range(len(cascaded_dataset.dataset_list)):
         dataset = cascaded_dataset.dataset_list[dataset_idx]
         all_valid_trajectory_range[str(dataset)] = np.array(dataset.data_storage.valid_trajectory_range) + cascaded_dataset.cumsum_length_list[dataset_idx]
    # Visualize K trajectories for each dataset 
    for dataset_name, valid_trajectory_range in all_valid_trajectory_range.items():
        pbar = tqdm(range(len(valid_trajectory_range)), desc=f"Visualizing {dataset_name}", unit="trajectory", leave=False, dynamic_ncols=True, position=0)
        for i_traj in pbar:
            if (dataset_name, i_traj) in metrics_backup:
                continue
            b, e = valid_trajectory_range[i_traj]
            T = int(e - b)
            L = T + window_size - 1
            
            ## get data
            # 1. step_data: only used for inference because of relative pose
            '''
            # slower
            with timer('step_data_loader'):
                step_loader = DataLoader(
                    Subset(cascaded_dataset, range(b, e)),
                    batch_size=T,  # Load entire trajectory at once
                    num_workers=4,
                    collate_fn=default_collate_fn,
                    pin_memory=True
                )
                step_data = next(iter(step_loader))
            '''
            # step_data = default_collate_fn([cascaded_dataset[_] for _ in range(b,e)])
            step_data = cascaded_dataset[slice(b, e)]

            # 2. ref_data: used for initial MANO parameters and GT MANO parameters
            dataset_idx, sample_idx = cascaded_dataset.get_dataset_sample_id_by_idx(b)
            dataset = cascaded_dataset.dataset_list[dataset_idx]
            '''
            # slower
            class RefDataset(Dataset):
                def __init__(self, dataset, sample_idx, length):
                    self.dataset = dataset
                    self.sample_idx = sample_idx
                    self.length = length
                    
                def __len__(self):
                    return self.length
                    
                def __getitem__(self, idx):
                    sample = self.dataset.__get_ref_item__(self.sample_idx + idx, device='cpu')
                    return {k: v.cuda() for k, v in sample.items()}
            
            with timer('ref_data_loader'):
                ref_loader = DataLoader(
                    RefDataset(dataset, sample_idx, T),
                    batch_size=T,  # Load entire trajectory at once
                    num_workers=4,
                    collate_fn=default_collate_fn,
                    pin_memory=True
                )
                ref_data = next(iter(ref_loader))
            '''
            # ref_data = default_collate_fn([dataset.__get_ref_item__(sample_idx+_) for _ in range(T)])
            ref_data = dataset.__get_ref_item__(slice(sample_idx, sample_idx+T))

            assert len(np.unique(ref_data['task_name'])) == 1
            # 3. vis_data: occupation for visualization
            task_name = ref_data['task_name'][0][0]
            if 'subtask' in task_name:
                task_name, subtask = os.path.split(task_name)
                isub = int(subtask.replace('subtask', ''))
            else:
                isub = 0    
            vis_data = dataset.task_example_for_vis(task_name, isub)
            while L + 1 != len(vis_data['extrinsic_matrix_sequence']):
                isub += 1
                vis_data = dataset.task_example_for_vis(task_name, isub)
            assert all([x in vis_data['save_video_file'] for x in task_name.split('/')])
            '''
            # debug single step inference
            pred_action_single = {
                'action.theta':[],
                'action.rot':[],
                'action.trans':[]
            }
            for timestep in range(T):
                step_data_t = {k: v[timestep:timestep+1] for k, v in step_data.items()}
                pred_action_t = policy.get_action(step_data_t)
                for k in pred_action_single:
                    pred_action_single[k].append(pred_action_t[k])
            pred_action = {k: np.concatenate(v, 0) for k, v in pred_action_single.items()}
            '''
            ## visualize depth prediction
            pred_action, mid_info = policy.get_action(step_data, True)
            if vis_dep_gen:
                depth_pred = mid_info['depth_pred'].squeeze(1).cpu().numpy()
                depth_gt = step_data['depth.ego_view'].squeeze().cpu().numpy()
                rgb = np.stack([cv2.resize(img, depth_gt.shape[::-1], interpolation=cv2.INTER_LINEAR) for img in step_data['video'].squeeze()])
                out_dir = f'vis_depth/{dataset_name}_traj{i_traj:03d}_T{T:03d}'
                for t in range(T):
                    visualize_RGBD(depth=depth_pred[t], depth_gt=depth_gt[t], rgb=rgb[t], save_file=os.path.join(out_dir, f'{t:03d}.png'))

            # forward prediction
            pred_mano = decode_mano_para_6d(pred_action)
            true_mano = {k: torch.cat([v[:-1,0], v[-1]], 0) for k, v in ref_data.items() if k in pred_mano}
            '''
            pred_mano / true_mano
            - theta: (L, W, 45)
            - theta9d: (L, W, 15, 3, 3)
            - rot: (L, W, 3)
            - rot9d: (L, W, 3, 3)
            - trans: (L, W, 3)
            '''
            # MANO pose from absolute cam pose to absolute world pose
            extrinsic_matrix_sequence = torch.Tensor(vis_data['extrinsic_matrix_sequence']).cuda()
            true_mano['theta'] = true_mano['theta'].view(-1,45)
            mask = true_mano['theta'].norm(dim=1) > 0 if task_name.startswith('scene_') else torch.ones(len(true_mano['theta']), device=true_mano['theta'].device, dtype=torch.bool)
            true_mano['rot9d'], true_mano['trans'] = get_pose12d_from_camera_to_world(
                true_mano['rot9d'],
                true_mano['trans'],
                extrinsic_matrix_sequence[1:]
            )
            true_mano['rot'] = batch_rot2aa(true_mano['rot9d'])
            pred_mano_rot9d, pred_mano_trans = [], []
            # used for relative pose -> absolute cam pose
            mano_beta = ref_data['state.beta']          # (T, 10)
            init_rot9d = ref_data['state.rot9d']        # (T, 3, 3)
            init_trans = ref_data['state.trans']        # (T, 3)
            for t in range(T):
                # relative pose -> absolute cam pose
                pred_mano_rot9d_t, pred_mano_trans_t = decode_absolute_pose_from_relative(  # (W, 3, 3), (W, 3)
                    init_rot9d[[t]],
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
            pred_mano_theta9d_smooth, pred_mano_rot9d_smooth, pred_mano_trans_smooth = [], [], []
            for t in range(L):
                pred_mano_rot9d_chunk_t = pred_mano_rot9d[indices == t]  # at most (W, 3, 3)
                pred_mano_trans_chunk_t = pred_mano_trans[indices == t]  # at most (W, 3)
                pred_mano_theta9d_chunk_t = pred_mano['theta9d'][indices == t]  # at most (W, 15, 3, 3)
                chunk_size = len(pred_mano_rot9d_chunk_t)
                '''
                weights = torch.exp(-1.5 * torch.arange(chunk_size, dtype=torch.float32, device=pred_mano_trans_chunk_t.device))
                weights = weights / weights.sum() 
                '''
                weights = torch.ones(chunk_size, device=pred_mano_trans_chunk_t.device) / chunk_size  # average
                U, S, Vh = torch.linalg.svd(torch.sum(pred_mano_rot9d_chunk_t * weights.view(-1,1,1), dim=0))
                pred_mano_rot9d_smooth.append(U @ Vh)  # (3, 3)
                U, S, Vh = torch.linalg.svd(torch.sum(pred_mano_theta9d_chunk_t * weights.view(-1,1,1,1), dim=0))
                pred_mano_theta9d_smooth.append(U @ Vh)  # (15, 3, 3)
                pred_mano_trans_smooth.append(torch.sum(pred_mano_trans_chunk_t * weights.view(-1,1), dim=0))  # (3)
                '''
                if t % 10 == 0:
                    print(f'[{t}/{L}]')
                    print(f"origin: rot:{mse(true_mano['rot9d'][t],pred_mano_rot9d_chunk_t[0]):.6f} | trans:{mse(true_mano['trans'][t], pred_mano_trans_chunk_t[0]):.6f} | theta:{mse(true_mano['theta9d'][t],pred_mano_theta9d_chunk_t[0]):.6f}")
                    print(f"smooth: rot:{mse(true_mano['rot9d'][t],pred_mano_rot9d_smooth[t]):.6f} | trans:{mse(true_mano['trans'][t], pred_mano_trans_smooth[t]):.6f} | theta:{mse(true_mano['theta9d'][t],pred_mano_theta9d_smooth[t]):.6f}")
                '''
            pred_mano_rot9d_smooth = torch.stack(pred_mano_rot9d_smooth).view(L,3,3)                    # (L, 3, 3)
            pred_mano_rot_smooth = batch_rot2aa(pred_mano_rot9d_smooth).view(L,3)                       # (L, 3)
            pred_mano_theta9d_smooth = torch.stack(pred_mano_theta9d_smooth).view(L,15,3,3)             # (L, 15, 3, 3)
            pred_mano_theta_smooth = batch_rot2aa(pred_mano_theta9d_smooth.view(-1,3,3)).view(L,45)     # (L, 45)
            pred_mano_trans_smooth = torch.stack(pred_mano_trans_smooth)                                # (L, 3)
            pred_mano_beta_smooth = repeat(mano_beta[0], 'L -> B L', B=L)                               # (L, 10)
            pred_mano_smooth = {
                'theta': pred_mano_theta_smooth,
                'theta9d': pred_mano_theta9d_smooth,
                'rot': pred_mano_rot_smooth,
                'rot9d': pred_mano_rot9d_smooth,
                'trans': pred_mano_trans_smooth,
                'beta': pred_mano_beta_smooth,
            }
            pred_hand_vertices, pred_hand_joints = mano_forward(  # (L, 778, 3), (L, 21, 3)
                pred_mano_theta_smooth, 
                pred_mano_beta_smooth, 
                pred_mano_rot_smooth, 
                pred_mano_trans_smooth, 
                side
            )[:2]
            
            ## calculate trajectory metrics
            true_hand_vertices, true_hand_joints = mano_forward(  # (L, 778, 3), (L, 21, 3)
                true_mano['theta'], 
                mano_beta[0], 
                true_mano['rot'], 
                true_mano['trans'], 
                side
            )[:2]
            batch_avg_metrics = {
                **calculate_mano_outputs_metrics(pred_hand_vertices, pred_hand_joints, true_hand_vertices, true_hand_joints, mask),
                **calculate_mano_parameters_metrics(pred_mano_smooth, true_mano, mask),
            }
            ## update metrics
            pbar.set_description(f"Dataset: {dataset_name} | Trajectory: {i_traj} | Frames: {T}")
            pbar.set_postfix({k: f"{v:.4f}" for k, v in batch_avg_metrics.items()})
            row = {
                'Dataset': dataset_name,
                'Trajectory': i_traj,
                'Frames': T,
                **{k: round(v, 4) for k, v in batch_avg_metrics.items()},
            }
            writer.writerow(row)
            writer_file.flush()
            metrics_backup.add((dataset_name, i_traj))
            ## visualize: 
            if vis_ren_ori:  # save original video
                offline_render(**vis_data)
            if vis_ren_gen:  # save generated video: overwrite 'cam_{side}_hand_verts', 'world_{side}_hand_verts', save_video_file
                vis_data[f'world_{side}_hand_verts'][1:L+1] = pred_hand_vertices.cpu().numpy()
                vis_data['save_video_file'] = f"/{os.path.basename(os.path.dirname(args.pretrain_ckpt)).split('_')[0]}/".join(vis_data['save_video_file'].split('/'))
                offline_render(**vis_data)

    writer_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", type=str, default="eval", choices=["eval", "test"], help="testset name")
    parser.add_argument("--dataset_path", type=str, default="/mnt/vepfs/zbh/MHMP/datasets/")
    parser.add_argument("--data_config", type=str, default="hand_motion", choices=list(DATA_CONFIG_MAP.keys()),help="data config name",)
    parser.add_argument("--pretrain_ckpt", type=str, default=None, help="Path to the model checkpoint",)
    parser.add_argument("--debug", type=int, default=0)
    ### visualize options
    parser.add_argument("--vis_dep_gen", action='store_true', default=False)
    parser.add_argument("--vis_ren_ori", action='store_true', default=False)
    parser.add_argument("--vis_ren_gen", action='store_true', default=False)
    args = parser.parse_args()
    
    # choose testset
    if args.testset == "eval":
        dataset_list="fpha,h2o,hoi4d,hot3d,taco,oakink2"
        split_portion=0.99
    elif args.testset == "test":
        dataset_list="arctic"
        split_portion=0.0
    else:
        raise ValueError(f"Invalid testset: {args.testset}")
    
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
        dataset_list=dataset_list.split(','),
        split_portion=split_portion,
        data_config=data_config,
        data_storage_kwargs=data_storage_kwargs,
        test=True,
        debug=args.debug,
    )
    cascaded_dataset.testset = args.testset
    
    # evaluate
    test_and_visualize(cascaded_dataset, policy, args.vis_dep_gen, args.vis_ren_ori, args.vis_ren_gen)
