import os, pickle, joblib, torch, trimesh
import numpy as np
from os.path import join, basename, dirname
from natsort import natsorted
from glob import glob
from tqdm import tqdm

from dataset_loader.storage import Storage, MetaDataset
from utils.util import mp42imagefiles
from utils.pc import get_pc_from_world_to_camera
from utils.rot import batch_rodrigues, get_pose12d_from_world_to_camera


# mandatory
class EgoVideoStorage(Storage):
    def __init__(self, window_size):
        super().__init__(window_size)
    
    def add(self, task_data):
        self.storage["text"].append(task_data["instruction"])   # (1,)                                        
        self.storage["ego_rgb_filenames"].extend(task_data["real_image_filenames"]['ego_rgb'])   # (T,)       
        self.storage["ego_depth_filenames"].extend(task_data["render_image_filenames"]['ego_depth'])   # (T,)
        for side in ['left', 'right']:
            self.storage[f"motion_sequences"][side]['theta'].append(task_data["hand_dict"][side]['theta'])   # (T,...)
            self.storage[f"motion_sequences"][side]['beta'].append(task_data["hand_dict"][side]['beta'][None,:])     # (1,...)
            self.storage[f"motion_sequences"][side]['rot'].append(task_data["hand_dict"][side]['pose_seqs']['cam']['rot'])       # (T,...)
            self.storage[f"motion_sequences"][side]['trans'].append(task_data["hand_dict"][side]['pose_seqs']['cam']['trans'])   # (T,...)


# mandatory    
class EgoVideoDataset(MetaDataset):
    def __init__(self, 
        root_path='',
        window_size=30,
    ):
        '''
        real_ego_video_path: contains all egocentric rgb videos *.mp4
        real_ego_frame_path: contains all egocentric rgb frames */*.jpg
        task_list: absolute *.mp4 file list
        '''
        super().__init__(root_path)
        # load task
        self.task_list = glob(join(root_path, "*.mp4"))
        self.initialize_data_storage(EgoVideoStorage, window_size)
        
    def load_task(self, real_ego_rgb_mp4_file):
        # real egocentric RGB images
        seq_folder = real_ego_rgb_mp4_file.replace(".mp4", "")
        real_ego_frame_path = join(seq_folder, "extracted_images")
        real_rgb_filenames = glob(join(real_ego_frame_path, "*.jpg"))
        
        # camera
        slam_path = glob(join(seq_folder, "SLAM/hawor_slam_w_scale_*_*.npz"))[0]
        cam_params = dict(np.load(slam_path, allow_pickle=True))
        intrinsic_matrix = cam_params['intri']  # (3, 3), unused
        extrinsic_matrix_sequence = cam_params['extri_seqs']  # (T, 3, 4)
        extrinsic_matrix_sequence = torch.FloatTensor(extrinsic_matrix_sequence).cuda()
        
        # hand
        hand_dict = dict()
        pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = joblib.load(join(seq_folder, "world_space_res.pth"))
        for hand_idx, side in enumerate(['left', 'right']):
            hand_trans = pred_trans[hand_idx]    # (T, 3)
            hand_rot9d = batch_rodrigues(pred_rot[hand_idx])  # (T, 3, 3)
            hand_rot9d_cam, hand_trans_cam = get_pose12d_from_world_to_camera(hand_rot9d, hand_trans, extrinsic_matrix_sequence)  # (T, 3, 3), (T, 3)
            hand_theta = pred_hand_pose[hand_idx]  # (T, 45)
            hand_beta = pred_betas[hand_idx][0]  # (10,)
            hand_dict[side] = dict(
                theta=hand_theta.cpu().numpy(),  # (T, 45)
                beta=hand_beta.cpu().numpy(),    # (10,)
                pose_seqs=dict(
                    world=dict(
                        rot=hand_rot9d.cpu().numpy(),     # (T, 3, 3)
                        trans=hand_trans.cpu().numpy(), # (T, 3)
                    ),
                    cam=dict(
                        rot=hand_rot9d_cam.cpu().numpy(),  # (T, 3, 3)
                        trans=hand_trans_cam.cpu().numpy(),# (T, 3)
                    ),
                ),
                masks=pred_valid[hand_idx],  # (T, 1), whether visible hand
            )
        
        # object
        object_dict = dict(
            verts=dict(
                world=None,  # (T, M, 3)
                cam=None  # (T, M, 3)
            ),   
            faces=None,
            pose_seqs=dict(
                world=dict(
                    rot=None,  # (T, 3, 3)
                    trans=None,  # (T, 3)
                ),
                cam=dict(
                    rot=None,  # (T, 3, 3)
                    trans=None,  # (T, 3)
                ),
            ),
            masks=None,  # (T, 1), whether visible object
        )
            
        
        # check length for each input items
        num_frames = self.check_length(
            real_rgb_filenames, 
            hand_dict['left']['theta'], hand_dict['left']['pose_seqs']['cam']['rot'], hand_dict['left']['pose_seqs']['cam']['trans'],
            hand_dict['right']['theta'], hand_dict['right']['pose_seqs']['cam']['rot'], hand_dict['right']['pose_seqs']['cam']['trans'],
            extrinsic_matrix_sequence
        )    
        video_name = basename(seq_folder)
        return dict(
            # camera: for debug
            intrinsic_matrix=dict(ego=intrinsic_matrix, allo=None), extrinsic_matrix_sequence=dict(ego=extrinsic_matrix_sequence, allo=None),
            # real images
            real_image_filenames=dict(ego_rgb=real_rgb_filenames, ego_depth=None, ego_seg=None, allo_rgb=None, allo_depth=None, allo_seg=None),  
            # hand-object mesh projection rendered images
            render_image_filenames=dict(ego_rgb=None, ego_depth=None),  
            # hand
            hand_dict=hand_dict,
            # object
            object_dict=object_dict,
            # length
            length=num_frames,
            # text: TODO: video2text
            task_name=video_name,  
            instruction=video_name + '.', 
        )
        


if __name__ == "__main__":
    egovideo_dataset = EgoVideoDataset()
    print(len(egovideo_dataset))
    batch = egovideo_dataset[0]
    breakpoint()
    