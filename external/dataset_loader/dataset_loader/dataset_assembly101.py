from glob import glob
import json
import os

from natsort import natsorted
import numpy as np
import torch
import trimesh
from dataset_loader.storage import Storage, MetaDataset
from utils.pc import get_pc_from_world_to_camera
from utils.rot import batch_rodrigues, get_pose12d_from_world_to_camera
from utils.util import mano_forward
from smplx import MANO

class Assembly101Storage(Storage):
    def __init__(self, **config):
        super().__init__(**config)
    
    def fill_data(self, task_data):
        if task_data["hand_dict"][self.side] is not None:
            return dict(
                text=task_data["instruction"],
                ego_rgb_files=task_data["real_image_filenames"]['ego_rgb'],                 
                ego_dep_files=task_data["real_image_filenames"]['ego_depth'],                        
                theta_seqs=task_data["hand_dict"][self.side]['theta'],                      # (T, 15*3)
                beta_seqs=task_data["hand_dict"][self.side]['beta'].reshape(-1,10),         # (1, 10)
                rot_seqs=task_data["hand_dict"][self.side]['pose_seqs']['cam']['rot'],      # (T, 3, 3)
                trans_seqs=task_data["hand_dict"][self.side]['pose_seqs']['cam']['trans'],  # (T, 3)
                len=task_data["length"],
            )

class Assembly101Dataset(MetaDataset):
    def __init__(self, 
        root = "/Volumes/T9/Assembly101",
        window_size=30,
    ):
        super().__init__(root)

        self.root = root
        self.task_list = ["nusar-2021_action_both_9013-a02_9013_user_id_2021-02-02_130807"]
    

    def load_task(self, task_name = "nusar-2021_action_both_9013-a02_9013_user_id_2021-02-02_130807"):
        rgb_path = os.path.join(self.root, "recording", task_name, "HMC_84358933_mono10bit.mp4")
        camera_position_ego_path = os.path.join(self.root, "assembly101_camera_and_hand_poses", "camera_position_ego", f"{task_name}.json")
        timestamp_path = os.path.join(self.root, "assembly101_camera_and_hand_poses", "timestamp", f"{task_name}.json")
        hand_confidences_path = os.path.join(self.root, "assembly101_camera_and_hand_poses", "hand_confidences", f"{task_name}.json")
        camera_extrinsics_ego_path = os.path.join(self.root, "assembly101_camera_and_hand_poses", "camera_extrinsics_ego", f"{task_name}.json")
        landmarks3D_path = os.path.join(self.root, "assembly101_camera_and_hand_poses", "landmarks3D", f"{task_name}.json")
        xf_transf_path = os.path.join(self.root, "assembly101_camera_and_hand_poses", "xf_transf", f"{task_name}.json")
        landmarks2D_path = os.path.join(self.root, "assembly101_camera_and_hand_poses", "landmarks2D", f"{task_name}.json")
        camera_extrinsics_fixed_path = os.path.join(self.root, "assembly101_camera_and_hand_poses", "camera_extrinsics_fixed", f"{task_name}.json")
        camera_position_fixed_path = os.path.join(self.root, "assembly101_camera_and_hand_poses", "camera_position_fixed", f"{task_name}.json")
        hand_bboxes_path = os.path.join(self.root, "assembly101_camera_and_hand_poses", "hand_bboxes", f"{task_name}.json")

        




        


