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

class H2OStorage(Storage):
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

class H2ODataset(MetaDataset):
    def __init__(self, 
        root = "/Volumes/zhan_4T/H2O",
        window_size=30,
    ):
        super().__init__(root)

        self.root = root

        # self.task_types = ['h1', 'h2', 'k1', 'k2', 'o1', 'o2']
        # self.task_list = [f"subject{subject}/{task}" for subject in range(1, 5) for task in self.task_types]
        self.task_list = ["subject1_ego/h1/1/cam4"]
        self.mapping = {
            0: "background", 1: "book", 2: "espresso", 3: "lotion", 4: "spray", 5: "milk", 6: "cocoa", 7: "chips", 8: "cappuccino",
        }

        self.action_mapping = {
            0: "background", 1: "grab book", 2: "grab espresso", 3: "grab lotion",
            4: "grab spray", 5: "grab milk", 6: "grab cocoa", 7: "grab chips",
            8: "grab cappuccino", 9: "place book", 10: "place espresso",
            11: "place lotion", 12: "place spray", 13: "place milk",
            14: "place cocoa", 15: "place chips", 16: "place cappuccino",
            17: "open lotion", 18: "open milk", 19: "open chips",
            20: "close lotion", 21: "close milk", 22: "close chips",
            23: "pour milk", 24: "take out espresso", 25: "take out cocoa",
            26: "take out chips", 27: "take out cappuccino",
            28: "put in espresso", 29: "put in cocoa", 30: "put in cappuccino",
            31: "apply lotion", 32: "apply spray", 33: "read book",
            34: "read espresso", 35: "spray spray", 36: "squeeze lotion"
        }



    def load_task(self, task_name = "subject1_ego/h1/1/cam4"):

        rgb_path = os.path.join(self.root, task_name, "rgb")
        depth_path = os.path.join(self.root, task_name, "depth")
        cam_pose_path = os.path.join(self.root, task_name, "cam_pose")
        hand_pose_MANO_path = os.path.join(self.root, task_name, "hand_pose_MANO")
        obj_pose_RT_path = os.path.join(self.root, task_name, "obj_pose_RT")
        action_label_path = os.path.join(self.root, task_name, "action_label")
        verb_label_path = os.path.join(self.root, task_name, "verb_label")
        cam_intrinsic_path = os.path.join(self.root, task_name, "cam_intrinsics.txt")

        rgb_filenames = natsorted(glob(os.path.join(rgb_path, "*.png")))
        depth_filenames = natsorted(glob(os.path.join(depth_path, "*.png")))

        object_pose_RT_filenames = natsorted(glob(os.path.join(obj_pose_RT_path, "*.txt")))
        hand_pose_files = natsorted(glob(os.path.join(hand_pose_MANO_path, "*.txt")))

        cam_extrinsic_path = natsorted(glob(os.path.join(cam_pose_path, "*.txt")))
        cam_extrinsic = self._load_cam_extrinsic(cam_extrinsic_path)
        cam_intrinsic = self._read_intrinsics(cam_intrinsic_path)

        
        action_label_files = natsorted(glob(os.path.join(action_label_path, "*.txt")))
        action_label = self._load_action_label(action_label_files)

        def _split_task_by_action_label(one_action_label):

            label, start, end = one_action_label
            instruction = self.action_mapping[label]
            rgb_filenames_split = rgb_filenames[start:end]
            depth_filenames_split = depth_filenames[start:end]
            hand_pose_files_split = hand_pose_files[start:end]
            cam_extrinsic_split = cam_extrinsic[start:end]

            left_hand_dict_split = self._load_mano_params(hand_pose_files_split, cam_extrinsic_split, hand='left')
            right_hand_dict_split = self._load_mano_params(hand_pose_files_split, cam_extrinsic_split, hand='right')
            hand_dict_split = dict(left=left_hand_dict_split, right=right_hand_dict_split)

            object_pose_RT_filenames_split = object_pose_RT_filenames[start:end]
            obj_class, obj_pose_world_split = self._load_object_pose(object_pose_RT_filenames_split)
            object_mesh_path = os.path.join(self.root, "object", self.mapping[obj_class], f"{self.mapping[obj_class]}.obj")
            vertices, faces = self._read_obj(object_mesh_path)
            object_dict_split = self._return_object_dict(obj_pose_world_split, vertices, faces, cam_extrinsic_split)


            # check length for each input items
            num_frames = self.check_length(
                rgb_filenames_split,
                depth_filenames_split,
                hand_dict_split['right']['theta'], hand_dict_split['right']['pose_seqs']['cam']['rot'], hand_dict_split['right']['pose_seqs']['cam']['trans'],
                *((object_dict_split[0]["pose_seqs"]['world']['rot'], object_dict_split[0]["pose_seqs"]['world']['trans'])),
                cam_extrinsic_split
            )

            return dict(
                # camera
                intrinsic_matrix=dict(ego=cam_intrinsic, allo=None),
                extrinsic_matrix_sequence=dict(ego=cam_extrinsic_split, allo=None),
                # real images
                real_image_filenames=dict(ego_rgb=rgb_filenames_split, ego_depth=depth_filenames_split, ego_seg=None, allo_rgb=None, allo_depth=None, allo_seg=None),
                # hand-object mesh projection rendered images
                render_image_filenames=dict(ego_rgb=None, ego_depth=None),
                # hand
                hand_dict=hand_dict_split,
                # object
                object_dict=object_dict_split,
                # length
                length=num_frames,
                # text
                task_name=task_name,
                instruction=instruction,
            )
        
        task_data_list = []

        for one_action_label in action_label:
            task_data_list.append(_split_task_by_action_label(one_action_label))

        return task_data_list[1:]
    

    def _load_mano_params(self, file_list, extrinsic_matrix_sequence, hand='left', device='cpu'):

        all_pose = []
        all_shape = []
        all_trans = []

        for file_path in file_list:
            with open(file_path, 'r') as f:
                nums = list(map(float, f.read().strip().split()))
            
            if len(nums) != 124:
                raise ValueError(f"{file_path} 长度为 {len(nums)}，应为 124")

            if hand == 'left':
                base = 0
            elif hand == 'right':
                base = 62  # 前面是左手的 1+3+48+10 = 62
            else:
                raise ValueError("hand 参数必须为 'left' 或 'right'")

            trans = torch.tensor(nums[base + 1 : base + 4], dtype=torch.float32, device=device).unsqueeze(0)
            pose = torch.tensor(nums[base + 4 : base + 52], dtype=torch.float32, device=device).unsqueeze(0)
            shape = torch.tensor(nums[base + 52 : base + 62], dtype=torch.float32, device=device).unsqueeze(0)

            all_pose.append(pose)
            all_shape.append(shape)
            all_trans.append(trans)

        pose = torch.cat(all_pose, dim=0)
        shape = torch.cat(all_shape, dim=0)
        trans = torch.cat(all_trans, dim=0)
            
        rot = pose[:, :3]       # shape: (T, 3)
        theta = pose[:, 3:]     # shape: (T, 45)

        is_right = hand == 'right'

        # verts, joints, faces = mano_forward(theta, shape, rot, trans, hand)
        mano_model = MANO(
            "manopth/mano/models",
            use_pca=False,
            flat_hand_mean=True,
            create_transl=True,
            is_rhand=is_right
        ).to(device)
    
        output = mano_model(
            global_orient=rot.to(device),
            hand_pose=theta.to(device),
            betas=shape.to(device),
            transl=trans.to(device)
        )

        # faces
        mano_root = "manopth/mano/models"
        face_path = f"{mano_root}/{hand}_faces.npy"
        faces = np.load(face_path)

        verts = output.vertices  # shape: [T, 778, 3]
        joints = output.joints  # shape: [T, 16, 3]
        verts_cam = get_pc_from_world_to_camera(verts.to(device), torch.tensor(extrinsic_matrix_sequence, dtype=torch.float32).to(device))

        rot_mat = batch_rodrigues(rot)  # shape: [T, 3, 3]
        rot_cam, trans_cam = get_pose12d_from_world_to_camera(rot_mat.to(device), trans.to(device), torch.tensor(extrinsic_matrix_sequence, dtype=torch.float32).to(device))

        return dict(
            theta=theta.cpu().numpy(),  # (T, 45)
            beta=shape.cpu().numpy(),    # (10,)
            pose_seqs=dict(
                world=dict(
                    rot=rot_mat.cpu().numpy(),     # (T, 3, 3)
                    trans=trans.cpu().numpy(), # (T, 3)
                ),
                cam=dict(
                    rot=rot_cam.cpu().numpy(),  # (T, 3, 3)
                    trans=trans_cam.cpu().numpy(),# (T, 3)
                ),
            ),
            mesh=dict(
                verts=dict(
                    world=verts.detach().cpu().numpy(),  # (T, 778, 3)
                    cam=verts_cam.detach().cpu().numpy(),  # (T, 778, 3)
                ),
                faces=faces,     # (M, 3)
                keypoints=dict(
                    world=joints.detach().cpu().numpy(),  # (T, 21, 3)
                )
            ),
            masks=None,  # (T, 1), whether visible hand
        )
    

    def _load_action_label(self, file_list):
        action_label = []
        for file_path in file_list:
            with open(file_path, 'r') as f:
                nums = f.read().strip()
                if nums != "":
                    action_label.append(int(nums))

        # Step 1: 先分段，记录每段的 (label, start_idx, end_idx)
        segments = []
        start_idx = 0
        for i in range(1, len(action_label)):
            if action_label[i] != action_label[i - 1]:
                segments.append((action_label[start_idx], start_idx, i - 1))
                start_idx = i
        segments.append((action_label[start_idx], start_idx, len(action_label) - 1))

        # Step 2: 合并0段（0合并到前面段，开头为0则合并到后面）
        merged_indices = []
        i = 0
        while i < len(segments):
            label, start, end = segments[i]

            if label != 0:
                merged_indices.append((label, start, end))
                i += 1
            else:
                if i == 0:
                    # 开头是0段 → 合并到后一个非0段
                    if len(segments) > 1:
                        next_label, next_start, next_end = segments[i + 1]
                        merged_indices.append((next_label, start, next_end))
                        i += 2
                    else:
                        # 整个序列都是0
                        merged_indices.append((0, start, end))
                        i += 1
                else:
                    # 中间或结尾的0段 → 合并到前一个段
                    prev_label, prev_start, prev_end = merged_indices.pop()
                    merged_indices.append((prev_label, prev_start, end))
                    i += 1

        return merged_indices


    def _load_hand_pose(self, hand_pose_files):
        left_hands = []
        right_hands = []

        for file_path in hand_pose_files:
            with open(file_path, 'r') as f:
                nums = list(map(float, f.read().strip().split()))
            
            if len(nums) != 128:
                raise ValueError(f"文件 {file_path} 的数据长度为 {len(nums)}，但预期是 128 个数字")

            # 解析左手
            left_hand = np.array(nums[1:64]).reshape(21, 3)
            left_hands.append(left_hand)

            # 解析右手
            right_hand = np.array(nums[65:]).reshape(21, 3)
            right_hands.append(right_hand)

        # 转换为 numpy 数组，形状为 (T, 21, 3)
        left_hands = np.stack(left_hands, axis=0)
        right_hands = np.stack(right_hands, axis=0)

        return left_hands, right_hands


    def _read_obj(self, file_path):
        vertices = []
        faces = []

        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('v '):  # 顶点
                    parts = line.strip().split()
                    vertex = list(map(float, parts[1:4]))
                    vertices.append(vertex)
                elif line.startswith('f '):  # 面
                    parts = line.strip().split()
                    face = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                    faces.append(face)

        return np.array(vertices), np.array(faces)


    def _return_object_dict(self, obj_pose_world, vertices, faces, cam_extrinsic, device = "cpu"):
        obj_rotmat = obj_pose_world[:, :3, :3]  # (T, 3, 3)
        obj_trans = obj_pose_world[:, :3, 3]  # (T, 3)

        obj_rotmat_cam, obj_trans_cam = get_pose12d_from_world_to_camera(
                    torch.tensor(obj_rotmat, dtype=torch.float32).to(device), 
                    torch.tensor(obj_trans, dtype=torch.float32).to(device),
                    torch.tensor(cam_extrinsic, dtype=torch.float32).to(device)
                )
        
        world_object_verts = np.einsum('nc,tcd->tnd', vertices, obj_rotmat.transpose(0, 2, 1)) + obj_trans[:, None, :]
        cam_object_verts = get_pc_from_world_to_camera(
            torch.tensor(world_object_verts, dtype=torch.float32).to(device),
            torch.tensor(cam_extrinsic, dtype=torch.float32).to(device)
        ).cpu().numpy()

        object_dict = dict()
        object_dict[0] = dict(
                verts=dict(
                    world=world_object_verts,
                    cam=cam_object_verts
                ),
                faces=faces,
                pose_seqs=dict(
                    world=dict(
                        rot= obj_rotmat,    
                        trans = obj_trans,
                    ),
                    cam=dict(
                        rot= obj_rotmat_cam,
                        trans = obj_trans_cam,
                    )
                ),
                masks=None,  # (T, 1), whether visible object
            )
        
        return object_dict

    
    def _load_cam_extrinsic(self, cam_extrinsic_path):
        cam_extrinsic = []
        for cam_extrinsic_filename in cam_extrinsic_path:
            with open(cam_extrinsic_filename, 'r') as f:
                line = f.readline().strip()
                nums = list(map(float, line.split()))
                cam_extrinsic.append(np.array(nums).reshape(4, 4))
        return np.stack(cam_extrinsic, axis=0)


    def _read_intrinsics(self, file_path):
        with open(file_path, 'r') as f:
            line = f.readline().strip()
            fx, fy, cx, cy, width, height = map(float, line.split())
        
        K = np.array([
            [fx,  0,  cx],
            [0,   fy, cy],
            [0,   0,  1 ]
        ])
        
        return K

    
    def _load_object_pose(self, object_pose_RT_filenames):

        object_pose_list = []
        for object_pose_RT_filename in object_pose_RT_filenames:
            with open(object_pose_RT_filename, 'r') as f:
                line = f.readline().strip()  # 只读取一行并去除换行符
                nums = list(map(float, line.split()))
                if len(nums) != 17:
                    raise ValueError("这行不包含 17 个数字，格式可能有误")
                obj_class = int(nums[0])
                matrix = np.array(nums[1:]).reshape(4, 4)
                object_pose_list.append(matrix)
        
        return obj_class, np.stack(object_pose_list, axis=0)

                



if __name__ == "__main__":
    h2o_dataset = H2ODataset()
    h2o_dataset.initialize_data_storage(H2OStorage, dict(window_size=30, side='right'))
    print(len(h2o_dataset))
    batch = h2o_dataset[0]  # batch_size=1
    breakpoint()