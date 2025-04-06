import json
import os
import numpy as np
import open3d as o3d
import pandas as pd
from tqdm import tqdm
import trimesh
from dataset_loader.storage import Storage, MetaDataset
from utils.rot import get_pose12d_from_world_to_camera, batch_rodrigues
from utils.util import mano_forward, mp42imagefiles, ari2imagefiles
from glob import glob
from natsort import natsorted
from scipy.spatial.transform import Rotation as Rt
import torch
import pickle
from collections import defaultdict
import re
from utils.pc import VisOpen3D, get_pc_from_world_to_camera
from smplx import MANO
from PIL import Image
from scipy.spatial.transform import Rotation as R


class HOI4DStorage(Storage):
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

class HOI4DDataset(MetaDataset):
    def __init__(self, 
        root = "/media/ps/zhan/DATASET/HOI4D/",
        window_size=30,
    ):
        super().__init__(root)

        self.root = root

        self.mapping = [
            '', 'ToyCar', 'Mug', 'Laptop', 'StorageFurniture', 'Bottle',
            'Safe', 'Bowl', 'Bucket', 'Scissors', '', 'Pliers', 'Kettle',
            'Knife', 'TrashCan', '', '', 'Lamp', 'Stapler', '', 'Chair'
        ]
        self.mappting_type = {
            "articulated": ['Bucket', 'Lamp',  'Laptop', 'Pliers', 'Safe', 'Scissors', 'Stapler', 'StorageFurniture', 'TrashCan'],
            "rigid": ['ToyCar', 'Mug', 'Bottle', 'Knife', 'Chair', 'Bowl', 'Kettle'],
        }

        with open(os.path.join(self.root, 'task_list.txt'), 'r') as f:
            self.task_list = [line.strip() for line in f]

        # 测试
        self.task_list = ["ZY20210800004/H4/C14/N30/S335/s02/T2", "ZY20210800001/H1/C13/N17/S179/s05/T4"]

        self.task_defination = pd.read_csv(os.path.join(self.root, "definitions", "task", "task_definitions.csv" ))

    def read_rtd(self, file):
        """
        读取一个 .rtd 文件，返回一个 dict，key 是 object id（从 0 开始），
        value 是 {'rot': rotation_vector, 'trans': translation_vector}
        
        rot: [3] numpy array (axis-angle)
        trans: [3] numpy array
        """
        with open(file, 'r') as f:
            cont = eval(f.read())

        # 获取所有 object（dataList or objects）
        if "dataList" in cont:
            annos = cont["dataList"]
        else:
            annos = cont["objects"]

        result = {}

        for i, anno in enumerate(annos):
            trans = anno["center"]
            rot = anno["rotation"]

            trans = np.array([trans['x'], trans['y'], trans['z']], dtype=np.float32)
            rot = np.array([rot['x'], rot['y'], rot['z']], dtype=np.float32)
            rot_matrix = Rt.from_euler('XYZ', rot).as_matrix().astype(np.float32)  # [3, 3]

            result[i] = {
                'rot': rot_matrix,     # shape: [3]
                'trans': trans      # shape: [3]
            }

        return result
    
    def split_into_sequences_by_filename(self, file_paths):

        """
        输入是一系列文件路径，按文件名中的数字顺序排序，
        并根据是否连续，拆成若干段连续序列（list of list）。
        """
        # 提取文件名和编号
        files_with_numbers = []
        for path in file_paths:
            filename = os.path.basename(path)
            number = int(re.findall(r'\d+', filename)[0])  # 只从文件名中提取数字
            files_with_numbers.append((number, path))

        # 按数字排序
        files_with_numbers.sort(key=lambda x: x[0])

        # 按连续性划分序列
        sequences = []
        current_seq = []

        for i, (number, path) in enumerate(files_with_numbers):
            if not current_seq:
                current_seq.append(path)
            else:
                prev_number = int(re.findall(r'\d+', os.path.basename(current_seq[-1]))[0])
                if number == prev_number + 1:
                    current_seq.append(path)
                else:
                    sequences.append(current_seq)
                    current_seq = [path]

        if current_seq:
            sequences.append(current_seq)

        return sequences
    
    def load_mano_hand(self, pkl_files_path, extrinsic_matrix_sequence, hand_type='right', device='cpu'):
        
        theta_list =[]
        beta_list =[]
        trans_list =[]
        rotation_list =[]

        for pkl_file in pkl_files_path:
            # Load hand parameters from .pkl file
            with open(pkl_file, 'rb') as f:
                hand_info = pickle.load(f, encoding='latin1')

            rotation = hand_info['poseCoeff'][:3]                         # shape: [3]
            mano_pose = hand_info['poseCoeff'][3:]                               # shape: [45]

            theta_list.append(torch.FloatTensor(mano_pose).unsqueeze(0))
            beta_list.append(torch.FloatTensor(hand_info['beta']).unsqueeze(0))
            trans_list.append(torch.FloatTensor(hand_info['trans']).unsqueeze(0))
            rotation_list.append(torch.FloatTensor(rotation).unsqueeze(0))

        theta = torch.cat(theta_list, dim=0).to(device) #(T, 45)
        beta = torch.cat(beta_list, dim=0).to(device)   #(T, 10)
        trans = torch.cat(trans_list, dim=0).to(device) #(T, 3)
        rot = torch.cat(rotation_list, dim=0).to(device) #(T, 3)
        rot_mat = batch_rodrigues(rot)  # shape: [T, 3, 3]

        mano_model = MANO(
            "manopth/mano/models",
            use_pca=False,
            flat_hand_mean=True,
            create_transl=True,
            is_rhand=hand_type
        ).to(device)
    
        output = mano_model(
            global_orient=rot.to(device),
            hand_pose=theta.to(device),
            betas=beta.to(device),
            transl=trans.to(device)
        )

        verts = output.vertices  # shape: [T, 778, 3]
        joints = output.joints  # shape: [T, 16, 3]

        # verts, joints, hand_faces = mano_forward(theta, beta, rot, trans, hand_type, device)

        verts_cam = get_pc_from_world_to_camera(verts.to(device), extrinsic_matrix_sequence.to(device))

        # faces
        mano_root = "manopth/mano/models"
        face_path = f"{mano_root}/{hand_type}_faces.npy"
        hand_faces = np.load(face_path)

        rot_cam, trans_cam = get_pose12d_from_world_to_camera(rot_mat.to(device), trans.to(device), extrinsic_matrix_sequence.to(device))

        def to_numpy(*tensors):
            return [t.detach().cpu().numpy() for t in tensors]

        return *to_numpy(theta, beta, rot_mat, trans, rot_cam, trans_cam, verts, verts_cam, joints), hand_faces

    def hand_data_to_dict(self, theta, beta, rot, trans, rot_cam, trans_cam, verts, verts_cam, joints, faces):
        
        return dict(
                    theta=theta,  # (T, 45)
                    beta=beta[0],    # (10,)
                    pose_seqs=dict(
                        world=dict(
                            rot=rot,     # (T, 3, 3)
                            trans=trans, # (T, 3)
                        ),
                        cam=dict(
                            rot=rot_cam,  # (T, 3, 3)
                            trans=trans_cam,# (T, 3)
                        ),
                    ),
                    mesh=dict(
                        verts=dict(
                            world=verts,  # (T, 778, 3)
                            cam = verts_cam,
                        ),
                        faces=faces,
                        keypoints=dict(
                            world=joints  # (T, 21, 3)
                        )
                    ),
                    masks=None,  # (T, 1), whether visible hand
                )

    def load_task_generate_object_pose_sequence_from_verts(
            self,
            frame_pose_verts,
            mesh_dict,
            extrinsic_matrix_sequence,
            device='cpu'
        ):

        pose_dict = defaultdict(lambda: {'rot': [], 'trans': []})

        for frame_data in frame_pose_verts:
            for obj_id, pose in frame_data.items():
                pose_dict[obj_id]['rot'].append(pose['rot'])       # 已是 3x3 matrix
                pose_dict[obj_id]['trans'].append(pose['trans'])   # 已是 3D vector

        for obj_id in pose_dict:
            rot_seq = np.stack(pose_dict[obj_id]['rot'])      # [T, 3, 3]
            trans_seq = np.stack(pose_dict[obj_id]['trans'])  # [T, 3]
            pose_dict[obj_id]['rot'] = rot_seq
            pose_dict[obj_id]['trans'] = trans_seq

        object_pose_verts = dict(pose_dict)
        
        object_dict = dict()

        for obj_id, pose in object_pose_verts.items():
            rot = pose['rot']     # [T, 3, 3]
            trans = pose['trans'] # [T, 3]
            mesh_verts = mesh_dict[obj_id].vertices  # [N, 3], numpy array

            obj_rotmat_cam, obj_trans_cam = get_pose12d_from_world_to_camera(
                    torch.tensor(rot).to(device), 
                    torch.tensor(trans).to(device),
                    torch.tensor(extrinsic_matrix_sequence, dtype=torch.float32).to(device)
                )

            # 旋转 + 平移（world 坐标系）
            world_object_verts = np.einsum('nc,tcd->tnd', mesh_verts, rot.transpose(0, 2, 1)) + trans[:, None, :]

            # 转换到相机坐标
            cam_object_verts = get_pc_from_world_to_camera(
                torch.tensor(world_object_verts, dtype=torch.float32).to(device),
                torch.tensor(extrinsic_matrix_sequence, dtype=torch.float32).to(device)
            ).cpu().numpy()
            
            object_dict[obj_id] = dict(
                verts=dict(
                    world=world_object_verts,
                    cam=cam_object_verts
                ),
                faces=mesh_dict[obj_id].faces,
                pose_seqs=dict(
                    world=dict(
                        rot= rot,
                        trans = trans,
                    ),
                    cam=dict(
                        rot= obj_rotmat_cam,
                        trans = obj_trans_cam,
                    )
                ),
                masks=None,  # (T, 1), whether visible object
            )
        
        return object_dict
    
    def get_mesh_paths_from_result(self, base_dir):
        """
        读取 result.json，返回所有 mesh 文件的完整路径。
        :return: list of mesh file paths
        """
        result_json_path = os.path.join(base_dir, "result.json")
        with open(result_json_path, 'r') as f:
            result = json.load(f)

        mesh_paths = []

        def traverse(node):
            if "objs" in node:
                for obj_name in node["objs"]:
                    mesh_path = os.path.join(base_dir, "objs", f"{obj_name}-align.obj")
                    mesh_paths.append(mesh_path)
            if "children" in node:
                for child in node["children"]:
                    traverse(child)

        for top_node in result:
            traverse(top_node)

        return mesh_paths

    def load_task(self, task_name = "ZY20210800004/H4/C14/N30/S335/s02/T2"):

        camera_id = task_name.split("/")[0]
        task_id = task_name.split("/")[-1]

        object_path = os.path.join(self.root, "HOI4D_CAD_Model_for_release")
        object_class_id = task_name.split("/")[2]
        object_class = object_class_id.split("C")[1]
        object_name = self.mapping[int(object_class)]

        object_type = "articulated" if object_name in self.mappting_type["articulated"] else "rigid"
        object_inedx = task_name.split("/")[3].split("N")[1]

        mesh_dict = {}
        if object_type == "rigid":
            object_path = os.path.join(object_path, object_type, object_name)
            obj_files = glob(os.path.join(object_path, '**', '*.obj'), recursive=True)
            object_specific_name = object_inedx + ".obj"
            for obj_file in obj_files:
                if object_specific_name in obj_file:
                    mesh = trimesh.load(obj_file)
                    mesh_dict[0] = mesh
                    break

        elif object_type == "articulated":
            object_path = os.path.join(object_path, object_type, object_name, "0"+object_inedx)
            mesh_paths = self.get_mesh_paths_from_result(object_path)
            mesh_dict = {}
            for i in range(len(mesh_paths)):
                mesh = trimesh.load(mesh_paths[i])
                mesh_dict[i] = mesh

        task_defination = os.path.join(self.root, "definitions", "task", "task_definitions.csv" )
        task_defination = pd.read_csv(task_defination)
        instruction = task_defination.loc[task_defination['Category ID'] == object_class_id, task_id].iloc[0]

        depth_path = os.path.join(self.root, "HOI4D_depth_video", task_name,"align_depth", "depth_video.avi")
        save_depth_frames_dir = os.path.join(self.root, "HOI4D_depth_video", task_name, "depth_image")
        ari2imagefiles(depth_path, save_depth_frames_dir) if os.path.exists(depth_path) else None
        image_depth_filenames = natsorted(glob(os.path.join(save_depth_frames_dir, "*.png"))) if os.path.exists(save_depth_frames_dir) else []

        rgb_path = os.path.join(self.root, "HOI4D_release", task_name, "align_rgb", "image.mp4")
        save_rgb_frames_dir = os.path.join(self.root, "HOI4D_release", task_name, "rgb_image")
        mp42imagefiles(rgb_path, save_rgb_frames_dir)
        image_rgb_filenames = natsorted(glob(os.path.join(save_rgb_frames_dir, "*.jpg")))

        # 相机内参
        camera_intrin_path = os.path.join(self.root, "camera_params", camera_id, "intrin.npy")
        camera_intrin = np.load(camera_intrin_path) # 3x3

        camera_parameters_path = os.path.join(self.root, "HOI4D_annotations", task_name, "3Dseg", "output.log")
        camera_trajectory = o3d.io.read_pinhole_camera_trajectory(camera_parameters_path)
        extrinsics_list = [np.asarray(param.extrinsic) for param in camera_trajectory.parameters]

        object_pose_path = os.path.join(self.root, "HOI4D_annotations", task_name, "objpose")
        object_pose_files = glob(os.path.join(object_pose_path, "**/*.json"), recursive=True)
        object_pose_files = natsorted(object_pose_files)

        # 确定有多少物体，num = 0 是指第一个物体
        object_pose_list = [self.read_rtd(file) for file in object_pose_files]

        right_hand_pose = os.path.join(self.root, "HOI4D_Hand_pose", "Hand_pose", "handpose_right_hand", task_name)
        right_hand_pose_files = glob(os.path.join(right_hand_pose, "**/*.pickle"), recursive=True)
        right_hand_pose_files = natsorted(right_hand_pose_files)
        split_right_hand_pose_files = self.split_into_sequences_by_filename(right_hand_pose_files)

        left_hand_pose = os.path.join(self.root, "HOI4D_Hand_pose", "Hand_pose", "handpose_left_hand", task_name)
        left_hand_pose_files = glob(os.path.join(left_hand_pose, "**/*.pickle"), recursive=True)
        left_hand_pose_files = natsorted(left_hand_pose_files)
        split_left_hand_pose_files = self.split_into_sequences_by_filename(left_hand_pose_files)

        def _load_task_process_hand_data(begin_idx, end_idx, left_files, right_files):

            image_rgb_split = image_rgb_filenames[begin_idx:end_idx+1]
            image_depth_split = image_depth_filenames[begin_idx:end_idx+1] if image_depth_filenames else []

            extrinsic_matrix_sequence = torch.tensor(np.stack(extrinsics_list[begin_idx:end_idx+1]), dtype=torch.float32)

            right_hand_data = self.load_mano_hand(right_files, extrinsic_matrix_sequence) if right_files else None
            left_hand_data = self.load_mano_hand(left_files, extrinsic_matrix_sequence) if left_files else None

            hand_dict = dict()
            hand_dict['left'] = self.hand_data_to_dict(*left_hand_data) if left_hand_data else None
            hand_dict['right'] = self.hand_data_to_dict(*right_hand_data) if right_hand_data else None

            
            # 处理物体数据
            object_dict = self.load_task_generate_object_pose_sequence_from_verts(
                object_pose_list[begin_idx:end_idx+1], mesh_dict, extrinsic_matrix_sequence
            )
            render_depth_filenames = None
            '''
            if self.data_storageis_render_depth and not image_depth_filenames:
                render_depth_filenames = [os.join(save_depth_frames_dir, f"{i:06d}.png") for i in range(len(image_rgb_split))]
                vis = VisOpen3D(*Image.open(image_rgb_split[0]).size, visible=False)
                for i, rdf in tqdm(enumerate(render_depth_filenames), desc=f"Rendering depth images for {task_name}", total=len(render_depth_filenames)):
                    vis.save_rgbd_from_3d(
                        meshes_list=[
                            dict(verts=object_dict[0]['verts']['world'][i], faces=object_dict[0]['faces']),
                            *[dict(verts=hand_dict[side]['mesh']['verts']['world'][i], faces=hand_dict[side]['mesh']['faces']) for side in ['left', 'right'] if hand_dict[side] is not None],
                        ],
                        intrinsic=camera_intrin, extrinsic=extrinsic_matrix_sequence[i],
                        save_path=dict(D=rdf, RGB=None)
                    )
                vis.destroy_window()  
            '''
            hand_ls = [hand_dict[side]['pose_seqs']['cam']['rot'] for side in ['left', 'right'] if hand_dict.get(side, None) is not None]
            obj_ls = [object_dict[obj_id]['pose_seqs']['cam']['rot'] for obj_id in object_dict.keys()]

            # check length for each input items
            num_frames = self.check_length(
                image_rgb_split, 
                image_depth_split,
                *hand_ls,
                *obj_ls,
                extrinsic_matrix_sequence
            )    

            return dict(
            # camera: for debug
            intrinsic_matrix=dict(ego=camera_intrin, allo=None), extrinsic_matrix_sequence=dict(ego=extrinsic_matrix_sequence.cpu().numpy(), allo=None),
            # real images
            real_image_filenames=dict(ego_rgb=image_rgb_split, ego_depth=image_depth_split, ego_seg=None, allo_rgb=None, allo_depth=None, allo_seg=None),  
            # hand-object mesh projection rendered images
            render_image_filenames=dict(ego_rgb=None, ego_depth=render_depth_filenames),  
            # hand
            hand_dict=hand_dict,
            # object
            object_dict=object_dict,
            # length
            length=num_frames,
            # text
            task_name=task_name,
            instruction=instruction, 
        )


        task_data_list = []
        if not split_left_hand_pose_files and split_right_hand_pose_files:
        # 只有右手数据
            for i, right_files in enumerate(split_right_hand_pose_files):
                numbers = [int(re.findall(r'\d+', os.path.basename(path))[0]) for path in right_files]
                begin_idx, end_idx = min(numbers), max(numbers)

                task_data_list.append(_load_task_process_hand_data(begin_idx, end_idx, [], right_files))
                
        elif len(split_left_hand_pose_files) == len(split_right_hand_pose_files):
            # 左右手数据都存在
            for i in range(len(split_right_hand_pose_files)):

                left_files_all = {os.path.basename(p): p for p in split_left_hand_pose_files[i]}
                right_files_all = {os.path.basename(p): p for p in split_right_hand_pose_files[i]}
                common_names = set(left_files_all.keys()) & set(right_files_all.keys())

                # 提取编号并排序
                common_files = [(int(re.findall(r'\d+', name)[0]), name) for name in common_names]
                common_files.sort()

                begin_idx, end_idx = common_files[0][0], common_files[-1][0]
                sorted_names = [name for _, name in common_files]
                left_files = [left_files_all[name] for name in sorted_names]
                right_files = [right_files_all[name] for name in sorted_names]

                task_data_list.append(_load_task_process_hand_data(begin_idx, end_idx, left_files, right_files))

  
        return task_data_list
    

if __name__ == "__main__":
    hoi4d_dataset = HOI4DDataset()
    hoi4d_dataset.initialize_data_storage(HOI4DStorage, dict(window_size=30, side='right', is_render_depth=False))
    print(len(hoi4d_dataset))
    batch = hoi4d_dataset[0]  # batch_size=1
    breakpoint()