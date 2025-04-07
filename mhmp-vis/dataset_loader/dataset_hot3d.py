from glob import glob
import json
import os

from natsort import natsorted
import numpy as np
import torch
import trimesh
from dataset_loader.storage import Storage, MetaDataset
from scipy.spatial.transform import Rotation as R
from utils.pc import VisOpen3D, get_pc_from_world_to_camera
from utils.rot import batch_rodrigues, get_pose12d_from_world_to_camera
from PIL import Image
from tqdm import tqdm
import smplx

class HOT3DStorage(Storage):
    def __init__(self, **config):
        super().__init__(**config)
    
    def fill_data(self, task_data):
        if task_data["hand_dict"][self.side] is not None:
            return dict(
                text=task_data["instruction"],
                ego_rgb_files=task_data["real_image_filenames"]['ego_rgb'],                 
                ego_dep_files=task_data["render_image_filenames"]['ego_depth'],                        
                theta_seqs=task_data["hand_dict"][self.side]['theta'],                      # (T, 15*3)
                beta_seqs=task_data["hand_dict"][self.side]['beta'].reshape(-1,10),         # (1, 10)
                rot_seqs=task_data["hand_dict"][self.side]['pose_seqs']['cam']['rot'],      # (T, 3, 3)
                trans_seqs=task_data["hand_dict"][self.side]['pose_seqs']['cam']['trans'],  # (T, 3)
                len=task_data["length"],
            )

class HOT3DDataset(MetaDataset):
    def __init__(self, 
        root = "/media/ps/T9/hot3d/",
    ):
        super().__init__(root)

        self.root = root
        with open(os.path.join(self.root, 'task_list.txt'), 'r') as f:
            self.task_list = [line.strip() for line in f]
        self.task_list = ["train_aria/clip-001849", "train_quest3/clip-000000"]
        self.all_camera_ids = ['214-1', '1201-1', '1201-2']
    
    def load_task(self, task_name="train_quest3/clip-000000"):
        camera_id='214-1' if 'aria' in task_name else '1201-2'
        task_path = os.path.join(self.root, task_name)
        hand_shapes_json_file = os.path.join(task_path, "__hand_shapes.json__")
        with open(hand_shapes_json_file, 'r') as f:
            beta = np.array(json.load(f).get("mano", None))

        # load camera matrices
        extrin_matrices = []
        intrin_matrices = []
        for file_path in natsorted(glob(os.path.join(task_path, "*.cameras.json"))):
            with open(file_path, "r") as f:
                camera_data = json.load(f)
            info = camera_data[camera_id]
            # 内参
            proj = info["calibration"]["projection_params"]
            fx, cx, cy = proj[0], proj[1], proj[2]
            K = np.array([
                [fx,  0, cx],
                [ 0, fx, cy],
                [ 0,  0,  1]
            ])
            intrin_matrices.append(K)
            # 外参
            trans_wc = np.array(info["T_world_from_camera"]["translation_xyz"])
            quat_wc = np.array(info["T_world_from_camera"]["quaternion_wxyz"])
            '''
            trans_dc = np.array(info["calibration"]["T_device_from_camera"]["translation_xyz"])
            quat_dc = np.array(info["calibration"]["T_device_from_camera"]["quaternion_wxyz"])
            device_to_camera = build_extrinsic_matrix(trans_dc, quat_dc)
            world_to_device = world_to_camera @ camera_to_device  
            '''
            world_to_camera = np.concatenate([np.concatenate([
                R.from_quat([quat_wc[1],quat_wc[2],quat_wc[3],quat_wc[0]]).as_matrix(),
                np.array(trans_wc).reshape(3, 1)], axis=1), 
            [[0, 0, 0, 1]]], axis=0)

            camera_to_world = np.linalg.inv(world_to_camera)
            extrin_matrices.append(camera_to_world)

        extrin_matrices, intrin_matrices = np.stack(extrin_matrices), intrin_matrices[0] # (N, 4, 4), (3, 3)

        hand_json_files = natsorted(glob(os.path.join(task_path, "*.hands.json")))
        left_hand_data = self._load_all_hand_params(beta, hand_json_files, torch.tensor((extrin_matrices), dtype=torch.float32), "left")
        right_hand_data = self._load_all_hand_params(beta, hand_json_files, torch.tensor((extrin_matrices), dtype=torch.float32), "right")

        hand_dict = dict()
        hand_dict['left'] = self.hand_data_to_dict(*left_hand_data) if left_hand_data else None
        hand_dict['right'] = self.hand_data_to_dict(*right_hand_data) if right_hand_data else None

        object_json_files = natsorted(glob(os.path.join(task_path, "*.objects.json")))
        object_dict = self._load_all_object_poses(object_json_files, torch.tensor((extrin_matrices), dtype=torch.float32))

        image_rgb_split = natsorted(glob(os.path.join(task_path, f"*.image_{camera_id}.jpg")))
        save_depth_frames_dir = os.path.join(task_path, "depth_image")

        render_depth_filenames = [os.path.join(save_depth_frames_dir, f"{i:06d}.png") for i in range(len(image_rgb_split))]
        # render depth
        vis = VisOpen3D(*Image.open(image_rgb_split[0]).size, visible=False)
        for i, rdf in tqdm(enumerate(render_depth_filenames), desc=f"Rendering depth images for {task_name}", total=len(render_depth_filenames)):
            vis.save_rgbd_from_3d(
                meshes_list=[
                    dict(verts=object_dict[0]['verts']['world'][i], faces=object_dict[0]['faces']),
                    *[dict(verts=hand_dict[side]['mesh']['verts']['world'][i], faces=hand_dict[side]['mesh']['faces']) for side in ['left', 'right'] if hand_dict[side] is not None],
                ],
                intrinsic=intrin_matrices, extrinsic=extrin_matrices[i],
                save_path=dict(D=rdf, RGB=None)
            )
        vis.destroy_window()  

        hand_ls = [hand_dict[side]['pose_seqs']['cam']['rot'] for side in ['left', 'right'] if hand_dict.get(side, None) is not None]
        obj_ls = [object_dict[obj_id]['pose_seqs']['cam']['rot'] for obj_id in object_dict.keys()]

        # check length for each input items
        num_frames = self.check_length(
            image_rgb_split, 
            render_depth_filenames,
            *hand_ls,
            *obj_ls,
            extrin_matrices
        )    


        return dict(
            # camera: for debug
            intrinsic_matrix=dict(ego=intrin_matrices, allo=None), extrinsic_matrix_sequence=dict(ego=extrin_matrices, allo=None),
            # real images
            real_image_filenames=dict(ego_rgb=image_rgb_split, ego_depth=None, ego_seg=None, allo_rgb=None, allo_depth=None, allo_seg=None),  
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
            instruction="", 
        )

    def _load_all_hand_params(self, beta, file_list, extrin_matrices, hand_type = "right", device = "cpu"):
        rotations = []
        translations = []
        full_thetas = []
        full_thetas15 = []

        is_rhand = True if hand_type == "right" else False
        mano_path = "manopth/mano/models/MANO_RIGHT.pkl" if is_rhand else "manopth/mano/models/MANO_LEFT.pkl"

        mano_model = smplx.create(
            mano_path,
            "mano",
            use_pca=True,
            is_rhand=is_rhand,
            num_pca_comps=15,
        )

        for file_path in file_list:
            with open(file_path) as f:
                hand_data = json.load(f).get(hand_type, None)
                if hand_data is None:
                    return None, None, None
                else:
                    pose = hand_data["mano_pose"]
                    wrist_xform = torch.FloatTensor(pose["wrist_xform"])
                    rotvec = wrist_xform[:3]
                    trans = wrist_xform[3:6]
                    theta15 = torch.FloatTensor(pose["thetas"])
                    pca_basis = mano_model.hand_components  # (6, 45)
                    theta45 = theta15 @ pca_basis  # (6,) x (6, 45) → (45,)

                rotations.append(rotvec.unsqueeze(0))
                translations.append(trans.unsqueeze(0))
                full_thetas.append(theta45.unsqueeze(0))
                full_thetas15.append(theta15.unsqueeze(0))
        
        theta = torch.cat(full_thetas, dim=0).to(device) #(T, 45)
        theta15 = torch.cat(full_thetas15, dim=0).to(device) #(T, 15)
        beta = torch.FloatTensor(beta).unsqueeze(0).repeat(theta.shape[0], 1).to(device)
        trans = torch.cat(translations, dim=0).to(device) #(T, 3)
        rot = torch.cat(rotations, dim=0).to(device) #(T, 3)
        rot_mat = batch_rodrigues(rot)  # shape: [T, 3, 3]

        output = mano_model(
            betas=beta,
            global_orient=rot,
            hand_pose=theta15,
            transl=trans,
        )

        verts = output.vertices  # shape: [T, 778, 3]
        joints = output.joints  # shape: [T, 16, 3]

        verts_cam = get_pc_from_world_to_camera(verts.to(device), extrin_matrices.to(device))

        # faces
        mano_root = "manopth/mano/models"
        face_path = f"{mano_root}/{hand_type}_faces.npy"
        hand_faces = np.load(face_path)

        rot_cam, trans_cam = get_pose12d_from_world_to_camera(rot_mat.to(device), trans.to(device), extrin_matrices.to(device))

        def to_numpy(*tensors):
            return [t.detach().cpu().numpy() for t in tensors]

        return *to_numpy(theta, beta, rot_mat, trans, rot_cam, trans_cam, verts, verts_cam, joints), hand_faces

    def _load_all_object_poses(self, file_list, extrinsic_matrix_sequence, device = "cpu"):
        
        object_data = {}

        for file_path in file_list:
            with open(file_path, "r") as f:
                data = json.load(f)

            for obj_id, obj_info_list in data.items():
                obj_info = obj_info_list[0]
                
                trans = np.array(obj_info["T_world_from_object"]["translation_xyz"])  # (3,)
                quat = np.array(obj_info["T_world_from_object"]["quaternion_wxyz"])   # (w, x, y, z)
                rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()   # (3, 3)
                
                if obj_id not in object_data:
                    object_data[obj_id] = {"rot": [], "trans": []}

                object_data[obj_id]["rot"].append(rot)
                object_data[obj_id]["trans"].append(trans)
            
        for obj_id in object_data:
            object_data[obj_id]["rot"] = np.stack(object_data[obj_id]["rot"])
            object_data[obj_id]["trans"] = np.stack(object_data[obj_id]["trans"])

        
        object_dict = dict()

        index = 0
        for obj_id, pose in object_data.items():
            rot = pose['rot']     # [T, 3, 3]
            trans = pose['trans'] # [T, 3]
            
            object_mesh_path = os.path.join(self.root, "object_models", f"obj_{int(obj_id):06d}.glb")
            mesh_or_scene = trimesh.load(object_mesh_path)
            if isinstance(mesh_or_scene, trimesh.Scene):
                mesh = trimesh.util.concatenate(mesh_or_scene.dump())
            else:
                mesh = mesh_or_scene
            verts = mesh.vertices  # [N, 3], numpy array
            faces = mesh.faces  # [F, 3], numpy array

            obj_rotmat_cam, obj_trans_cam = get_pose12d_from_world_to_camera(
                torch.tensor(rot, dtype=torch.float32).to(device), 
                torch.tensor(trans, dtype=torch.float32).to(device),
                torch.tensor(extrinsic_matrix_sequence, dtype=torch.float32).to(device)
            )

            # 旋转 + 平移（world 坐标系）
            world_object_verts = np.einsum('nc,tcd->tnd', verts, rot.transpose(0, 2, 1)) + trans[:, None, :]

            # 转换到相机坐标
            cam_object_verts = get_pc_from_world_to_camera(
                torch.tensor(world_object_verts, dtype=torch.float32).to(device),
                torch.tensor(extrinsic_matrix_sequence, dtype=torch.float32).to(device)
            ).cpu().numpy()
            
            object_dict[index] = dict(
                verts=dict(
                    world=world_object_verts,
                    cam=cam_object_verts
                ),
                faces=faces,
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
            index += 1
        
        return object_dict

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
    
if __name__ == "__main__":
    hot3d_dataset = HOT3DDataset()
    hot3d_dataset.initialize_data_storage(HOT3DStorage, dict(window_size=30, side='right'))
    print(len(hot3d_dataset))
    batch = hot3d_dataset[0]  # batch_size=1
    