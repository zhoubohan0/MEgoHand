import json
import os
from glob import glob

import numpy as np
import trimesh
from matplotlib import pyplot as plt
from PIL import Image
import open3d as o3d
import torch
from natsort import natsorted
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset_loader.storage import Storage, MetaDataset
from utils.pc import get_pc_from_world_to_camera, VisOpen3D
from utils.rot import get_pose12d_from_world_to_camera, batch_rodrigues
from utils.util import mano_forward

from inv_mano import DirectInvMANOTrainer


# Loading utilities
def load_objects(obj_root):
    object_names = ['milk']#['juice', 'liquid_soap', 'milk', 'salt']
    all_models = {}
    for obj_name in object_names:
        obj_path = os.path.join(obj_root, '{}_model'.format(obj_name), '{}_model.ply'.format(obj_name))
        mesh = trimesh.load(obj_path)
        all_models[obj_name] = {
            'verts': np.array(mesh.vertices),  # * 1000, # m convet to mm
            'faces': np.array(mesh.faces)
        }
    return all_models


def get_skeleton(skeleton_path):
    skeleton_vals = np.loadtxt(skeleton_path)
    skeleton = skeleton_vals[:, 1:].reshape(skeleton_vals.shape[0], 21, -1) / 1000
    reorder_idx = np.array([0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19, 20])
    skeleton = skeleton[:, reorder_idx]
    return skeleton

def get_obj_transform_all_frames(seq_path):
    """
    读取 object_pose.txt 中所有帧的变换矩阵
    返回形状为 (T, 4, 4) 的 numpy 数组
    """
    with open(seq_path, 'r') as seq_f:
        raw_lines = seq_f.readlines()

    all_transforms = []
    for line in raw_lines:
        parts = line.strip().split(' ')
        mat = np.array(parts[1:]).astype(np.float32).reshape(4, 4).T  # transpose
        all_transforms.append(mat)

    all_transforms = np.stack(all_transforms, axis=0)  # shape: (T, 4, 4)
    all_transforms[:, :3, 3] /= 1000  # mm
    return all_transforms


# Display utilities
def _visualize_joints_2d(ax, joints, joint_idxs=True, links=None, alpha=1):
    """Draw 2d skeleton on matplotlib axis"""
    if links is None:
        links = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                 (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]
    # Scatter hand joints on image
    x = joints[:, 0]
    y = joints[:, 1]
    ax.scatter(x, y, 1, 'r')

    # Add idx labels to joints
    for row_idx, row in enumerate(joints):
        if joint_idxs:
            plt.annotate(str(row_idx), (row[0], row[1]))
    _draw2djoints(ax, joints, links, alpha=alpha)


def _draw2djoints(ax, annots, links, alpha=1):
    """Draw segments, one color per link"""
    colors = ['r', 'm', 'b', 'c', 'g']

    for finger_idx, finger_links in enumerate(links):
        for idx in range(len(finger_links) - 1):
            _draw2dseg(
                ax,
                annots,
                finger_links[idx],
                finger_links[idx + 1],
                c=colors[finger_idx],
                alpha=alpha)

def _visualize_tensors_as_pointcloud(tensor_list):
    """
    将 PyTorch Tensor 转换为 Open3D 点云并可视化。
    tensor_list[0] 将按照 links 构造骨架线段。
    """
    links = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
             (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]

    geometries = []

    for i, tensor in enumerate(tensor_list):
        np_points = tensor.cpu().numpy()

        if i == 0:
            # 创建 LineSet 骨架
            lines = []
            for link in links:
                for j in range(len(link) - 1):
                    lines.append([link[j], link[j + 1]])

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(np_points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.paint_uniform_color([0, 1, 0])  # 绿色骨架线

            # 也加上关节点本身（点云形式）
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np_points)
            pcd.paint_uniform_color([1, 0, 0])  # 红色点
            geometries.extend([pcd, line_set])
        else:
            # 其他 tensor 仅作为点云展示
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np_points)
            pcd.paint_uniform_color([0, 0, 1])  # 蓝色点
            geometries.append(pcd)

    o3d.visualization.draw_geometries(geometries)


def _draw2dseg(ax, annot, idx1, idx2, c='r', alpha=1):
    """Draw segment of given color"""
    ax.plot(
        [annot[idx1, 0], annot[idx2, 0]], [annot[idx1, 1], annot[idx2, 1]],
        c=c,
        alpha=alpha)


def _visualize_hand_object_projection(
    img_path,
    skel_proj,
    skel_camcoords,
    verts_proj=None,
    verts_camcoords=None,
):
    """
    可视化手部和物体的 2D 投影 + 3D 多视图投影

    参数：
    - img_path: 图像路径
    - skel_proj: 手部 2D 投影坐标 (21, 2)
    - skel_camcoords: 手部 3D 相机坐标 (21, 3)
    - verts_proj: 物体点 2D 投影 (可选)
    - verts_camcoords: 物体点 3D 相机坐标 (可选)
    """
    fig = plt.figure()

    # ----------- 子图 1：图像 + 手部2D投影点 -------------
    ax = fig.add_subplot(221)
    print('Loading image from {}'.format(img_path))
    img = Image.open(img_path)
    ax.imshow(img)
    _visualize_joints_2d(ax, skel_proj, joint_idxs=False)

    if verts_proj is not None:
        ax.scatter(verts_proj[:, 0], verts_proj[:, 1], alpha=0.01, c='r')

    # ----------- 子图 2-4：不同3D投影视图 -------------
    for proj_idx, (proj_1, proj_2) in enumerate([[0, 1], [1, 2], [0, 2]]):
        ax = fig.add_subplot(2, 2, 2 + proj_idx)
        if proj_idx == 0:
            ax.invert_yaxis()  # 与图像方向一致
        ax.set_aspect('equal')

        if verts_camcoords is not None:
            ax.scatter(verts_camcoords[:, proj_1], verts_camcoords[:, proj_2], s=1)

        _visualize_joints_2d(
            ax,
            np.stack(
                [skel_camcoords[:, proj_1], skel_camcoords[:, proj_2]],
                axis=1),
            joint_idxs=False)

    plt.tight_layout()
    plt.show()

def _load_action_object_mapping(file_path):
    """
    读取动作-物体映射的 txt 文件，并返回 {action_name: object_name} 的字典
    """
    # 检测文件是否存在
    if not os.path.exists(file_path):
        # print(f"Error: File '{file_path}' not found.")
        return None

    action_object_dict = {}

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 跳过表头，从第二行开始解析
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) < 3:
            continue  # 确保至少有 action_id, action_name, object_name

        action_name = parts[1]  # 第二列是 action_name
        object_name = parts[2]  # 第三列是 object_name
        action_object_dict[action_name] = object_name

    return action_object_dict

def _parse_fpha_txt(file_path):

    filename = os.path.basename(file_path).replace("_info.txt", "")

    with open(file_path, 'r') as f:
        lines = f.readlines()

    data = []
    start_parsing = False  # 用于标记何时开始解析数据

    for line in lines:
        line = line.strip()

        # 跳过前面的元数据，找到数据部分的起始点
        if line.startswith("name_action num_instance num_frames"):
            start_parsing = True
            continue

        if start_parsing and line:
            parts = line.split()
            name_action = parts[0]
            num_instance = int(parts[1])
            num_frames = int(parts[2])
            data.append((filename, name_action, num_instance, num_frames))

    return data


def _load_images(file_path):
    image_extensions = ["*.jpeg", "*.jpg", "*.png"]  # 支持多个格式
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(file_path, ext)))  # 递归添加所有匹配文件
    image_files = natsorted(image_files)
    return image_files


class FPHAStorage(Storage):
    def __init__(self, **config):
        super().__init__(**config)
    
    def fill_data(self, task_data):
        if task_data["hand_dict"][self.side] is not None:
            return dict(
                text=task_data["instruction"],
                ego_rgb_files=task_data["real_image_filenames"]['ego_rgb'],                 
                ego_dep_files=task_data["real_image_filenames"]['ego_depth'],               # can be real_image_filenames / render_image_filenames
                theta_seqs=task_data["hand_dict"][self.side]['theta'],                      # (T, 15*3)
                beta_seqs=task_data["hand_dict"][self.side]['beta'].reshape(-1,10),         # (1, 10)
                rot_seqs=task_data["hand_dict"][self.side]['pose_seqs']['cam']['rot'],      # (T, 3, 3)
                trans_seqs=task_data["hand_dict"][self.side]['pose_seqs']['cam']['trans'],  # (T, 3)
                len=task_data["length"],
            )

class FPHADataset(MetaDataset):
    def __init__(self,
        root_path="/media/ps/zhan/DATASET/FPHA/",
        is_render_depth=False
    ):
        super().__init__(root_path)

        self.video_path = os.path.join(root_path, "Video_files")
        self.subject_info_path = os.path.join(root_path, "Subjects_info")
        self.skeleton_root = os.path.join(root_path, "Hand_pose_annotation_v1")
        self.obj_trans_root = os.path.join(root_path, "Object_6D_pose_annotation_v1_1")
        self.obj_root = os.path.join(root_path, 'Object_models')

        # Camera extrinsic & intrinsic
        self.cam_extr = np.array([
            [0.999988496304, -0.00468848412856, 0.000982563360594, 25.7 / 1000],
            [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22 / 1000],
            [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902 / 1000],
            [0, 0, 0, 1]
        ])

        self.cam_intr = np.array([
            [1395.749023, 0, 935.732544],
            [0, 1395.749268, 540.681030],
            [0, 0, 1]
        ])

        # object
        self.action_object_dict = _load_action_object_mapping(os.path.join(self.root_path, "action_object_info.txt"))
        self.object_meshes = load_objects(self.obj_root)

        # inverse mano trainer
        self.inv_mano_trainer = DirectInvMANOTrainer()

        # load task
        self.task_list = [f'{x[0]}/{x[1]}/{x[2]}/{x[3]}' for x in self._load_subject_info()]
        self.task_list = ['Subject_1/close_milk/1/89', 'Subject_1/close_milk/2/78']  # , 'Subject_1/close_milk/3/85', 'Subject_1/close_milk/4/86'

        self.is_render_depth=is_render_depth

    def _load_subject_info(self):

        """
        解析 Subjects_info 目录，获取所有tasks
        """

        subject_files = glob(os.path.join(self.subject_info_path, "Subject_*.txt"))
        subject_list = []
        for f in subject_files:
            subject_list += _parse_fpha_txt(f)
        return natsorted(subject_list)

    def load_task(self, task_name='Subject_1/close_milk/2/78'):
        subject_name, action_name, seq_idx, frame_length = task_name.split('/')
        frame_length = int(frame_length)

        # real egocentric RGB & depth images
        subject_file = os.path.join(self.video_path, subject_name, action_name, seq_idx)
        real_rgb_filenames = _load_images(os.path.join(subject_file, "color"))
        real_depth_filenames = _load_images(os.path.join(subject_file, "depth"))
        
        # camera 
        intrinsic_matrix = self.cam_intr
        extrinsic_matrix_sequence = torch.Tensor(np.tile(self.cam_extr[None, :, :], (frame_length, 1, 1))).cuda()
        extrinsic_matrix_sequence[:, :3, 3] /= 1000

        # hand
        side = 'right'
        hand_dict = dict(left=None)
        # Load hand skeleton: [T, 21, 3] 
        skeleton_path = os.path.join(self.skeleton_root, subject_name, action_name, seq_idx, 'skeleton.txt')
        hand_skeleton_world = get_skeleton(skeleton_path)
        # optimize
        save_file = skeleton_path.replace('skeleton.txt', 'mano.json')
        if not os.path.exists(save_file):  # training
            hand_mano_param = self.inv_mano_trainer.conduct_training(torch.Tensor(hand_skeleton_world).cuda(), side, save_file, is_writer=False)
        else:
            with open(save_file, 'r') as f:
                hand_mano_param = json.load(f)
                hand_mano_param = {k: torch.Tensor(v).cuda() for k, v in hand_mano_param.items()}
        
        hand_theta45, hand_beta, hand_rot, hand_trans = hand_mano_param['theta'], hand_mano_param['beta'], hand_mano_param['rot'], hand_mano_param['trans'], 
        hand_rot9d = batch_rodrigues(hand_rot)  # (T, 3, 3)
        hand_rot9d_cam, hand_trans_cam = get_pose12d_from_world_to_camera(hand_rot9d, hand_trans, extrinsic_matrix_sequence)  # (T, 3, 3), (T, 3)
        hand_vertices, hand_keypoints, hand_faces = mano_forward(hand_theta45, hand_beta, hand_rot, hand_trans, side)
        hand_vertices_cam = get_pc_from_world_to_camera(hand_vertices, extrinsic_matrix_sequence)  # (T, 778, 3)
        hand_dict[side] = dict(
            theta=hand_theta45.cpu().numpy(),  # (T, 45)
            beta=hand_beta[0].cpu().numpy(),    # (10,)
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
            mesh=dict(
                verts=dict(
                    world=hand_vertices.detach().cpu().numpy(),  # (T, 778, 3)
                    cam=hand_vertices_cam.detach().cpu().numpy(),  # (T, 778, 3)
                ),
                faces=hand_faces.detach().cpu().numpy(),     # (1538, 3)
                keypoints=dict(
                    world=hand_keypoints.detach().cpu().numpy(),  # (T, 21, 3)
                )
            ),
            masks=None,  # (T, 1), whether visible hand
        )
        
        # object
        object_dict = dict()
        obj_faces = None
        object_model = self.object_meshes.get(self.action_object_dict[action_name])
        if object_model is not None:            
            # object pose
            obj_pose_path = os.path.join(self.obj_trans_root, subject_name, action_name, seq_idx, 'object_pose.txt')
            obj_pose_world = get_obj_transform_all_frames(obj_pose_path)  # [T, 4, 4]
            obj_rotmat = obj_pose_world[:, :3, :3]  # (T, 3, 3)
            obj_trans = obj_pose_world[:, :3, 3]  # (T, 3)
            obj_rotmat_cam, obj_trans_cam = get_pose12d_from_world_to_camera(
                torch.Tensor(obj_rotmat).cuda(),
                torch.Tensor(obj_trans).cuda(),
                extrinsic_matrix_sequence,
            )

            # object mesh
            obj_verts = object_model['verts'] # [N, 3]
            obj_faces = object_model['faces']  # [N, 3]
            obj_verts_hom = np.concatenate([obj_verts, np.ones([obj_verts.shape[0], 1])], axis=1)  # [N, 4]
            obj_verts_world_hom = np.einsum('tij,nj->tni', obj_pose_world, obj_verts_hom)  # [T, N, 4]
            obj_verts_world = obj_verts_world_hom[:, :, :3]  # shape: [T, N, 3]
            obj_verts_cam = np.einsum('ij,tpj->tpi', self.cam_extr, obj_verts_world_hom)[:, :, :3]  # [T, N, 3]

            object_dict = {0:dict(
                verts=dict(
                    world=obj_verts_world,  # (T, M, 3)
                    cam=obj_verts_cam  # (T, M, 3)
                ),
                faces=obj_faces,  # (N, 3)
                pose_seqs=dict(
                    world=dict(
                        rot=obj_rotmat,  # (T, 3, 3)
                        trans=obj_trans,  # (T, 3)
                    ),
                    cam=dict(
                        rot=obj_rotmat_cam.cpu().numpy(),  # (T, 3, 3)
                        trans=obj_trans_cam.cpu().numpy(),  # (T, 3)
                    ),
                ),
                masks=None,  # (T, 1), whether visible object
            )}

        # save rendered RGBD and record the file paths
        extrinsic_matrix_sequence = extrinsic_matrix_sequence.cpu().numpy() 
        if self.is_render_depth:
            render_depth_filenames = [os.path.splitext(i)[0].replace('color','depth')+'.png' for i in real_rgb_filenames]
            vis = VisOpen3D(*Image.open(real_rgb_filenames[0]).size, visible=False)
            for i, rdf in tqdm(enumerate(render_depth_filenames), desc=f"Rendering depth images for {task_name}", total=len(render_depth_filenames)):
                vis.save_rgbd_from_3d(
                    meshes_list=[
                        dict(verts=object_dict[0]['verts']['world'][i], faces=object_dict[0]['faces']),
                        *[dict(verts=hand_dict[side]['mesh']['verts']['world'][i], faces=hand_dict[side]['mesh']['faces']) for side in ['left', 'right'] if hand_dict[side] is not None],
                    ],
                    intrinsic=intrinsic_matrix, extrinsic=extrinsic_matrix_sequence[i],
                    save_path=dict(D=rdf, RGB=None)
                )
            vis.destroy_window()  
        else:
            render_depth_filenames = None

        # check length for each input items
        self.check_length(
            real_rgb_filenames,
            real_depth_filenames,
            hand_dict['right']['theta'], hand_dict['right']['pose_seqs']['cam']['rot'], hand_dict['right']['pose_seqs']['cam']['trans'],
            *((object_dict[0]["pose_seqs"]['world']['rot'], object_dict[0]["pose_seqs"]['world']['trans']) if object_model is not None else ()),
            extrinsic_matrix_sequence
        )

        return dict(
            # camera
            intrinsic_matrix=dict(ego=intrinsic_matrix, allo=None),
            extrinsic_matrix_sequence=dict(ego=extrinsic_matrix_sequence, allo=None),
            # real images
            real_image_filenames=dict(ego_rgb=real_rgb_filenames, ego_depth=real_depth_filenames, ego_seg=None, allo_rgb=None, allo_depth=None, allo_seg=None),
            # hand-object mesh projection rendered images
            render_image_filenames=dict(ego_rgb=None, ego_depth=render_depth_filenames),
            # hand
            hand_dict=hand_dict,
            # object
            object_dict=object_dict,
            # length
            length=frame_length,
            # text
            task_name=task_name,
            instruction=action_name.replace('_', ' ').capitalize(),
        )



if __name__ == "__main__":
    fpha_dataset = FPHADataset(is_render_depth=False)
    fpha_dataset.initialize_data_storage(FPHAStorage, dict(window_size=30, side='right'))
    print(len(fpha_dataset))
    batch = fpha_dataset[0]  # batch_size=1
    breakpoint()