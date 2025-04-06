import os, pickle, cv2, torch, trimesh
import numpy as np
from os.path import join, basename, dirname
from natsort import natsorted
from glob import glob
from tqdm import tqdm
from PIL import Image

from dataset_loader.storage import Storage, MetaDataset
from utils.util import mp42imagefiles, mano_forward
from utils.pc import get_pc_from_world_to_camera, VisOpen3D
from utils.rot import batch_rodrigues, get_pose12d_from_world_to_camera


# mandatory
class TACOStorage(Storage):
    def __init__(self, **config):
        super().__init__(**config)
    
    def fill_data(self, task_data):
        if task_data["hand_dict"][self.side] is not None:
            return dict(
                text=task_data["instruction"],
                ego_rgb_files=task_data["real_image_filenames"]['ego_rgb'],                 # TODO: can be real_image_filenames / render_image_filenames
                ego_dep_files=task_data["real_image_filenames"]['ego_depth'],               # TODO: can be real_image_filenames / render_image_filenames
                theta_seqs=task_data["hand_dict"][self.side]['theta'],                      # (T, 15*3)
                beta_seqs=task_data["hand_dict"][self.side]['beta'].reshape(-1,10),         # (1, 10)
                rot_seqs=task_data["hand_dict"][self.side]['pose_seqs']['cam']['rot'],      # (T, 3, 3)
                trans_seqs=task_data["hand_dict"][self.side]['pose_seqs']['cam']['trans'],  # (T, 3)
                len=task_data["length"],
            )

# mandatory    
class TACODataset(MetaDataset):
    def __init__(self, 
        root_path='/home/zbh/Desktop/zbh/robot-rl/TACO-Instructions/dataset/preview/',
        object_model_path='/home/zbh/Desktop/zbh/robot-rl/BiDexHD/assets/TACOobjects/',
        is_render_depth=False
    ):
        super().__init__(root_path)
        # path
        self.hand_pose_path = join(root_path, "Hand_Poses/")
        self.object_pose_path = join(root_path, "Object_Poses/")
        self.real_ego_video_path = join(root_path, "Egocentric_RGB_Videos/")
        self.real_ego_frame_path = join(root_path, "Egocentric_RGB_Frames/")
        self.real_depth_frame_path = join(root_path, "Egocentric_Depth_Videos/")
        self.ego_param_path = join(root_path, "Egocentric_Camera_Parameters/")
        
        # MANO model
        '''
        from smplx import MANO
        self.left_hand_model = MANO(self.mano_path,create_transl=True,use_pca=False,flat_hand_mean=False,is_rhand=False).cuda()
        self.right_hand_model = MANO(self.mano_path,create_transl=True,use_pca=False,flat_hand_mean=False,is_rhand=True).cuda()
        from manotorch.manolayer import ManoLayer as ManoTorchLayer
        self.left_hand_torch_model = ManoTorchLayer(mano_assets_root=os.path.join(mano_path, '..'),use_pca=False,flat_hand_mean=True,rot_mode="axisang",side="left",center_idx=0,).cuda()
        self.right_hand_torch_model = ManoTorchLayer(mano_assets_root=os.path.join(mano_path, '..'),use_pca=False,flat_hand_mean=True,rot_mode="axisang",side="right",center_idx=0).cuda()
        '''
        
        # all object mesh: (verts, faces)
        self.object_model_files = natsorted(glob(join(object_model_path, "*_cm.obj")))
        self.object_meshes = {}
        for of in tqdm(self.object_model_files[:80], desc="Loading object meshs..."):
            obj_id = basename(of)[:-7]  # remove "_cm.obj"
            obj_mesh = trimesh.load(of)
            self.object_meshes[obj_id] = dict(verts=obj_mesh.vertices*0.01, faces=obj_mesh.faces)
        
        # load task
        self.task_list = [each[len(self.hand_pose_path):] for each in glob(join(self.hand_pose_path, "*/*"))]  # triplet/sequence_name
        self.task_list = ['(smear, glue gun, plate)/20231010_194']
        self.is_render_depth=is_render_depth
        
    def load_task(self, task_name='(smear, glue gun, plate)/20231010_194'):
        triplet, seq_name = task_name.split("/")
        verb, tool, target = triplet[1:-1].split(", ")
        instruction = f'{verb.capitalize()} the {target} with the {tool}.'
        if ' ' in verb:
            verb = ''.join(verb.split(' '))
        if ' ' in tool:
            tool = ''.join(tool.split(' '))
        if ' ' in target:
            target = ''.join(target.split(' '))
        clean_triplet = f"{verb},{tool},{target}"

        # real egocentric RGB images
        rgb_video_file = join(self.real_ego_video_path, triplet, seq_name, "color.mp4")
        save_real_frames_dir = join(self.real_ego_frame_path, clean_triplet, seq_name)
        mp42imagefiles(rgb_video_file, save_real_frames_dir)  
        real_rgb_filenames = natsorted(glob(join(save_real_frames_dir, "*.jpg")))
        
        # camera
        intrinsic_matrix = np.loadtxt(join(self.ego_param_path, triplet, seq_name, "egocentric_intrinsic.txt"))  # (3, 3)
        extrinsic_matrix_sequence = np.load(join(self.ego_param_path, triplet, seq_name, "egocentric_frame_extrinsic.npy"))[:,:3]  # (T, 3, 4)
        extrinsic_matrix_sequence = torch.FloatTensor(extrinsic_matrix_sequence).cuda()
        # hand
        hand_dict = dict()
        hand_pose_dir = join(self.hand_pose_path, triplet, seq_name)
        for side in ['left', 'right']:
            hand_theta = pickle.load(open(join(hand_pose_dir, f"{side}_hand.pkl"), "rb"))
            timesteps = natsorted(list(hand_theta.keys()))
            hand_theta, hand_trans = torch.stack([hand_theta[t]["hand_pose"] for t in timesteps]).cuda(), torch.stack([hand_theta[t]["hand_trans"] for t in timesteps]).cuda()  # (T, 48), (T, 3)
            hand_theta45, hand_rot = hand_theta[:, 3:], hand_theta[:, :3]  # (T, 45), (T, 3)
            hand_beta = pickle.load(open(join(hand_pose_dir, f"{side}_hand_shape.pkl"), "rb"))["hand_shape"]   # (10,)
            hand_beta = torch.Tensor(hand_beta).cuda().unsqueeze(0).repeat(hand_theta45.shape[0], 1)
            hand_rot9d = batch_rodrigues(hand_rot)  # (T, 3, 3)
            hand_rot9d_cam, hand_trans_cam = get_pose12d_from_world_to_camera(hand_rot9d, hand_trans, extrinsic_matrix_sequence)  # (T, 3, 3), (T, 3)
            
            # MANO forward
            hand_vertices, hand_keypoints, hand_faces = mano_forward(hand_theta45, hand_beta, hand_rot, hand_trans, side)
            hand_vertices_cam = get_pc_from_world_to_camera(hand_vertices, extrinsic_matrix_sequence)  # (T, 778, 3)
            '''
            # ARCTIC
            hand_out = getattr(self, f"{side}_hand_model")(
                global_orient=hand_rot,
                hand_pose=hand_theta45,
                betas=hand_beta,
                transl=hand_trans
            )
            # OAKONK2
            hand_out_torch = getattr(self, f"{side}_hand_torch_model")(pose_coeffs=hand_theta, betas=hand_beta)
            hand_out_torch_joints = hand_out_torch.joints + hand_trans.unsqueeze(1) 
            hand_out_torch_verts = hand_out_torch.verts + hand_trans.unsqueeze(1)
            # visualizer
            vis = VisOpen3D()
            timestep=0
            vis.add_mesh(dict(verts=hand_vertices[timestep].detach().cpu().numpy(), faces=hand_faces.tolist()))
            vis.draw_points3D(hand_keypoints[timestep].cpu().numpy())
            vis.add_mesh(dict(verts=hand_out.vertices[timestep].detach().cpu().numpy(), faces=hand_faces.tolist()))
            vis.draw_points3D(hand_out.joints[timestep].detach().cpu().numpy(), color=(0.5,0,0.5))
            vis.add_mesh(dict(verts=hand_out_torch_verts[timestep].detach().cpu().numpy(), faces=hand_faces.tolist()))
            vis.draw_points3D(hand_out_torch_joints[timestep].detach().cpu().numpy(), color=(0,0.5,0.5))
            vis.run()
            vis.destroy_window()
            breakpoint()
            '''
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
        object_pose_dir = join(self.object_pose_path, triplet, seq_name)
        for obj in ["tool", "target"]:
            obj_filename = glob(join(object_pose_dir, f"{obj}_*"))[0]
            obj_poses = np.load(obj_filename)
            obj_id = basename(obj_filename).split(".")[0].split("_")[-1]
            obj_rotmat = obj_poses[:, :3, :3]
            obj_trans = obj_poses[:, :3, 3]
            obj_rotmat_cam, obj_trans_cam = get_pose12d_from_world_to_camera(
                torch.Tensor(obj_rotmat).cuda(), 
                torch.Tensor(obj_trans).cuda(),
                extrinsic_matrix_sequence,
            )
            '''
            np.allclose(self.object_meshes[obj_id]['verts'] @ obj_rotmat_cam.cpu().numpy().transpose(0,2,1) + obj_trans_cam.cpu().numpy()[:,None,:], cam_object_verts.cpu().numpy(), atol=1e-5)
            '''
            world_object_verts = self.object_meshes[obj_id]['verts'] @ obj_rotmat.transpose(0,2,1) + obj_trans[:,None,:]  # (T, M, 3)
            cam_object_verts = get_pc_from_world_to_camera(torch.Tensor(world_object_verts).cuda(), extrinsic_matrix_sequence)  # (T, M, 3)

            object_dict[obj] = dict(
                verts=dict(
                    world=world_object_verts,  # (T, M, 3)
                    cam=cam_object_verts.cpu().numpy()  # (T, M, 3)
                ),   
                faces=self.object_meshes[obj_id]['faces'],
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
            )
            
        # save DEPTH images: optional!
        real_depth_filenames = glob(join(self.real_depth_frame_path, clean_triplet, seq_name, '.png'))

        # save rendered RGBD and record the file paths
        extrinsic_matrix_sequence = extrinsic_matrix_sequence.cpu().numpy() 
        if self.data_storage.is_render_depth:
            render_depth_filenames = [os.path.splitext(i)[0].replace('RGB','depth')+'.png' for i in real_rgb_filenames]
            vis = VisOpen3D(*Image.open(real_rgb_filenames[0]).size, visible=False)
            for i, rdf in tqdm(enumerate(render_depth_filenames), desc=f"Rendering depth images for {task_name}", total=len(render_depth_filenames)):
                vis.save_rgbd_from_3d(
                    meshes_list=[
                        *[dict(verts=object_dict[obj]['verts']['world'][i], faces=object_dict[obj]['faces']) for obj in ['tool', 'target']],
                        *[dict(verts=hand_dict[side]['mesh']['verts']['world'][i], faces=hand_dict[side]['mesh']['faces']) for side in ['left', 'right']],
                    ],
                    intrinsic=intrinsic_matrix, extrinsic=extrinsic_matrix_sequence[i],
                    save_path=dict(D=rdf, RGB=None)
                )
            vis.destroy_window()  
        else:
            render_depth_filenames = None
            
        # check length for each input items
        num_frames = self.check_length(
            real_rgb_filenames, 
            # real_depth_filenames,
            hand_dict['left']['theta'], hand_dict['left']['pose_seqs']['cam']['rot'], hand_dict['left']['pose_seqs']['cam']['trans'],
            hand_dict['right']['theta'], hand_dict['right']['pose_seqs']['cam']['rot'], hand_dict['right']['pose_seqs']['cam']['trans'],
            object_dict["tool"]["pose_seqs"]['world']['rot'], object_dict["tool"]["pose_seqs"]['world']['trans'],
            object_dict["target"]["pose_seqs"]['world']['rot'], object_dict["target"]["pose_seqs"]['world']['trans'],
            extrinsic_matrix_sequence
        )    
        
        return dict(
            # camera: for debug
            intrinsic_matrix=dict(ego=intrinsic_matrix, allo=None), extrinsic_matrix_sequence=dict(ego=extrinsic_matrix_sequence, allo=None),
            # real images
            real_image_filenames=dict(ego_rgb=real_rgb_filenames, ego_depth=real_depth_filenames, ego_seg=None, allo_rgb=None, allo_depth=None, allo_seg=None),  
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
        

class TACODatasetCollector(MetaDataset):
    def __init__(self, 
        root_path='/home/zbh/Desktop/zbh/robot-rl/TACO-Instructions/dataset/preview/',
    ):
        super().__init__(root_path)
        # path
        self.hand_pose_path = join(root_path, "Hand_Poses/")
        
        # collect from task
        self.mano_in_out_collector = dict(  # under param: ManoLayer(use_pca=False, flat_hand_mean=True, ncomps=45, side='right', center_idx=0)
            task_list=[],
            left=dict(
                keypoints=[],
                mano_params=dict(
                    theta=[],
                    beta=[],
                    rot=[],
                    trans=[],
                ),
                cum_length=[0],
            ),
            right=dict(
                keypoints=[],
                mano_params=dict(
                    theta=[],
                    beta=[],
                    rot=[],
                    trans=[],
                ),
                cum_length=[0]
            )
        )
        self.task_list = [each[len(self.hand_pose_path):] for each in glob(join(self.hand_pose_path, "*/*"))]  # triplet/sequence_name
        self.triplet_backup = set()
        pbar = tqdm(self.task_list, desc="Loading tasks...")
        for task_name in pbar:
            self.load_task(task_name)
            pbar.set_postfix_str(f"Loaded {task_name} | total={self.mano_in_out_collector['right']['cum_length'][-1]}")
        np.savez("inv_mano/mano_pair.npz", **self.mano_in_out_collector)
        breakpoint()
        
    def load_task(self, task_name):
        triplet, seq_name = task_name.split("/")
        # if triplet in self.triplet_backup:
        #     return
        # else:
        #     self.triplet_backup.add(triplet)
        
        # hand
        hand_pose_dir = join(self.hand_pose_path, triplet, seq_name)
        for side in ['left', 'right']:
            hand_theta = pickle.load(open(join(hand_pose_dir, f"{side}_hand.pkl"), "rb"))
            timesteps = natsorted(list(hand_theta.keys()))
            hand_theta, hand_trans = torch.stack([hand_theta[t]["hand_pose"] for t in timesteps]).cuda(), torch.stack([hand_theta[t]["hand_trans"] for t in timesteps]).cuda()  # (T, 48), (T, 3)
            hand_theta45, hand_rot = hand_theta[:, 3:], hand_theta[:, :3]  # (T, 45), (T, 3)
            hand_beta = pickle.load(open(join(hand_pose_dir, f"{side}_hand_shape.pkl"), "rb"))["hand_shape"]   # (10,)
            hand_beta = torch.Tensor(hand_beta).cuda().unsqueeze(0).repeat(hand_theta45.shape[0], 1)  # (T, 10)
            num_frames = hand_theta45.shape[0]
            
            # MANO forward
            hand_vertices, hand_keypoints, hand_faces = mano_forward(hand_theta45, hand_beta, hand_rot, hand_trans, side)
            self.mano_in_out_collector['task_list'].append(task_name)
            self.mano_in_out_collector[side]['cum_length'].append(self.mano_in_out_collector[side]["cum_length"][-1] + num_frames)
            self.mano_in_out_collector[side]['keypoints'] += hand_keypoints.tolist()  # (T, 21, 3)
            self.mano_in_out_collector[side]['mano_params']['theta'] += hand_theta45.tolist()  # (T, 45)
            self.mano_in_out_collector[side]['mano_params']['beta'] += hand_beta.tolist()  # (T, 10)
            self.mano_in_out_collector[side]['mano_params']['rot'] += hand_rot.tolist()  # (T, 3)
            self.mano_in_out_collector[side]['mano_params']['trans'] += hand_trans.tolist()  # (T, 3)
            
   
if __name__ == "__main__":
    # TACODatasetCollector()
    taco_dataset = TACODataset(is_render_depth=False)
    taco_dataset.initialize_data_storage(TACOStorage, dict(window_size=30, side='right'))
    print(len(taco_dataset))
    batch = taco_dataset[0]
    breakpoint()
    