import os, pickle, cv2, trimesh, torch, json
import numpy as np
import matplotlib.pyplot as plt
import hashlib
from PIL import Image, ImageDraw, ImageFont
from bisect import bisect_left, bisect_right
from tqdm import tqdm
from PIL import Image

from manopth.manotorch.manolayer import ManoLayer
from oakink2_toolkit.dataset import OakInk2__Dataset

from dataset_loader.storage import Storage, MetaDataset
from utils.util import transform_points
from utils.pc import get_pc_from_world_to_camera, get_pc_from_camera_to_world, VisOpen3D
from utils.rot import batch_rodrigues, quaternion_to_axis_angle, get_pose12d_from_world_to_camera, get_pose12d_from_camera_to_world
# mandatory
class OAKINK2Storage(Storage):
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

# mandatory    
class OAKINK2Dataset(MetaDataset):
    def __init__(self, 
        root_path='/media/ps/zhan_4T/OakInk-v2',
        mano_path = "manopth/mano/models",
    ):
        super().__init__(root_path)
        self.oakink2_dataset = OakInk2__Dataset(dataset_prefix=root_path,return_instantiated=True,)
        # path
        self.mano_path = mano_path

        # MANO
        self.left_mano_layer = ManoLayer(mano_assets_root=os.path.join(mano_path, '..'),rot_mode="quat",side="left",center_idx=0,use_pca=False,flat_hand_mean=True,).cuda()
        self.right_mano_layer = ManoLayer(mano_assets_root=os.path.join(mano_path, '..'),rot_mode="quat",side="right",center_idx=0,use_pca=False,flat_hand_mean=True,).cuda()

        # all object mesh: Trimesh
        self.object_meshes = {}
        
        # load task
        self.task_list = self.oakink2_dataset.__dict__["all_seq_list"]  # 627
        self.task_list = ['scene_01__A001/seq__0b3d183d6b2e90eefd66__2023-04-27-20-38-16']
        
    def load_task(
        self, 
        task_name='scene_01__A001/seq__0b3d183d6b2e90eefd66__2023-04-27-20-38-16',
        cam_name='egocentric',
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    ):
        complex_task_data = self.oakink2_dataset.load_complex_task(task_name)
        primitive_task_data_list = self.oakink2_dataset.load_primitive_task(complex_task_data)
        if not isinstance(primitive_task_data_list, list):
            primitive_task_data_list = [primitive_task_data_list]
        
        
        task_data_list = []
        for i_ptask, ptask_data in enumerate(primitive_task_data_list):
            # load camera intrinsic
            anno_filepath = os.path.join(self.oakink2_dataset.anno_prefix, f"{complex_task_data.seq_token}.pkl")
            with open(anno_filepath, "rb") as ifs:
                anno_info = pickle.load(ifs)
            cam_intr = next(iter(anno_info["cam_intr"][cam_name].values()))  # (3, 3)
            extrinsic_series = anno_info["cam_extr"][cam_name]  # (T, 4, 4)
            frame_id_list = anno_info["frame_id_list"]
            cam_view_id = {v: k for k, v in anno_info["cam_def"].items()}
            
            # object mesh
            for scene_object_id in complex_task_data.scene_obj_list:
                if scene_object_id not in self.object_meshes:
                    breakpoint()
                    affordance_data = self.oakink2_dataset.load_affordance(scene_object_id)
                    self.object_meshes[scene_object_id] = dict(vertices=affordance_data.obj_mesh.vertices, faces=affordance_data.obj_mesh.faces)
        
            primitive_task_description = ptask_data.task_desc   # 'Press the button on the scale.'

            # get timesteps and batch process
            fids = np.array(frame_id_list[bisect_left(frame_id_list, ptask_data.frame_range[0]):bisect_right(frame_id_list, ptask_data.frame_range[1]-1)])  # (T)
            
            # camera extrinsic
            extrinsic_matrix_sequence = np.stack([extrinsic_series[fid] for fid in fids], axis=0)  # (T, 4, 4)
            extrinsic_matrix_sequence = torch.FloatTensor(extrinsic_matrix_sequence).cuda()
            
            # real egocentric RGB images
            real_rgb_filenames = [os.path.join(self.oakink2_dataset.data_prefix, complex_task_data.seq_token, cam_view_id[cam_name], f"{fid:0>6}.png") for fid in fids]
            
            # hands
            hand_dict = dict()
            if ptask_data.frame_range_lh is not None:
                lh_pose_info = {k: v[fids-ptask_data['frame_range_lh'][0]].to(dtype=torch.float32, device=device) for k, v in ptask_data.lh_param.items()}
                lh_out = mano_forward(self.left_mano_layer, lh_pose_info)  # (T, 778, 3)
                left_pose_coeffs = quaternion_to_axis_angle(lh_pose_info['pose_coeffs'].cuda())  # [w,x,y,z] -> axis-angle, (T, 16, 3)
                left_rot, left_theta = batch_rodrigues(left_pose_coeffs[:, 0]), left_pose_coeffs[:, 1:]  # (T, 3, 3), (T, 15, 3)
                left_trans = lh_pose_info['tsl']  # (T, 3)
                left_beta = lh_pose_info['betas'][0]  # (10)
                left_rot_cam, left_trans_cam = get_pose12d_from_world_to_camera(
                    left_rot, 
                    left_trans,
                    extrinsic_matrix_sequence,
                )
                hand_dict['left'] = dict(
                    theta=left_theta.cpu().numpy(),  # (T, 45)
                    beta=left_beta.cpu().numpy(),    # (10,)
                    pose_seqs=dict(
                        world=dict(
                            rot=left_rot.cpu().numpy(),     # (T, 3, 3)
                            trans=left_trans.cpu().numpy(), # (T, 3)
                        ),
                        cam=dict(
                            rot=left_rot_cam.cpu().numpy(),  # (T, 3, 3)
                            trans=left_trans_cam.cpu().numpy(),# (T, 3)
                        ),
                    ),
                    mesh=dict(
                        verts=dict(
                            world=lh_out['v'],  # (T, 778, 3)
                            cam=get_pc_from_world_to_camera(torch.Tensor(lh_out['v']).to(device), extrinsic_matrix_sequence).cpu().numpy(),  # (T, 778, 3)
                        ),
                        faces=np.load(os.path.join(self.mano_path, f"left_faces.npy")),     # (1538, 3)
                        keypoints=dict(
                            world=lh_out['j'],  # (T, 21, 3)
                        )
                    ),
                    masks=None,  # (T, 1), whether visible hand
                )
            else:
                hand_dict['left'] = None
    
            if ptask_data.frame_range_rh is not None:
                rh_pose_info = {k: v[fids-ptask_data['frame_range_rh'][0]].to(dtype=torch.float32, device=device) for k, v in ptask_data.rh_param.items()} 
                rh_out = mano_forward(self.right_mano_layer, rh_pose_info)  # (T, 778, 3)
                right_pose_coeffs = quaternion_to_axis_angle(rh_pose_info['pose_coeffs'].cuda())  # [w,x,y,z] -> axis-angle, (T, 16, 3)
                right_rot, right_theta = batch_rodrigues(right_pose_coeffs[:, 0]), right_pose_coeffs[:, 1:]  # (T, 3, 3), (T, 15, 3)
                right_trans = rh_pose_info['tsl']  # (T, 3)
                right_beta = rh_pose_info['betas'][0]  # (10)
                right_rot_cam, right_trans_cam = get_pose12d_from_world_to_camera(
                    right_rot, 
                    right_trans,
                    extrinsic_matrix_sequence,
                )
                hand_dict['right'] = dict(
                    theta=right_theta.cpu().numpy(),  # (T, 45)
                    beta=right_beta.cpu().numpy(),    # (10,)
                    pose_seqs=dict(
                        world=dict(
                            rot=right_rot.cpu().numpy(),     # (T, 3, 3)
                            trans=right_trans.cpu().numpy(), # (T, 3)
                        ),
                        cam=dict(
                            rot=right_rot_cam.cpu().numpy(),  # (T, 3, 3)
                            trans=right_trans_cam.cpu().numpy(),# (T, 3)
                        ),
                    ),
                    mesh=dict(
                        verts=dict(
                            world=rh_out['v'],  # (T, 778, 3)
                            cam=get_pc_from_world_to_camera(torch.Tensor(rh_out['v']).to(device), extrinsic_matrix_sequence).cpu().numpy(),  # (T, 778, 3)
                        ),
                        faces=np.load(os.path.join(self.mano_path, f"right_faces.npy")),     # (1538, 3)
                        keypoints=dict(
                            world=rh_out['j'],  # (T, 21, 3)
                        )
                    ),
                    masks=None,  # (T, 1), whether visible hand
                )
            else:
                hand_dict['right'] = None
            
            # object 
            extrinsic_matrix_sequence = extrinsic_matrix_sequence.cpu().numpy() 
            object_dict = dict()
            for obj_id in ptask_data.task_obj_list:
                world_object_transforms = ptask_data.obj_transf[obj_id][fids - ptask_data.frame_range[0]]
                cam_object_transforms = extrinsic_matrix_sequence @ world_object_transforms
                cam_object_verts = self.object_meshes[obj_id]['vertices'] @ cam_object_transforms[:, :3, :3].transpose(0,2,1) + cam_object_transforms[:, :3, 3][:,None,:]
                world_object_verts = get_pc_from_camera_to_world(torch.Tensor(cam_object_verts).to(device), torch.Tensor(extrinsic_matrix_sequence).to(device)).cpu().numpy()
                object_dict[obj_id] = dict(
                    verts=dict(
                        world=world_object_verts,  # (T, M, 3)
                        cam=cam_object_verts       # (T, M, 3)
                    ),   
                    faces=self.object_meshes[obj_id]['faces'],
                    pose_seqs=dict(
                        world=dict(
                            rot=world_object_transforms[:, :3, :3],  # (T, 3, 3)
                            trans=world_object_transforms[:, :3, 3],  # (T, 3)
                        ),
                        cam=dict(
                            rot=cam_object_transforms[:, :3, :3],  # (T, 3, 3)
                            trans=cam_object_transforms[:, :3, 3],  # (T, 3)
                        ),
                    ),
                    masks=None,  # (T, 1), whether visible object
                )
                
            # save rendered RGBD and record the file paths
            render_depth_filenames = [i.replace(os.path.basename(os.path.dirname(i)), 'ego_depth') for i in real_rgb_filenames]
            vis = VisOpen3D(*Image.open(real_rgb_filenames[0]).size, visible=False)
            for t, rdf in tqdm(enumerate(render_depth_filenames), desc=f"Rendering depth images for {task_name}", total=len(render_depth_filenames)):
                if not os.path.exists(rdf):
                    vis.save_rgbd_from_3d(
                        meshes_list=[
                            *[dict(verts=obj_mesh['verts']['world'][t], faces=obj_mesh['faces']) for obj_mesh in object_dict.values()],
                            *[dict(verts=hand_dict[side]['mesh']['verts']['world'][t], faces=hand_dict[side]['mesh']['faces']) for side in ['left', 'right'] if hand_dict.get(side, None) is not None],
                        ],
                        intrinsic=cam_intr, extrinsic=extrinsic_matrix_sequence[t],
                        save_path=dict(D=rdf, RGB=None)
                    )
            vis.destroy_window()    
                
            # check length for each input items
            hand_ls = [hand_dict[side]['pose_seqs']['cam']['rot'] for side in ['left', 'right'] if hand_dict.get(side, None) is not None]
            obj_ls = [object_dict[obj_id]['pose_seqs']['cam']['rot'] for obj_id in ptask_data.task_obj_list]
            num_frames = self.check_length(
                real_rgb_filenames, 
                render_depth_filenames,
                *hand_ls,
                *obj_ls,
                extrinsic_matrix_sequence
            )    
                
            task_data_list.append(
                dict( 
                    # camera: for debug
                    intrinsic_matrix=dict(ego=cam_intr, allo=None), extrinsic_matrix_sequence=dict(ego=extrinsic_matrix_sequence, allo=None),
                    # real images
                    real_image_filenames=dict(ego_rgb=real_rgb_filenames, ego_depth=None, ego_seg=None, allo_rgb=None, allo_depth=None, allo_seg=None),  
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
                    instruction=primitive_task_description,
                )
            )
        return task_data_list
        

def caption_combined_view(combine_image, caption="This is a caption", font="FreeMonoBold.ttf"):
    # create a thin white canvas
    canvas_pil = Image.fromarray(np.ones((30, combine_image.shape[1], 3), dtype=np.uint8) * 255)
    draw = ImageDraw.Draw(canvas_pil)
    # draw text on the canvas
    draw.text((20, 5), caption, font=ImageFont.truetype(font, size=20), fill=(0, 0, 0))
    # put the caption on top of the image
    return np.concatenate([np.array(canvas_pil), combine_image], axis=0)

def slice_param(param, batch_size=512):
    key_list = list(param.keys())

    total_sample = param[key_list[0]].shape[0]
    num_batch = (total_sample + batch_size - 1) // batch_size

    res = []
    for batch_id in range(num_batch):
        store = {}
        start = batch_id * batch_size
        end = min((batch_id + 1) * batch_size, total_sample)
        for k in key_list:
            store[k] = param[k][start:end]
        res.append(store)
    return res

def mano_forward(mano_layer, hand_pose_param):
    '''
    hand_pose_param['pose_coeffs']: (T, 16, 4)
    hand_pose_param['betas']: (T, 10)
    hand_pose_param['tsl']: (T, 3)
    '''
    j_list, v_list = [], []
    for hand_pose_item in slice_param(hand_pose_param, batch_size=8192):  # to reduce GPU burden
        mano_out_sl = mano_layer(pose_coeffs=hand_pose_item["pose_coeffs"], betas=hand_pose_item["betas"])
        j_sl = mano_out_sl.joints + hand_pose_item["tsl"].unsqueeze(1)
        v_sl = mano_out_sl.verts + hand_pose_item["tsl"].unsqueeze(1)
        j = j_sl.clone().cpu().numpy()
        v = v_sl.clone().cpu().numpy()
        # j = transform_points(camera_extrinsic, j)
        # v = transform_points(camera_extrinsic, v)
        j_list.append(j)
        v_list.append(v)
    return dict(
        j=np.concatenate(j_list, axis=0),  # keypoints, not used
        v=np.concatenate(v_list, axis=0),
    )

def load_data_for_oakink2_and_visualize():
    dataset_root='/home/zbh/Desktop/zbh/hand-motion-pretraining/OakInk2/data'
    mano_path = 'manopth/mano'
    seq_key = 'scene_01__A001/seq__0b3d183d6b2e90eefd66__2023-04-27-20-38-16'
    cam_name = 'egocentric' # "allocentric_top"
    image_size = (848, 480)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enable_background = True

    mano_layer_rh = ManoLayer(
        mano_assets_root=mano_path,
        rot_mode="quat",
        side="right",
        center_idx=0,
        use_pca=False,
        flat_hand_mean=True,
    ).to(device)

    hand_faces_rh = mano_layer_rh.get_mano_closed_faces().cpu().numpy()
    mano_layer_lh = ManoLayer(
        mano_assets_root=mano_path,
        rot_mode="quat",
        side="left",
        center_idx=0,
        use_pca=False,
        flat_hand_mean=True,
    ).to(device)

    hand_faces_lh = np.concatenate([mano_layer_lh.th_faces.detach().cpu().numpy(), np.array([
        [92, 38, 122],
        [234, 92, 122],
        [239, 234, 122],
        [279, 239, 122],
        [215, 279, 122],
        [215, 122, 118],
        [215, 118, 117],
        [215, 117, 119],
        [215, 119, 120],
        [215, 120, 108],
        [215, 108, 79],
        [215, 79, 78],
        [215, 78, 121],
        [214, 215, 121],
    ],dtype=np.int64)[:, [2, 1, 0]]])

    oakink2_dataset = OakInk2__Dataset(dataset_prefix=dataset_root,return_instantiated=True,)
    complex_task_data = oakink2_dataset.load_complex_task(seq_key)
    primitive_task_data_list = oakink2_dataset.load_primitive_task(complex_task_data)

    if not isinstance(primitive_task_data_list, list):
        primitive_task_data_list = [primitive_task_data_list]
    # load all objects in the scene
    cm = plt.get_cmap("turbo")
    hash_str = lambda attr_str, trunc=20: hashlib.sha256(attr_str.encode()).hexdigest()[:trunc]
    obj_map = {}
    for obj_part_id in complex_task_data.scene_obj_list:
        affordance_data = oakink2_dataset.load_affordance(obj_part_id)
        _v = affordance_data.obj_mesh.vertices
        _f = affordance_data.obj_mesh.faces
        _vc = ((np.array(cm(int(hash_str(obj_part_id), 16) % 256))[0:3] ** 1.2) * 0.8 + 0.2).reshape((1, 3)).repeat(_v.shape[0], axis=0)  # Good! set color!
        obj_map[obj_part_id] = trimesh.Trimesh(vertices=_v, faces=_f, vertex_colors=_vc, process=False)
    '''
    cam_object_dict_list = []
    for obj_part_id in complex_task_data.scene_obj_list:
        affordance_data = oakink2_dataset.load_affordance(obj_part_id)
        cam_object_dict_list.append(dict(object_faces=affordance_data.obj_mesh.faces, cam_object_verts=affordance_data.obj_mesh.vertices))
    '''

    # load cam_extr and cam_intr
    anno_filepath = os.path.join(oakink2_dataset.anno_prefix, f"{complex_task_data.seq_token}.pkl")
    with open(anno_filepath, "rb") as ifs:
        anno_info = pickle.load(ifs)
    cam_intr = next(iter(anno_info["cam_intr"][cam_name].values()))
    cam_extr = next(iter(anno_info["cam_extr"][cam_name].values()))
    frame_id_list = anno_info["frame_id_list"]
    cam_view_id = {v: k for k, v in anno_info["cam_def"].items()}
    
    # create renderer
    renderer = PyMultiObjRenderer(width=image_size[0], height=image_size[1], obj_map=obj_map, cam_intr=cam_intr, raymond=True)
    for i_ptask, ptask_data in enumerate(primitive_task_data_list):
        # prepare mano hand vertices sequence
        if ptask_data.frame_range_lh is not None:
            lh_pose_info = {k: v[ptask_data.lh_in_range_mask].to(dtype=torch.float32, device=device) for k, v in ptask_data.lh_param.items()}
            lh_out = mano_forward(mano_layer_lh, lh_pose_info, cam_extr)
        if ptask_data.frame_range_rh is not None:
            rh_pose_info = {k: v[ptask_data.rh_in_range_mask].to(dtype=torch.float32, device=device) for k, v in ptask_data.rh_param.items()} 
            rh_out = mano_forward(mano_layer_rh, rh_pose_info, cam_extr)
        # visualize
        subsample_rate = 20
        for _f in range(*ptask_data.frame_range, subsample_rate):
            fid = frame_id_list[np.searchsorted(frame_id_list, _f, side='left')] if enable_background else _f # the first element >= _f
            extra_mesh = []
            if enable_background: # use image cam_extr rather than constant for background fusing
                cam_extr = anno_info["cam_extr"][cam_name][fid]
            # obj_pose = world2cam @ object_transformation
            offset = fid - ptask_data.frame_range[0]
            obj_pose_map = {obj_id: (cam_extr @ ptask_data.obj_transf[obj_id][offset] if obj_id in ptask_data.task_obj_list else None) for obj_id in ptask_data.scene_obj_list}

            if ptask_data.frame_range_lh is not None:
                offset_lh = fid - ptask_data.frame_range_lh[0]
                if 0 <= offset_lh and offset_lh < lh_out["v"].shape[0]:
                    v_lh = lh_out["v"][offset_lh]
                    extra_mesh.append(trimesh.Trimesh(vertices=v_lh, faces=hand_faces_lh))
            if ptask_data.frame_range_rh is not None:
                offset_rh = fid - ptask_data.frame_range_rh[0]
                if 0 <= offset_rh and offset_rh < rh_out["v"].shape[0]:
                    v_rh = rh_out["v"][offset_rh]
                    extra_mesh.append(trimesh.Trimesh(vertices=v_rh, faces=hand_faces_rh))

            if enable_background:
                rgb_file = os.path.join(oakink2_dataset.data_prefix, complex_task_data.seq_token, cam_view_id[cam_name], f"{fid:0>6}.png")
                bg_param = dict(background=cv2.imread(rgb_file))
            else:
                bg_param = dict(background=np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 255,)
            img = renderer(obj_pose_map=obj_pose_map, extra_mesh=extra_mesh, stick=True, **bg_param)
            img = caption_combined_view(img, ptask_data.task_desc)

            while True:
                cv2.imshow("x", img)
                key = cv2.waitKey(1)
                if key == ord("\r"):
                    break


if __name__ == "__main__":
    oakink2_dataset = OAKINK2Dataset()
    oakink2_dataset.initialize_data_storage(OAKINK2Storage, dict(window_size=30, side='right'))
    print(len(oakink2_dataset))
    batch = oakink2_dataset[0]  # batch_size=1
    breakpoint()
    