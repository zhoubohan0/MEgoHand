import os, json
import numpy as np
import trimesh
from natsort import natsorted
from glob import glob
from tqdm import tqdm
from einops import repeat
from PIL import Image
import torch
from torch import nn
from smplx import MANO

from dataset_loader.storage import Storage, MetaDataset
from utils.pc import get_pc_from_world_to_camera, VisOpen3D
from utils.util import thing2dev, pad_tensor_list
from utils.rot import axis_angle_to_quaternion, quaternion_apply, batch_rodrigues, get_pose12d_from_world_to_camera




OBJECTS = [
    "capsulemachine",
    "box",
    "ketchup",
    "laptop",
    "microwave",
    "mixer",
    "notebook",
    "espressomachine",
    "waffleiron",
    "scissors",
    "phone",
]

def construct_obj(object_model_dir):
    from easydict import EasyDict
    from scipy.spatial.distance import cdist
    # load vtemplate
    mesh_p = os.path.join(object_model_dir, "mesh.obj")
    parts_p = os.path.join(object_model_dir, f"parts.json")
    json_p = os.path.join(object_model_dir, "object_params.json")
    
    top_sub_p = os.path.join(object_model_dir, "top_keypoints_300.json")
    bottom_sub_p = top_sub_p.replace("top_", "bottom_")
    with open(top_sub_p, "r") as f:
        sub_top = np.array(json.load(f)["keypoints"])

    with open(bottom_sub_p, "r") as f:
        sub_bottom = np.array(json.load(f)["keypoints"])
    sub_v = np.concatenate((sub_top, sub_bottom), axis=0)

    with open(parts_p, "r") as f:
        parts = np.array(json.load(f), dtype=bool)

    assert os.path.exists(mesh_p), f"Not found: {mesh_p}"

    mesh = trimesh.exchange.load.load_mesh(mesh_p, process=False)
    mesh_v = mesh.vertices

    mesh_f = torch.LongTensor(mesh.faces)
    vidx = np.argmin(cdist(sub_v, mesh_v, metric="euclidean"), axis=1)
    parts_sub = parts[vidx]

    vsk = object_model_dir.split("/")[-1]

    with open(json_p, "r") as f:
        params = json.load(f)
        rest = EasyDict()
        rest.top = np.array(params["mocap_top"])
        rest.bottom = np.array(params["mocap_bottom"])
        bbox_top = np.array(params["bbox_top"])
        bbox_bottom = np.array(params["bbox_bottom"])
        kp_top = np.array(params["keypoints_top"])
        kp_bottom = np.array(params["keypoints_bottom"])

    obj = EasyDict()
    obj.name = vsk
    obj.obj_name = "".join([i for i in vsk if not i.isdigit()])
    obj.v = torch.FloatTensor(mesh_v)
    obj.v_sub = torch.FloatTensor(sub_v)
    obj.f = torch.LongTensor(mesh_f)
    obj.parts = torch.LongTensor(parts)
    obj.parts_sub = torch.LongTensor(parts_sub)

    with open(os.path.join(object_model_dir, "../../object_meta.json"), "r") as f:
        object_meta = json.load(f)
    obj.diameter = torch.FloatTensor(np.array(object_meta[obj.obj_name]["diameter"]))
    obj.bbox_top = torch.FloatTensor(bbox_top)
    obj.bbox_bottom = torch.FloatTensor(bbox_bottom)
    obj.kp_top = torch.FloatTensor(kp_top)
    obj.kp_bottom = torch.FloatTensor(kp_bottom)
    obj.mocap_top = torch.FloatTensor(np.array(params["mocap_top"]))
    obj.mocap_bottom = torch.FloatTensor(np.array(params["mocap_bottom"]))
    return obj

def construct_obj_tensors(root_path, object_names):
    obj_list = [construct_obj(os.path.join(root_path, "data/arctic_data/data/meta/object_vtemplates/", object_name)) for object_name in object_names]

    bbox_top_list = []
    bbox_bottom_list = []
    mocap_top_list = []
    mocap_bottom_list = []
    kp_top_list = []
    kp_bottom_list = []
    v_list = []
    v_sub_list = []
    f_list = []
    parts_list = []
    parts_sub_list = []
    diameter_list = []
    for obj in obj_list:
        v_list.append(obj.v)
        v_sub_list.append(obj.v_sub)
        f_list.append(obj.f)

        # root_list.append(obj.root)
        bbox_top_list.append(obj.bbox_top)
        bbox_bottom_list.append(obj.bbox_bottom)
        kp_top_list.append(obj.kp_top)
        kp_bottom_list.append(obj.kp_bottom)
        mocap_top_list.append(obj.mocap_top / 1000)
        mocap_bottom_list.append(obj.mocap_bottom / 1000)
        parts_list.append(obj.parts + 1)
        parts_sub_list.append(obj.parts_sub + 1)
        diameter_list.append(obj.diameter)

    v_list, v_len_list = pad_tensor_list(v_list)
    p_list, p_len_list = pad_tensor_list(parts_list)
    ps_list = torch.stack(parts_sub_list, dim=0)
    assert (p_len_list - v_len_list).sum() == 0

    max_len = v_len_list.max()
    mask = torch.zeros(len(obj_list), max_len)
    for idx, vlen in enumerate(v_len_list):
        mask[idx, :vlen] = 1.0

    v_sub_list = torch.stack(v_sub_list, dim=0)
    diameter_list = torch.stack(diameter_list, dim=0)

    f_list, f_len_list = pad_tensor_list(f_list)

    bbox_top_list = torch.stack(bbox_top_list, dim=0)
    bbox_bottom_list = torch.stack(bbox_bottom_list, dim=0)
    kp_top_list = torch.stack(kp_top_list, dim=0)
    kp_bottom_list = torch.stack(kp_bottom_list, dim=0)

    obj_tensors = {}
    obj_tensors["names"] = object_names
    obj_tensors["parts_ids"] = p_list
    obj_tensors["parts_sub_ids"] = ps_list

    obj_tensors["v"] = v_list.float() / 1000
    obj_tensors["v_sub"] = v_sub_list.float() / 1000
    obj_tensors["v_len"] = v_len_list
    obj_tensors["f"] = f_list
    obj_tensors["f_len"] = f_len_list
    obj_tensors["diameter"] = diameter_list.float()

    obj_tensors["mask"] = mask
    obj_tensors["bbox_top"] = bbox_top_list.float() / 1000
    obj_tensors["bbox_bottom"] = bbox_bottom_list.float() / 1000
    obj_tensors["kp_top"] = kp_top_list.float() / 1000
    obj_tensors["kp_bottom"] = kp_bottom_list.float() / 1000
    obj_tensors["mocap_top"] = mocap_top_list
    obj_tensors["mocap_bottom"] = mocap_bottom_list
    obj_tensors["z_axis"] = torch.FloatTensor(np.array([0, 0, -1])).view(1, 3)
    return obj_tensors

class ObjectTensors(nn.Module):
    def __init__(self, root_path):
        super(ObjectTensors, self).__init__()
        self.obj_tensors = thing2dev(construct_obj_tensors(root_path, OBJECTS), "cuda")
        self.dev = None

    def forward(self, angles, global_orient, transl, query_names):
        # _sanity_check: assume transl is in meter
        batch_size = angles.shape[0]
        assert angles.shape == (batch_size, 1)
        assert global_orient.shape == (batch_size, 3)
        if transl is not None:
            assert isinstance(transl, torch.Tensor)
            assert transl.shape == (batch_size, 3)
        assert len(query_names) == batch_size

        # store output
        out = dict()

        # meta info
        obj_idx = np.array([self.obj_tensors["names"].index(name) for name in query_names])
        out["diameter"] = self.obj_tensors["diameter"][obj_idx]
        out["f"] = self.obj_tensors["f"][obj_idx]
        out["f_len"] = self.obj_tensors["f_len"][obj_idx]
        out["v_len"] = self.obj_tensors["v_len"][obj_idx]

        max_len = out["v_len"].max()
        out["v"] = self.obj_tensors["v"][obj_idx][:, :max_len]
        out["mask"] = self.obj_tensors["mask"][obj_idx][:, :max_len]
        out["v_sub"] = self.obj_tensors["v_sub"][obj_idx]
        out["parts_ids"] = self.obj_tensors["parts_ids"][obj_idx][:, :max_len]
        out["parts_sub_ids"] = self.obj_tensors["parts_sub_ids"][obj_idx]

        # articulation + global rotation
        quat_arti = axis_angle_to_quaternion(self.obj_tensors["z_axis"] * angles)
        quat_global = axis_angle_to_quaternion(global_orient.view(-1, 3))

        # mm
        # collect entities to be transformed
        tf_dict = dict()
        tf_dict["v_top"] = out["v"].clone()
        tf_dict["v_sub_top"] = out["v_sub"].clone()
        tf_dict["v_bottom"] = out["v"].clone()
        tf_dict["v_sub_bottom"] = out["v_sub"].clone()
        tf_dict["bbox_top"] = self.obj_tensors["bbox_top"][obj_idx]
        tf_dict["bbox_bottom"] = self.obj_tensors["bbox_bottom"][obj_idx]
        tf_dict["kp_top"] = self.obj_tensors["kp_top"][obj_idx]
        tf_dict["kp_bottom"] = self.obj_tensors["kp_bottom"][obj_idx]

        # articulate top parts
        for key, val in tf_dict.items():
            if "top" in key:
                val_rot = quaternion_apply(quat_arti[:, None, :], val)
                tf_dict[key] = val_rot

        # global rotation for all
        for key, val in tf_dict.items():
            val_rot = quaternion_apply(quat_global[:, None, :], val)
            if transl is not None:
                val_rot = val_rot + transl[:, None, :]
            tf_dict[key] = val_rot

        # prep output
        top_idx = out["parts_ids"] == 1
        v_tensor = tf_dict["v_bottom"].clone()
        v_tensor[top_idx, :] = tf_dict["v_top"][top_idx, :]

        top_idx = out["parts_sub_ids"] == 1
        v_sub_tensor = tf_dict["v_sub_bottom"].clone()
        v_sub_tensor[top_idx, :] = tf_dict["v_sub_top"][top_idx, :]

        bbox = torch.cat((tf_dict["bbox_top"], tf_dict["bbox_bottom"]), dim=1)
        kp3d = torch.cat((tf_dict["kp_top"], tf_dict["kp_bottom"]), dim=1)

        out["v"] = v_tensor
        out["v_sub"] = v_sub_tensor
        out["bbox3d"] = bbox
        out["kp3d"] = kp3d
        return out


    def to(self, dev):
        self.obj_tensors = thing2dev(self.obj_tensors, dev)
        self.dev = dev

# mandatory
class ARCTICStorage(Storage):
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
class ARCTICDataset(MetaDataset):
    # uncomment /home/zbh/anaconda3/envs/foundationpose/lib/python3.8/site-packages/smplx/body_models.py L1681
    # joints = self.vertex_joint_selector(vertices, joints)   
    def __init__(self, 
        root_path='/home/zbh/Desktop/zbh/robot-rl/arctic/',
        mano_path = "manopth/mano/models",
    ):
        super().__init__(root_path)
        # preprocess data, then save all hand & object data to file, and remove processed_data
        '''
        task=${1:-"s01/box_grab_01"}
        out_dir=${2:-"data/arctic_data/data/raw_seqs/"}
        python scripts_data/process_seqs.py --export_verts --mano_p $out_dir${task}.mano.npy 
        '''
        
        # path
        self.raw_seq_path = os.path.join(root_path, "data/arctic_data/data/raw_seqs/")
        self.real_rgb_path = os.path.join(root_path, "data/arctic_data/data/images/")
        self.mano_path = mano_path
        
        # MANO model
        self.left_hand_model = MANO(self.mano_path,create_transl=True,use_pca=False,flat_hand_mean=False,is_rhand=False).cuda()
        self.right_hand_model = MANO(self.mano_path,create_transl=True,use_pca=False,flat_hand_mean=False,is_rhand=True).cuda()
        
        
        # all object information: at least contain (verts, faces)
        '''
        for obj_name in OBJECTS:
            obj_dir = os.path.join(self.root_path, "data/arctic_data/data/meta/object_vtemplates/", obj_name)
            obj_model_file = os.path.join(obj_dir, "mesh.obj")
            obj_part_file = os.path.join(obj_dir, "part.json")
            obj_faces = trimesh.load(obj_model_file).faces  # (N, 3)
        '''
        self.object_layer = ObjectTensors(self.root_path).cuda()

        # load task
        self.task_list = [each[len(self.raw_seq_path):].split('.')[0] for each in glob(os.path.join(self.raw_seq_path, "s*/*.mano.npy"))]
        self.task_list = ['s01/box_use_01', 's01/box_grab_01']

        
        
    def load_task(
        self, 
        task_name='s01/box_use_01',
        view_idx = 0  # Egocentric view
    ):
        subject, seq_name = task_name.split("/")
        obj_name, verb, _ = seq_name.split("_")
        
        # real egocentric RGB images
        img_dir = os.path.join(self.real_rgb_path, f"{subject}/{seq_name}/{view_idx}")
        real_rgb_filenames = natsorted(glob(os.path.join(img_dir, "*.jpg"))) + natsorted(glob(os.path.join(img_dir, "*.png")))
        
        # camera
        '''
        K = processed_data["params"]["K_ego"][0]
        extrinsic_matrix_sequence = processed_data["params"]['world2ego'][:,:3]
        '''
        egocam_file = os.path.join(self.raw_seq_path, f"{task_name}.egocam.dist.npy")
        egocam = np.load(egocam_file, allow_pickle=True)[()]
        K = np.array(egocam['intrinsics'])  # (3, 3)
        Rt = egocam['R_k_cam_np']  # (T, 3, 3)
        Tt = egocam['T_k_cam_np']  # (T, 3, 1) 
        extrinsic_matrix_sequence = np.concatenate([Rt, Tt], axis=-1)  # (T, 3, 4)
        extrinsic_matrix_sequence = torch.FloatTensor(extrinsic_matrix_sequence).cuda()
        
        # hand
        hand_file = os.path.join(self.raw_seq_path, f"{task_name}.mano.npy")
        hand_data = np.load(hand_file, allow_pickle=True).item()
        num_frames = len(hand_data["right"]["rot"])

        hand_dict = dict()
        for side in ['left', 'right']:
            # mano parameters -> vertices in world / camera frame
            hand_verts = getattr(self, f"{side}_hand_model")(
                global_orient=torch.FloatTensor(hand_data[side]["rot"]).cuda(),
                hand_pose=torch.FloatTensor(hand_data[side]["pose"]).cuda(),
                betas=torch.FloatTensor(hand_data[side]["shape"]).repeat(num_frames, 1).cuda(),
                transl=torch.FloatTensor(hand_data[side]["trans"]).cuda()
            ).vertices  # (T, 778, 3)
            hand_verts_cam = get_pc_from_world_to_camera(hand_verts, extrinsic_matrix_sequence)  # (T, 778, 3)
            '''
            cam_verts_r = get_pc_from_world_to_camera(world_verts_r, extrinsic_matrix_sequence)
            cam_verts_l = get_pc_from_world_to_camera(world_verts_l, extrinsic_matrix_sequence)
            
            np.allclose(cam_verts_r,processed_data["cam_coord"]["verts.right"][:,0],atol=1e-6)
            np.allclose(cam_verts_l,processed_data["cam_coord"]["verts.left"][:,0],atol=1e-6)        
            '''
            
            hand_data[side]["rot9d"] = batch_rodrigues(torch.FloatTensor(hand_data[side]["rot"]).cuda())  # (T, 3, 3)
            hand_data[side]["rot9d_cam"], hand_data[side]["trans_cam"] = get_pose12d_from_world_to_camera(
                hand_data[side]["rot9d"], 
                torch.FloatTensor(hand_data[side]["trans"]).cuda(),
                extrinsic_matrix_sequence,
            )
            
            hand_dict[side] = dict(
                theta=hand_data[side]["pose"],  # (T, 45)
                beta=hand_data[side]["shape"],  # (10)
                # beta=repeat(hand_data[side]["shape"], 'D -> T D', T=num_frames),  # (T, 10)
                pose_seqs=dict(
                    world=dict(
                        rot=hand_data[side]["rot9d"].cpu().numpy(),  # (T, 3, 3)
                        trans=hand_data[side]["trans"]  # (T, 3)
                    ),
                    cam=dict(
                        rot=hand_data[side]["rot9d_cam"].cpu().numpy(),  # (T, 3, 3)
                        trans=hand_data[side]["trans_cam"].cpu().numpy(),  # (T, 3)
                    ) 
                ),
                mesh=dict(
                    verts=dict(
                        world=hand_verts.cpu().numpy(),  # (T, 778, 3)
                        cam=hand_verts_cam.cpu().numpy()  # (T, 778, 3)
                    ),
                    faces=np.load(os.path.join(self.mano_path, f"{side}_faces.npy"))  # (N, 3)
                ),
                masks=None,  # (T, 1), whether visible hand
            )
        
        # object
        obj_param_file = os.path.join(self.raw_seq_path, f"{task_name}.object.npy")
        obj_param = torch.from_numpy(np.load(obj_param_file, allow_pickle=True)).cuda() 
        obj_arti, obj_axis_angle, obj_trans = obj_param[:, 0:1], obj_param[:, 1:4], obj_param[:, 4:] / 1000
        obj_out = self.object_layer.forward(
            angles=obj_arti,
            global_orient=obj_axis_angle,
            transl=obj_trans,
            query_names=[obj_name] * num_frames,
        )
        object_faces = obj_out['f'][0].cpu().numpy()  # (N, 3) 
        world_object_verts = obj_out['v']
        cam_object_verts = get_pc_from_world_to_camera(world_object_verts, extrinsic_matrix_sequence).cpu().numpy()
        world_object_verts = world_object_verts.cpu().numpy()
        '''
        np.allclose(world_object_verts, processed_data["world_coord"]["verts.object"], atol=1e-5)
        np.allclose(cam_object_verts, processed_data["cam_coord"]["verts.object"][:, view_idx], atol=1e-5)
        '''
        obj_rotmat = batch_rodrigues(obj_axis_angle)  # (T, 3, 3)
        obj_rotmat_cam, obj_trans_cam = get_pose12d_from_world_to_camera(
            obj_rotmat, 
            obj_trans,
            extrinsic_matrix_sequence,
        )
        object_dict = {0:dict(
            verts=dict(
                world=world_object_verts,  # (T, M, 3)
                cam=cam_object_verts  # (T, M, 3)
            ),          
            faces=object_faces, # (N, 3)
            pose_seqs=dict(
                world=dict(
                    rot=obj_rotmat,  # (T, 3, 3)
                    trans=obj_trans,  # (T, 3)
                ),
                cam=dict(
                    rot=obj_rotmat_cam,  # (T, 3, 3)
                    trans=obj_trans_cam,  # (T, 3)
                ),
            ),
            arti=obj_arti,  # (T, 1)
            masks=None,  # (T, 1), whether visible object
        )}
        
        # save rendered RGBD and record the file paths
        render_depth_filenames = [os.path.splitext(i)[0].replace('images','depths')+'.png' for i in real_rgb_filenames]
        ''''''
        # save DEPTH images
        extrinsic_matrix_sequence = extrinsic_matrix_sequence.cpu().numpy()
        vis = VisOpen3D(*Image.open(real_rgb_filenames[0]).size, visible=False)
        for i, rdf in tqdm(enumerate(render_depth_filenames), desc=f"Saving depth images for {task_name}", total=len(render_depth_filenames)):
            if not os.path.exists(rdf):
                vis.save_rgbd_from_3d(
                    meshes_list=[
                        dict(verts=world_object_verts[i],faces=object_faces),
                        *[dict(verts=hand_dict[side]['mesh']['verts']['world'][i], faces=hand_dict[side]['mesh']['faces']) for side in ['left', 'right']],
                    ],
                    intrinsic=K, extrinsic=extrinsic_matrix_sequence[i],
                    save_path=dict(D=rdf, RGB=None)
                )
        vis.destroy_window()  
        
        # check length for each input items
        self.check_length(
            real_rgb_filenames, 
            render_depth_filenames,
            hand_dict['left']['theta'], hand_dict['left']['pose_seqs']['cam']['rot'], hand_dict['left']['pose_seqs']['cam']['trans'],
            hand_dict['right']['theta'], hand_dict['right']['pose_seqs']['cam']['rot'], hand_dict['right']['pose_seqs']['cam']['trans'],
            object_dict[0]["pose_seqs"]['world']['rot'], object_dict[0]["pose_seqs"]['world']['trans'],
            extrinsic_matrix_sequence
        )    
        
        return dict(
            # camera
            intrinsic_matrix=dict(ego=K, allo=None), 
            extrinsic_matrix_sequence=dict(ego=extrinsic_matrix_sequence, allo=None),
            # real images
            real_image_filenames=dict(
                ego_rgb=real_rgb_filenames, ego_depth=None, ego_seg=None, 
                allo_rgb=None, allo_depth=None, allo_seg=None
            ),  
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
            instruction=f'{verb.capitalize()} the {obj_name}.',
        )
    


class ARCTICDatasetCollector(MetaDataset):
    def __init__(self, 
        root_path='/home/zbh/Desktop/zbh/robot-rl/arctic/',
        mano_path = "manopth/mano/models",
    ):
        super().__init__(root_path)
        # path
        self.raw_seq_path = os.path.join(root_path, "data/arctic_data/data/raw_seqs/")
        self.real_rgb_path = os.path.join(root_path, "data/arctic_data/data/images/")
        self.mano_path = mano_path
        
        # MANO model
        self.left_hand_model = MANO(self.mano_path,create_transl=True,use_pca=False,flat_hand_mean=False,is_rhand=False).cuda()
        self.right_hand_model = MANO(self.mano_path,create_transl=True,use_pca=False,flat_hand_mean=False,is_rhand=True).cuda()
        
        # collect from task
        self.mano_in_out_collector = dict(  # under param: MANO(use_pca=False, flat_hand_mean=False, ncomps=45, side='right', center_idx=0)
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
        self.task_list = [each[len(self.raw_seq_path):].split('.')[0] for each in glob(os.path.join(self.raw_seq_path, "s*/*.mano.npy"))]
        for task_name in tqdm(self.task_list, desc="Loading tasks...", total=len(self.task_list)):
            self.load_task(task_name)
            # print(f"Loaded {task_name} | total={self.mano_in_out_collector['right']['cum_length'][-1]}")
        np.savez("inv_mano/mano_pair.npz", **self.mano_in_out_collector)
        breakpoint()
        
    def load_task(self, task_name,):
        # camera
        '''
        K = processed_data["params"]["K_ego"][0]
        extrinsic_matrix_sequence = processed_data["params"]['world2ego'][:,:3]
        '''
        egocam_file = os.path.join(self.raw_seq_path, f"{task_name}.egocam.dist.npy")
        egocam = np.load(egocam_file, allow_pickle=True)[()]
        K = np.array(egocam['intrinsics'])  # (3, 3)
        Rt = egocam['R_k_cam_np']  # (T, 3, 3)
        Tt = egocam['T_k_cam_np']  # (T, 3, 1) 
        extrinsic_matrix_sequence = np.concatenate([Rt, Tt], axis=-1)  # (T, 3, 4)
        extrinsic_matrix_sequence = torch.FloatTensor(extrinsic_matrix_sequence).cuda()
        
        # hand
        hand_file = os.path.join(self.raw_seq_path, f"{task_name}.mano.npy")
        hand_data = np.load(hand_file, allow_pickle=True).item()
        num_frames = len(hand_data["right"]["rot"])

        for side in ['left', 'right']:
            # mano parameters -> vertices in world / camera frame
            mano_out = getattr(self, f"{side}_hand_model")(
                global_orient=torch.FloatTensor(hand_data[side]["rot"]).cuda(),
                hand_pose=torch.FloatTensor(hand_data[side]["pose"]).cuda(),
                betas=torch.FloatTensor(hand_data[side]["shape"]).repeat(num_frames, 1).cuda(),
                transl=torch.FloatTensor(hand_data[side]["trans"]).cuda()
            )
            hand_vertices = mano_out.vertices  # (T, 778, 3)
            hand_keypoints = mano_out.joints   # (T, 21, 3)
            self.mano_in_out_collector[side]["keypoints"] += hand_keypoints.cpu().numpy().tolist()
            self.mano_in_out_collector[side]["mano_params"]["theta"] += hand_data[side]["pose"].tolist()
            self.mano_in_out_collector[side]["mano_params"]["beta"] += repeat(hand_data[side]["shape"], 'D -> T D', T=num_frames).tolist()
            self.mano_in_out_collector[side]["mano_params"]["rot"] += hand_data[side]["rot"].tolist()
            self.mano_in_out_collector[side]["mano_params"]["trans"] += hand_data[side]['trans'].tolist()
            self.mano_in_out_collector[side]["cum_length"].append(self.mano_in_out_collector[side]["cum_length"][-1] + num_frames)
            self.mano_in_out_collector['task_list'].append(task_name)

if __name__ == "__main__":
    # ARCTICDatasetCollector()
    arctic_dataset = ARCTICDataset()
    arctic_dataset.initialize_data_storage(ARCTICStorage, dict(window_size=30, side='right'))
    print(len(arctic_dataset))
    batch = arctic_dataset[0]  # batch_size=1
    breakpoint()
    