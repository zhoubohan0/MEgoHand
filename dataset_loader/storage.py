import numpy as np
from tqdm import tqdm
import torch, cv2
from torch.utils.data import Dataset
from utils.util import read_single_image, find_lower_left
from utils.rot import compute_relative_pose_in_sequence, rotmat_to_rot6d, aa_to_rot6d
from natsort import natsorted

class Storage:
    '''
    For sub-class, extra properties can be added
    1. side: left or right to index
    2. render_rgb / render_dep
    '''
    def __init__(self, window_size, **config):
        self.window_size = window_size  # (W,)
        self.length_list = []           # (L,) number of trajectory
        self.cumsum_length_list = [0]   # (L,) number of trajectory
        self.valid_timestep_list = []   # (T',) approximate number of frames, T = sum(L)
        # flexible fields exactly used for training batch!
        self.storage = {
            'text':[],                      # (T,) repeated task instruction
            'ego_rgb_filenames':[],             # (T,H,W,3) real images / hand-object mesh rendering
            'ego_depth_filenames':[],           # (T,H,W) real depth / hand-object mesh rendered depth
            'motion_sequences': dict( 
                theta=[],               # (T,15*3) 
                beta=[],                # (10)
                rot=[],                 # (T,3,3) rotmat
                trans=[],               # (T,3)  
            )
        }
        # extra config
        for k, v in config.items():
            setattr(self, k, v)
        
    def fill_data(self, task_data:dict) -> dict:
        return dict(
            text='',
            ego_rgb_files=['*.jpg'],
            ego_dep_files=['*.png'],
            theta_seqs=np.zeros((100, 15*3)),
            beta_seqs=np.zeros((1, 10)),
            rot_seqs=np.zeros((100, 3, 3)),
            trans_seqs=np.zeros((100, 3)),
            len=100,
        )
        
    def add(self, task_data_dict):
        fill_dict = self.fill_data(task_data_dict)
        if fill_dict is None:
            return
        # text
        self.storage["text"].append(fill_dict['text'])   # (1,)                                        
        # vision
        self.storage["ego_rgb_filenames"].extend(fill_dict["ego_rgb_files"])        # (T,)     
        self.storage["ego_depth_filenames"].extend(fill_dict["ego_dep_files"])      # (T,)
        # hand
        self.storage["motion_sequences"]['theta'].append(fill_dict["theta_seqs"])  # (T,15*3)
        self.storage["motion_sequences"]['beta'].append(fill_dict["beta_seqs"])    # (1,10)
        self.storage["motion_sequences"]['rot'].append(fill_dict["rot_seqs"])      # (T,3,3)
        self.storage["motion_sequences"]['trans'].append(fill_dict["trans_seqs"])  # (T,3)
        # record length
        length = fill_dict['len']
        cumsum_length = self.cumsum_length_list[-1]
        self.length_list.append(length)
        self.cumsum_length_list.append(length + cumsum_length)
        self.valid_timestep_list += list(range(cumsum_length, cumsum_length + length - self.window_size))
    
    def extend(self, task_data_list):
        for task_data in task_data_list:
            self.add(task_data)
    
    def post_initialization(self):  # list to numpy
        self.valid_timestep_list = np.array(self.valid_timestep_list)
        for k, v in self.storage[f"motion_sequences"].items():
            if len(v):
                self.storage[f"motion_sequences"][k] = np.concatenate(v, axis=0)

    
    def __len__(self):
        return len(self.valid_timestep_list)
    
    def __getitem__(self, idx, relative=False):
        '''
        idx is the index of valid_timestep_list
        '''
        frame_id = self.valid_timestep_list[idx]
        # binary search the trajectory index, and index the special
        traj_id = find_lower_left(self.cumsum_length_list, frame_id)

        if relative:
            # calculate relative hand pose
            future_hand_rotmat, future_hand_trans = compute_relative_pose_in_sequence(
                torch.Tensor(self.storage["motion_sequences"]['rot'][frame_id:frame_id+self.window_size+1]),
                torch.Tensor(self.storage["motion_sequences"]['trans'][frame_id:frame_id+self.window_size+1]),
            )
        else:
            future_hand_rotmat = torch.Tensor(self.storage["motion_sequences"]['rot'][frame_id+1:frame_id+self.window_size+1]) 
            future_hand_trans = torch.Tensor(self.storage["motion_sequences"]['trans'][frame_id+1:frame_id+self.window_size+1])

        return {
            "video.ego_view": read_single_image(self.storage["ego_rgb_filenames"][frame_id]),  # (1, H, W, 3)
            "depth.ego_view": read_single_image(self.storage["ego_depth_filenames"][frame_id]),  # (1, H, W)
            # H(t)
            "state.theta": aa_to_rot6d(torch.Tensor(self.storage["motion_sequences"]['theta'][frame_id:frame_id+1]).view(-1,3)).view(-1,15*6).cpu().numpy(),  # (1, 15*6)
            "state.beta": self.storage["motion_sequences"]['beta'][traj_id:traj_id+1],  # (1, 10)
            "state.rot": rotmat_to_rot6d(torch.Tensor(self.storage["motion_sequences"]['rot'][frame_id:frame_id+1])).cpu().numpy(),  # (1, 1*6)
            "state.trans": self.storage["motion_sequences"]['trans'][frame_id:frame_id+1],  # (1, 3)
            # [Delta] H(t+1:t+W)
            "action.theta": aa_to_rot6d(torch.Tensor(self.storage["motion_sequences"]['theta'][frame_id+1:frame_id+self.window_size+1]).view(-1,3)).view(-1,15*6).cpu().numpy(),  # (T, 15*6)
            "action.rot": rotmat_to_rot6d(future_hand_rotmat).cpu().numpy(),  # (T, 1*6)
            "action.trans": future_hand_trans.cpu().numpy(),  # (T, 3)
            # instruction
            "annotation.human.action.task_description": self.storage["text"][traj_id:traj_id+1],  # (1,)
        } 

        
class MetaDataset(Dataset):
    def __init__(self, root_path):
        super().__init__()
        # path
        self.root_path = root_path
        
        # all object mesh: (verts, faces)
        self.object_meshes = None
        
        # load task
        self.task_list = []
        
        # 'side' data storage
        self.data_storage: Storage
    
    def __str__(self):
        class_name = self.__class__.__name__
        if class_name.endswith('Dataset'):
            return class_name[:-7]  # 'XDataset' -> 'X'
        return class_name
        
    def initialize_data_storage(self, data_storage_class, data_storage_kwargs, split=dict(train=0.95)):
        '''
        data_storage_class: Storage
        data_storage_kwargs: {
            'window_size': int,
            'side': str,
            'is_render_depth': bool,
        }
        split: {
            'train': float,
            'eval': float,
        }
        '''
        self.task_list = natsorted(self.task_list)  # ensure unique order
        if 'train' in split:  # train, can be 100%
            self.task_list = self.task_list[:int(len(self.task_list) * split['train'])]
        else:  # eval, default 5%
            self.task_list = self.task_list[int(len(self.task_list) * split['eval']):]
        self.data_storage = data_storage_class(**data_storage_kwargs)
        for task_name in tqdm(self.task_list, desc="Loading tasks..."):
            task_data_dict = self.load_task(task_name)
            if isinstance(task_data_dict, list):
                self.data_storage.extend(task_data_dict)
            else:
                self.data_storage.add(task_data_dict)
        self.data_storage.post_initialization()
    
    def load_task(self, task_name):
        raise NotImplementedError
    
    def task_example_for_vis(self): 
        task_data_dict = self.load_task()  # load_task should have default task_name
        if isinstance(task_data_dict, list):
            task_data_dict = task_data_dict[0]
        return dict(
            # common
            image_filenames=task_data_dict['real_image_filenames']['ego_rgb'],
            intrinsic_matrix=task_data_dict['intrinsic_matrix']['ego'],
            # for interactive rendering in camera frame
            cam_left_hand_verts=task_data_dict['hand_dict']['left']['mesh']['verts']['cam'] if task_data_dict['hand_dict']['left'] is not None else None,
            cam_right_hand_verts=task_data_dict['hand_dict']['right']['mesh']['verts']['cam'] if task_data_dict['hand_dict']['right'] is not None else None,
            cam_smplx_verts=None,
            cam_object_dict_list=[dict(cam_object_verts=obj_dic['verts']['cam'], object_faces=obj_dic['faces']) for obj_dic in task_data_dict['object_dict'].values()],
            # for offline rendering in world frame
            world_left_hand_verts=task_data_dict['hand_dict']['left']['mesh']['verts']['world'] if task_data_dict['hand_dict']['left'] is not None else None, 
            left_hand_faces=task_data_dict['hand_dict']['left']['mesh']['faces'] if task_data_dict['hand_dict']['left'] is not None else None,
            world_right_hand_verts=task_data_dict['hand_dict']['right']['mesh']['verts']['world'] if task_data_dict['hand_dict']['right'] is not None else None, 
            right_hand_faces=task_data_dict['hand_dict']['right']['mesh']['faces'] if task_data_dict['hand_dict']['right'] is not None else None,
            world_object_dict_list=[dict(world_object_verts=obj_dic['verts']['world'], object_faces=obj_dic['faces']) for obj_dic in task_data_dict['object_dict'].values()],
            extrinsic_matrix_sequence=task_data_dict['extrinsic_matrix_sequence']['ego'],
            save_video_file=f'offline_visualization/{str(self)}_{task_data_dict["task_name"].replace("/","++")}.mp4',
        )  
    
    @staticmethod    
    def check_length(*args):
        assert np.var([len(data) for data in args]) == 0
        return len(args[0])
        
    def __len__(self):
        return len(self.data_storage) if self.data_storage is not None else 0
    
    def __getitem__(self, idx):
        return self.data_storage[idx]
    

