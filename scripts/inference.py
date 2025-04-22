import os
import torch
import numpy as np
from pprint import pprint
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.policy import MHMPolicy
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.utils.util import glance_dict
from mhmp_vis.dataset_loader import *

# model
data_config = DATA_CONFIG_MAP["hand_motion"]
modality_config = data_config.modality_config()
glance_dict(modality_config)
modality_transform = data_config.transform(split='eval')
policy = MHMPolicy(
    model_cfg_path='model_config',
    modality_config=modality_config,
    modality_transform=modality_transform,
)
# print(policy.model)

'''
# dataset
DATASET_PATH = os.path.join("demo_data/robot_sim.PickNPlace")
dataset = LeRobotSingleDataset(
    dataset_path=DATASET_PATH,
    modality_configs=modality_config,
    video_backend="decord",
    video_backend_kwargs=None,
    transforms=None,  # We'll handle transforms separately through the policy
    embodiment_tag=EMBODIMENT_TAG,
)
step_data = dataset[0]
print("\n\n ====================================")
glance_dict(step_data)
'''

'''
# visualize data sample        
import matplotlib.pyplot as plt

traj_id = 0
max_steps = 150

state_joints_across_time = []
gt_action_joints_across_time = []
images = []

sample_images = 6

for step_count in range(max_steps):
    data_point = dataset.get_step_data(traj_id, step_count)
    state_joints = data_point["state.right_arm"][0]
    gt_action_joints = data_point["action.right_arm"][0]
    
   
    state_joints_across_time.append(state_joints)
    gt_action_joints_across_time.append(gt_action_joints)

    # We can also get the image data
    if step_count % (max_steps // sample_images) == 0:
        image = data_point["video.ego_view"][0]
        images.append(image)

# Size is (max_steps, num_joints == 7)
state_joints_across_time = np.array(state_joints_across_time)
gt_action_joints_across_time = np.array(gt_action_joints_across_time)


# Plot the joint angles across time
fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(8, 2*7))

for i, ax in enumerate(axes):
    ax.plot(state_joints_across_time[:, i], label="state joints")
    ax.plot(gt_action_joints_across_time[:, i], label="gt action joints")
    ax.set_title(f"Joint {i}")
    ax.legend()

plt.tight_layout()
plt.show()


# Plot the images in a row
fig, axes = plt.subplots(nrows=1, ncols=sample_images, figsize=(16, 4))

for i, ax in enumerate(axes):
    ax.imshow(images[i])
    ax.axis("off")
'''    
'''
step_data = dict()
step_data["video.ego_view"] = np.zeros((1,256,256,3), dtype=np.uint8)
step_data["state.theta"] =  np.zeros((1,90), dtype=np.float32)
step_data["state.beta"] = np.zeros((1,10), dtype=np.float32)
step_data["state.rot"] = np.zeros((1, 6), dtype=np.float32)
step_data["state.trans"] = np.zeros((1, 3), dtype=np.float32)
step_data["action.theta"] = np.zeros((16,90), dtype=np.float32)
step_data["action.rot"] = np.zeros((16, 6), dtype=np.float32)
step_data["action.trans"] = np.zeros((16, 3), dtype=np.float32)
step_data['annotation.human.action.task_description'] = ['pick the squash from the counter and place it in the plate']
'''

test_dataset_name_list = [
    # "TACO",       # zbh (135G) bash '/home/ps/Desktop/Multimodal-Hand-Motion-Pretraining/upload.sh' -p '/media/ps/ZBH/dataset/TACO/TACO-overall' -t /mnt/vepfs/zbh/MHMP/datasets/TACO -e "./2D_Segmentation.zip ./README.md"
    # "Arctic",     # zhan_4t (600G) 
    # "oakink2",    # zhan_4t (2.1T) 
    # "H2O",        # zhan_4t (751G) done
    "hoi4d",      # zhan (178G) bash '/home/ps/Desktop/Multimodal-Hand-Motion-Pretraining/upload.sh' -p '/media/ps/zhan/DATASET/HOI4D' -t /mnt/vepfs/zbh/MHMP/datasets -e "camera_params definitions HOI4D_CAD_Model_for_release HOI4D_release prepare_4Dseg utils README.md task_list.txt"
    # "fpha",         # zhan (28G) bash '/home/ps/Desktop/Multimodal-Hand-Motion-Pretraining/upload.sh' -p '/media/ps/zhan/DATASET/FPHA.tar.gz'  -t /mnt/vepfs/zbh/MHMP/datasets -e ""
    # "hot3d",      # T9 (247G) bash '/home/ps/Desktop/Multimodal-Hand-Motion-Pretraining/upload.sh' -p /media/ps/T7/HOT3D.tar.gz
    ##------------------------------
    # -----------------------
    # "Assembly101", # T9
    # -----------------------
    # "ConcatPose", # zbh
    # "DexYCB", # zbh
    # "GRAB", # zbh
    # "HIMO", # zbh
    # "HO3D", # zbh
    # "HOGraspNet", # zbh
    # -----------------------
]
test_dataset_list = []
for each in test_dataset_name_list:
    dataset_name = each.upper()
    dataset = eval(f"{dataset_name}Dataset")()
    dataset.initialize_data_storage(eval(f"{dataset_name}Storage"), dict(window_size=data_config.action_horizon, side='right'), split=dict(train=1.0))
    test_dataset_list.append(dataset)
cascaded_dataset = CascadedDataset(test_dataset_list)

# inference a sample
step_data = cascaded_dataset[0]
predicted_action = policy.get_action(step_data)
glance_dict(predicted_action)
breakpoint()
