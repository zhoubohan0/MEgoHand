import os
import sys
import json
import argparse
import pickle
import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
import imageio.v3 as iio
import huggingface_hub
from unidepth.models import UniDepthV2
sys.path.append(os.path.join(os.path.dirname(__file__),'../'*3))
from gr00t.utils.util import visualize_RGBD, clip_depth


def unproject_depth_to_3d(depth_map, intrinsic_matrix):
    """
    Unproject a 2D depth map to 3D coordinates using the camera intrinsic matrix.

    Parameters:
    - depth_map (np.ndarray): 2D array of shape (H, W) containing depth values.
    - intrinsic_matrix (np.ndarray): 3x3 intrinsic matrix of the camera.

    Returns:
    - points_3d (np.ndarray): 3D coordinates of shape (H, W, 3).
    """
    H, W = depth_map.shape
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    # Create a grid of (u, v) coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Compute the 3D coordinates
    X = (u - cx) * depth_map / fx
    Y = (v - cy) * depth_map / fy
    Z = depth_map

    # Stack X, Y, Z into a single (H, W, 3) array
    points_3d = np.dstack((X, Y, Z))

    return points_3d


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="UniDepth Inference Example")
    parser.add_argument('--rgb_path', '-i', type=str, required=True)
    parser.add_argument('--dep_path', '-d', type=str, required=True)
    parser.add_argument('--vit_type', type=str, default='s', choices=['s', 'b', 'l'])
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    args = parser.parse_args()

    # depth model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    version = 'v2'
    backbone = f'vit{args.vit_type}14'
    name = f"unidepth-{version}-{backbone}"  # vit*14 / cnvnxtl 
    # model = UniDepthV2.from_pretrained(f"lpiccinelli/{name}").to(device) 
    # model = torch.hub.load("lpiccinelli-eth/UniDepth", "UniDepth", version="v2", backbone="vits14", pretrained=True, trust_repo=True).to(device)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, "unidepth/configs", f"config_{version}_{backbone}.json")) as f:
        config = json.load(f)
    model = UniDepthV2(config)
    ckpt_path = f'/mnt/vepfs/zbh/backup/pretrain/unidepth/models--lpiccinelli--{name}/pytorch_model.bin'
    info = model.load_state_dict(torch.load(ckpt_path), strict=True)
    print(f"{name} is loaded with:")
    print(f"\t missing keys: {info.missing_keys}")
    print(f"\t additional keys: {info.unexpected_keys}")

    # load RGB (single)
    rgb = iio.imread(args.rgb_path)

    # inference
    predictions = model.infer(torch.from_numpy(rgb).permute(2, 0, 1))  # C, H, W
    # predictions = model.infer(rgb, intrinsics) # Uncomment this line if you have intrinsics
    depth_pred = predictions["depth"]    # Metric Depth Estimation
    xyz = predictions["points"]     # Point Cloud in Camera Coordinate
    intrinsics = predictions["intrinsics"] # Intrinsics Prediction

    # visualize 
    depth_pred = depth_pred[0,0].cpu().numpy()
    depth_pred = clip_depth(depth_pred)
    depth_gt = iio.imread(args.dep_path) / 1000
    depth_gt = clip_depth(depth_gt)
    print("GT depth image:", depth_gt.shape, depth_gt.min(), depth_gt.max())
    print("pred depth image:", depth_pred.shape, depth_pred.min(), depth_pred.max())
    visualize_RGBD(
        dict(depth=depth_pred, depth_gt=depth_gt, rgb=rgb), 
        save_file=os.path.join(args.outdir, 'unidepth', os.path.splitext(os.path.basename(args.rgb_path))[0] + '.png')
    )
    # unproject depth to 3D
    intrinsics = intrinsics[0].cpu().numpy()
    points_3d = unproject_depth_to_3d(depth_pred, intrinsics)
