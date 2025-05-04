import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
from depth_anything_v2.dpt import DepthAnythingV2
from gr00t.utils.util import clip_depth, concat4vis

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    parser.add_argument('--rgb_path', '-i', type=str, required=True)
    parser.add_argument('--dep_path', '-d', type=str, required=True)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'/mnt/vepfs/zbh/backup/pretrain/depth_anything_v2/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    if os.path.isfile(args.rgb_path):
        if args.rgb_path.endswith('txt'):
            with open(args.rgb_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.rgb_path]
    else:
        filenames = glob.glob(os.path.join(args.rgb_path, '**/*'), recursive=True)
    
    os.makedirs(args.outdir, exist_ok=True)
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        raw_image = cv2.imread(filename)
        # raw_image = cv2.resize(raw_image, (640,480)) # for fpha
        # raw_image = cv2.resize(raw_image, (1024, 768)) # for taco
        print("RGB image:", raw_image.shape, raw_image.min(), raw_image.max())
        depth = depth_anything.infer_image(raw_image, args.input_size)
        depth = clip_depth(depth)
        print("pred depth image:", depth.shape, depth.min(), depth.max())
        
        depth_png = cv2.imread(args.dep_path, cv2.IMREAD_ANYDEPTH)
        depth_png = clip_depth(depth_png / 1000)
        print("GT depth image:", depth_png.shape, depth_png.min(), depth_png.max())
        
        cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), concat4vis(depth, depth_png, raw_image))
        