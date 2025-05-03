import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
from depth_anything_v2.dpt import DepthAnythingV2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--rgb_path', '-i', type=str, default="/mnt/vepfs/zbh/MHMP/datasets/TACO/Egocentric_RGB_Frames/brush,brush,bowl/20230919_036/000001.jpg")
    parser.add_argument('--dep_path', '-d', type=str, default="/mnt/vepfs/zbh/MHMP/datasets/TACO/Egocentric_Depth_Frames/brush,brush,bowl/20230919_036/000001.png")
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
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
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        raw_image = cv2.imread(filename)
        # raw_image = cv2.resize(raw_image, (640,480))
        depth = depth_anything.infer_image(raw_image, args.input_size)
        depth = depth.clip(0,2)
        # normalization for visualization
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        if args.pred_only:
            cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), depth)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth])
            cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), combined_result)
        depth_png = cv2.imread(args.dep_path, cv2.IMREAD_ANYDEPTH)
        print(depth_png.min(), depth_png.max(), depth_png.shape)
        depth_png = (depth_png / 1000).clip(0,2)
        depth_png = (depth_png - depth_png.min()) / (depth_png.max() - depth_png.min()) * 255.0
        depth_png = (cmap(depth_png)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        breakpoint()
        cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '_cmp.png'), cv2.hconcat([depth, depth_png]))
        breakpoint()