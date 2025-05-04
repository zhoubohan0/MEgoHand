import torch
import torch.nn.functional as F
import cv2
import numpy as np
import imageio.v3 as iio
import matplotlib
import matplotlib.pyplot as plt
import os, sys, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from gr00t.utils.util import clip_depth, concat4vis

def add_noise_to_depth(noisy_depth, noise_config):
    if 'gaussian_kernel' in noise_config:  # Gaussian smooth depth map
        kernel_size = noise_config['gaussian_kernel']['kernel_size']
        sigma = noise_config['gaussian_kernel']['sigma']
        
        gaussian_kernel = torch.Tensor(cv2.getGaussianKernel(kernel_size, sigma).reshape(1, -1)).repeat(1, kernel_size).reshape(1, 1, kernel_size, kernel_size)  # (1, 1, k, k)
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()  # Normalize the kernel
        noisy_depth = F.conv2d(noisy_depth, gaussian_kernel, padding='same') 
        noisy_depth = torch.clamp(noisy_depth, min=0.0, max=2.0)  # Clip to valid range

    if 'canny_edges' in noise_config:  # Sharpen Canny edges
        threshold = noise_config['canny_edges']['threshold']
        augmentation = noise_config['canny_edges']['augmentation']

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        padded_depth = F.pad(noisy_depth, (1, 1, 1, 1), mode='reflect')
        grad_x = F.conv2d(padded_depth, sobel_x, padding=0)
        grad_y = F.conv2d(padded_depth, sobel_y, padding=0)

        grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
        edges = (grad_mag > 0.5).float()

        noisy_depth[:, :, edges.squeeze() > 0] += augmentation
        noisy_depth = torch.clamp(noisy_depth, min=0.0, max=2.0)

    if 'gaussian_noise' in noise_config:  # Global Gaussian noise
        amplitude = noise_config['gaussian_noise']['amplitude']
        noisy_depth = torch.clamp(noisy_depth + torch.randn_like(noisy_depth) * amplitude, min=0.0, max=2.0)

    return noisy_depth

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Noisy Depth')
    parser.add_argument('--dep_path', '-i', type=str, required=True)
    parser.add_argument('--outdir', type=str, default='/mnt/vepfs/zbh/MHMP/vis_depth')
    args = parser.parse_args()

    depth_png = iio.imread(args.dep_path)
    depth_png = clip_depth(depth_png / 1000)
    print("original depth image:", depth_png.shape, depth_png.min(), depth_png.max())

    '''
    noise_config = {
        'gaussian_kernel': dict(kernel_size=3, sigma=1.0),
        'canny_edges': dict(threshold=0.5, augmentation=0.5),
        'gaussian_noise': dict(amplitude=0.1),
    }
    '''
    noise_config = {
        'gaussian_kernel': dict(kernel_size=3, sigma=1.0),
    }
    depth_true = torch.Tensor(depth_png).reshape(1, 1, *depth_png.shape[:2])
    noisy_depth = add_noise_to_depth(depth_true, noise_config)

    # visualize the first
    noisy_depth = noisy_depth[0, 0].cpu().numpy()  # Corrected indexing
    print("noisy depth image:", noisy_depth.shape, noisy_depth.min(), noisy_depth.max())

    out_dir = args.outdir
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, 'noisy_depth.png'), concat4vis(noisy_depth, depth_png, tag='Noisy'))
