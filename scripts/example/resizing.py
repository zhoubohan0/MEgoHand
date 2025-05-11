import os
from pathlib import Path
from torchvision.io import read_image, write_jpeg
import torch
import torch.nn.functional as F
from tqdm import tqdm
from glob import glob
from natsort import natsorted
from gr00t.utils.util import timer
import multiprocessing as mp
from functools import partial

def resize_image(image_file, resize_size):
    """Resize a single image and return the resized tensor"""
    image = read_image(image_file)
    image = F.interpolate(image[None,...], 
                         size=resize_size, 
                         mode='bilinear', 
                         align_corners=True)[0]
    return image

def process_single_image(image_file, save_file, resize_size):
    """Process a single image with error handling"""
    try:
        # If file exists, verify it can be read
        if os.path.exists(save_file):
            try:
                read_image(save_file)
                return  # File exists and is valid
            except Exception:
                pass  # File exists but is corrupt, regenerate it
        
        # Generate resized image
        image = resize_image(image_file, resize_size)
        write_jpeg(image, save_file)
        
    except Exception as e:
        print(f"Error processing {image_file}: {e}")

def process_task(task, root_dir, resize_size=(256, 256)):
    """Process all images in a task directory"""
    print(f"Processing {task}")
    
    # Setup directories
    input_dir = os.path.join(root_dir, task, '0')
    output_dir = input_dir.replace("images", "images_resize256")
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all images
    image_files = natsorted(glob(os.path.join(input_dir, "*.jpg")))
    for image_file in image_files:
        save_file = image_file.replace("images", "images_resize256")
        process_single_image(image_file, save_file, resize_size)

if __name__ == '__main__':
    # Paths
    root_dir = "/mnt/vepfs/zbh/MHMP/datasets/ARCTIC/data/arctic_data/data/images"
    task_list_path = "/mnt/vepfs/zbh/MHMP/datasets/ARCTIC/task_list.txt"

    # Read task list
    with open(task_list_path, 'r') as f:
        task_list = natsorted([line.strip() for line in f if line.strip()])[::-1]

    # Process tasks in parallel with progress bar
    process_fn = partial(process_task, root_dir=root_dir)
    for task in tqdm(task_list, desc="Resizing"):
        process_fn(task)
    # with mp.Pool(processes=mp.cpu_count()) as pool:
    #     list(tqdm(pool.imap_unordered(process_fn, task_list), total=len(task_list), desc="Resizing"))

