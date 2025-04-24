from mhmp_vis.utils.pc import get_pc_from_world_to_camera, VisRenderer
from glob import glob
VisRenderer.visualize_depth_sequence(glob("datasets/ARCTIC/data/arctic_data/data/depths/s01/box_grab_01/0/*.png"),)