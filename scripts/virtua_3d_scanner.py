import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import pyrender
import trimesh
import numpy as np
import imageio, cv2

'''
conda install -c menpo osmesa

sudo apt update
sudo wget https://github.com/mmatl/travis_debs/raw/master/xenial/mesa_18.3.3-0.deb
sudo dpkg -i ./mesa_18.3.3-0.deb || true
sudo apt install -f
apt --fix-broken install
'''



def render_rgbd_sequence(mesh_sequence, intrinsics_sequence, extrinsics_sequence, width, height, output_dir):
    # Create an offscreen renderer
    r = pyrender.OffscreenRenderer(width, height)

    for i, (mesh, intrinsics, extrinsics) in enumerate(zip(mesh_sequence, intrinsics_sequence, extrinsics_sequence)):
        # Load the mesh and add it to the scene
        mesh = trimesh.load(mesh) if isinstance(mesh, str) else mesh
        mesh = pyrender.Mesh.from_trimesh(mesh)
        scene = pyrender.Scene()
        scene.add(mesh)

        # Set up the camera
        fx, fy, cx, cy = intrinsics
        camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
        scene.add(camera, pose=extrinsics)

        # Render the RGB and depth images
        rgb_image, depth_image = r.render(scene)
        breakpoint()
        # Save
        imageio.imwrite(f"{output_dir}/rgb_{i:04d}.png", rgb_image)
        cv2.imwrite(f"{output_dir}/depth_{i:04d}.png", (depth_image * 1000).astype(np.uint16))
        # depth_image = cv2.imread(f"{output_dir}/depth_{i:04d}.png", cv2.IMREAD_UNCHANGED)
    r.delete()

# Example usage
if __name__ == "__main__":
    # Define mesh sequence (replace with your mesh files or trimesh objects)
    mesh_sequence = ["/mnt/vepfs/zbh/MHMP/datasets/FPHA/Object_models/milk_model/milk_model.ply"]

    # Define camera intrinsics sequence (fx, fy, cx, cy)
    intrinsics_sequence = [
        (525.0, 525.0, 320.0, 240.0)
    ]

    # Define camera extrinsics sequence (4x4 transformation matrices)
    extrinsics_sequence = [
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 15],
            [0, 0, 0, 1]
        ])
    ]

    # Define image size and output directory
    width, height = 640, 480
    output_dir = "output_sequence"

    # Render the RGBD sequence
    render_rgbd_sequence(mesh_sequence, intrinsics_sequence, extrinsics_sequence, width, height, output_dir)