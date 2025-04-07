import cv2
import argparse
import numpy as np
import trimesh
import torch
import imageio
import importlib
from tqdm import tqdm
from PIL import Image
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import pytorch3d
from pytorch3d.renderer import PointLights, AmbientLights, RasterizationSettings, MeshRenderer, MeshRasterizer, SoftPhongShader, TexturesVertex, PerspectiveCameras, look_at_view_transform
from pytorch3d.structures import Meshes, join_meshes_as_scene


class MeshRendererWrapper:
    "a simple wrapper for the pytorch3d mesh renderer"
    def __init__(self, image_size=(1200, 900),
                 faces_per_pixel=1,
                 device='cuda:0',
                 blur_radius=0, lights=None,
                 materials=None, max_faces_per_bin=50000):
        self.image_size = image_size
        self.faces_per_pixel=faces_per_pixel
        self.max_faces_per_bin=max_faces_per_bin # prevent overflow, see https://github.com/facebookresearch/pytorch3d/issues/348
        self.blur_radius = blur_radius
        self.device = device
        self.lights=lights if lights is not None else AmbientLights(
            ambient_color=((0.5, 0.5, 0.5),), device=device
        )
        self.materials = materials
        self.renderer = self.setup_renderer()

    def setup_renderer(self):
        # for sillhouette rendering
        sigma = 1e-4
        raster_settings = RasterizationSettings(
            image_size=self.image_size[::-1],  # input: (h, w)
            blur_radius=self.blur_radius,
            # blur_radius=np.log(1. / 1e-4 - 1.) * sigma, # this will create large sphere for each face
            faces_per_pixel=self.faces_per_pixel,
            clip_barycentric_coords=False,
            max_faces_per_bin=self.max_faces_per_bin
        )
        shader = SoftPhongShader(
            device=self.device,
            lights=self.lights,
            materials=self.materials)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=raster_settings),
                shader=shader
        )
        return renderer

    def render(self, meshes, cameras, ret_mask=False):
        images = self.renderer(meshes, cameras=cameras)
        if ret_mask:
            mask = images[0, ..., 3].cpu().detach().numpy()
            return images[0, ..., :3].cpu().detach().numpy(), mask > 0
        return images[0, ..., :3].cpu().detach().numpy()


class Pyt3DWrapper:
    def __init__(self, image_size, device='cuda:0', use_fixed_cameras=False, eyes=None, intrin=None, extrin=None, lights=None):
        if lights is None:
            lights=PointLights(device='cuda', location=[[0.0, 0.0, 1]])
        self.device = device
        self.image_size = image_size
        self.renderer = MeshRendererWrapper(image_size, device=device, lights=lights)
        self.use_fixed_cameras = use_fixed_cameras
        if use_fixed_cameras:
            self.setup_intrin_extrin(intrin, extrin)
        else: 
            camera_properties = dict(device=device)
            if not eyes is None:
                camera_properties['eyes'] = eyes
                camera_properties['n_poses'] = len(eyes)
            self.cameras = self.get_surround_cameras(**camera_properties)
    
    def setup_intrin_extrin(self, intrin, extrin):
        focal_length = torch.tensor((-intrin[0, 0], -intrin[1, 1]), dtype=torch.float32).unsqueeze(0)
        cam_center = torch.tensor((intrin[0, 2], intrin[1, 2]), dtype=torch.float32).unsqueeze(0)
        R = torch.from_numpy(extrin[:3, :3].T).unsqueeze(0)
        T = torch.from_numpy(extrin[:3, 3]).unsqueeze(0)
        camera_properties = dict(focal_length=focal_length, principal_point=cam_center, image_size=((self.image_size[1], self.image_size[0]),), device=self.device, R=R, T=T)
        if pytorch3d.__version__ >= '0.6.0':
            camera_properties['in_ndc'] = False
        self.cameras = [PerspectiveCameras(**camera_properties)]

    @staticmethod
    def get_surround_cameras(radius=0.5, eyes=None, n_poses=30, up=(0.0, 0.0, 1.0), device='cuda:0'):
        breakpoint()
        # default
        fx, fy = 1000, 1000  # focal length
        cx, cy = 512, 512  # camera centers
        color_w, color_h = 1024, 1024  # mock
        cam_center = torch.tensor((cx, cy), dtype=torch.float32).unsqueeze(0)
        focal_length = torch.tensor((fx, fy), dtype=torch.float32).unsqueeze(0)

        cameras = []
        if eyes is None:
            eyes = []
            for theta in np.linspace(0, 2 * np.pi, n_poses + 1)[:-1]:
                if np.abs(up[1]) > 0:
                    eye = [np.cos(theta + np.pi / 2) * radius, 0.2, -np.sin(theta + np.pi / 2) * radius]
                else:
                    eye = [np.cos(theta + np.pi / 2) * radius, np.sin(theta + np.pi / 2) * radius, 0.2]
                print(eye)
                eyes.append(eye)
        
        for eye in eyes:
            R, T = look_at_view_transform(
                eye=(eye,),
                at=([0.0, 0.0, 0.0],),
                up=(up,),
            )
            camera_properties = dict(focal_length=focal_length, principal_point=cam_center, image_size=((color_h, color_w),), device=device, R=R, T=T)
            if pytorch3d.__version__ >= '0.6.0':
                camera_properties['in_ndc'] = False
            cam = PerspectiveCameras(**camera_properties)
            cameras.append(cam)
        return cameras

    def render_meshes(self, meshes, specified_vertices=None):
        """
        render a list of meshes
        :param meshes: a list of psbody meshes
        :return: rendered image
        """
        pyt3d_mesh = self.prepare_render(meshes, specified_vertices)
        rends = []
        for cam in self.cameras:
            img = self.renderer.render(pyt3d_mesh, cam)
            rends.append(img)
        return rends

    def prepare_render(self, meshes, specified_vertices, static_contact_color=[0.9, 0.2, 0.2]):
        COLOR_LIST = [
            [1.0, 0.0, 0.0],  # right hand
            [1.0, 1.0, 1.0],  # left hand
            [0.0, 0.0, 1.0],  # tool 
            [120/255, 120/255, 200/255],  # target object
            [200/255, 200/255, 20/255],  # smplx
        ]
        py3d_meshes = []
        if specified_vertices is None:
            specified_vertices = [None] * len(meshes)
        for mesh, sp_vs, color in zip(meshes, specified_vertices, COLOR_LIST):
            vc = np.zeros_like(mesh.vertices)
            vc[:, :] = color
            if not sp_vs is None:  # render contact
                vc[sp_vs] = static_contact_color
            text = TexturesVertex([torch.from_numpy(vc).float().to(self.device)])
            py3d_mesh = Meshes([torch.from_numpy(mesh.vertices).float().to(self.device)], [torch.from_numpy(mesh.faces.astype(int)).long().to(self.device)], text)
            py3d_meshes.append(py3d_mesh)
        joined = join_meshes_as_scene(py3d_meshes)
        return joined




def offline_render(
    image_filenames,
    world_left_hand_verts, left_hand_faces,
    world_right_hand_verts, right_hand_faces,
    world_object_dict_list,
    intrinsic_matrix, extrinsic_matrix_sequence,
    save_video_file,
    world_smplx_verts=None, smplx_faces=None,
    sampling_rate=1,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    **kwargs  # filter unused args
):
    '''
    world_object_dict_list: each object_info_dict consists of ['world_object_verts', 'object_faces'] 
    '''
    image_size = Image.open(image_filenames[0]).size if image_filenames is not None and len(image_filenames) > 0 else (1024, 750)
    pyt3d_wrapper = Pyt3DWrapper(image_size=image_size, use_fixed_cameras=True, intrin=intrinsic_matrix, extrin=extrinsic_matrix_sequence[0], device=device)

    # motion_writer = imageio.get_writer(save_video_file)
    overlay_writer = imageio.get_writer(save_video_file.replace(".mp4", "_overlay.mp4"))

    N_frame = len(image_filenames) if image_filenames is not None else len(world_right_hand_verts)
    sampling_rate = max(sampling_rate, np.ceil(N_frame / 250).astype(int))
    for frame_idx in tqdm(range(0, N_frame, sampling_rate), desc="Offline Rendering frames..."):
        # construct object meshes in this frame
        meshes = []
        if not world_right_hand_verts is None:
            meshes += [trimesh.Trimesh(world_right_hand_verts[frame_idx], right_hand_faces)] # right hand   
        if not world_left_hand_verts is None:
            meshes += [trimesh.Trimesh(world_left_hand_verts[frame_idx], left_hand_faces)]  # left hand
        meshes += [
            trimesh.Trimesh(object_info_dict['world_object_verts'][frame_idx], object_info_dict['object_faces']) # object
            for object_info_dict in world_object_dict_list
        ]
        if not world_smplx_verts is None and not smplx_faces is None:
            meshes += [trimesh.Trimesh(world_smplx_verts[frame_idx], smplx_faces)]
        
        # render motion
        pyt3d_wrapper.setup_intrin_extrin(intrin=intrinsic_matrix, extrin=extrinsic_matrix_sequence[frame_idx]) # setup extrinsic
        motion_img = (pyt3d_wrapper.render_meshes(meshes)[0] * 255).astype(np.uint8)  # (H, W, 3), [0] means one camera
        # motion_writer.append_data(cv2.resize(motion_img, (image_size[0], image_size[1])))
        
        # overlay RGB and motion
        if image_filenames is not None and len(image_filenames) > 0:
            rgb_img = np.array(Image.open(image_filenames[frame_idx]))
            overlay_img = cv2.addWeighted(rgb_img, 0.5, motion_img, 0.5, 0)
            overlay_writer.append_data(cv2.resize(overlay_img, (image_size[0], image_size[1])))
        
    # motion_writer.close()
    overlay_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", '-d', type=str, default="arctic")
    args = parser.parse_args()

    # data loading
    dataset_name = args.dataset_name.lower()
    module = importlib.import_module(f"dataset_loader.dataset_{dataset_name}")
    dataset_class = getattr(module, f"{dataset_name.upper()}Dataset")
    dataset = dataset_class()
    data_sample = dataset.task_example_for_vis()

    # offline render
    offline_render(**data_sample)

    
