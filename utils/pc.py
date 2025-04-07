import os
import open3d
import torch
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange, repeat

COLORS = {
    'red': (0.969, 0.106, 0.059),
    'blue': (0.0, 0.0, 1.0),
    'green': (1.0, 0.0, 0.0),
    'cyan': (0.051, 0.659, 0.051),
    'light-blue': (0.588, 0.5647, 0.9725),
    'cyan-light': (0.051, 0.659, 0.051),
    'dark-light': (0.404, 0.278, 0.278),
    'rice': (0.922, 0.922, 0.102),
    'white': (1.0, 1.0, 1.0),
}

class VisOpen3D:
    def initialize(self, width, height, visible=True):
        self.__vis = open3d.visualization.Visualizer()
        self.__vis.create_window(width=width, height=height, visible=visible)
        self.__width = width
        self.__height = height
        if visible:
            self.poll_events()
            self.update_renderer()

    def __init__(self, width=1920, height=1080, visible=True):
        self.initialize(width, height, visible)

    def __del__(self):
        self.__vis.destroy_window()

    def render(self):
        self.__vis.poll_events()
        self.__vis.update_renderer()
        self.__vis.run()

    def poll_events(self):
        self.__vis.poll_events()

    def update_renderer(self):
        self.__vis.update_renderer()

    def run(self):
        self.__vis.run()

    def destroy_window(self):
        self.__vis.destroy_window()

    def add_geometry(self, data):
        self.__vis.add_geometry(data)

    def add_pointcloud(self, pc):
        '''
        pc: (n, 3) array
        '''
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(pc)
        self.add_geometry(pcd)
        
    def add_mesh(self, mesh_dict):
        '''
        mesh_dict: dict(verts, faces)
        '''
        mesh = open3d.geometry.TriangleMesh()
        mesh.vertices = open3d.utility.Vector3dVector(mesh_dict['verts'])
        mesh.triangles = open3d.utility.Vector3iVector(mesh_dict['faces'])
        mesh.compute_vertex_normals()
        self.add_geometry(mesh)

    def update_view_point(self, intrinsic, extrinsic):
        ctr = self.__vis.get_view_control()
        param = self.convert_to_open3d_param(intrinsic, extrinsic)
        ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
        self.__vis.update_renderer()

    def get_view_point_intrinsics(self):
        ctr = self.__vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        intrinsic = param.intrinsic.intrinsic_matrix
        return intrinsic

    def get_view_point_extrinsics(self):
        ctr = self.__vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        extrinsic = param.extrinsic
        return extrinsic

    def get_view_control(self):
        return self.__vis.get_view_control()

    def save_view_point(self, filename):
        ctr = self.__vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        open3d.io.write_pinhole_camera_parameters(filename, param)

    def convert_to_open3d_param(self, intrinsic, extrinsic):
        param = open3d.camera.PinholeCameraParameters()
        param.intrinsic = open3d.camera.PinholeCameraIntrinsic()
        param.intrinsic.intrinsic_matrix = intrinsic
        param.extrinsic = extrinsic
        return param

    def capture_screen_float_buffer(self, show=False):
        image = self.__vis.capture_screen_float_buffer(do_render=True)

        if show:
            plt.imshow(image)
            plt.show()

        return image

    def capture_screen_image(self, filename):
        if os.path.exists(filename):
            return 
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.__vis.capture_screen_image(filename, do_render=True)

    def capture_depth_float_buffer(self, show=False):
        depth = self.__vis.capture_depth_float_buffer(do_render=True)

        if show:
            plt.imshow(depth)
            plt.show()

        return np.array(depth)

    def capture_depth_image(self, filename):
        if os.path.exists(filename):
            return 
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.__vis.capture_depth_image(filename, do_render=True)

        # to read the saved depth image file use:
        # depth = open3d.io.read_image(filename)
        # plt.imshow(depth)
        # plt.show()

    def draw_camera(self, intrinsic, extrinsic, scale=1, color=None):
        # intrinsics
        K = intrinsic

        # convert extrinsics matrix to rotation and translation matrix
        extrinsic = np.linalg.inv(extrinsic)
        R = extrinsic[0:3,0:3]
        t = extrinsic[0:3,3]

        width = self.__width
        height = self.__height

        geometries = draw_camera(K, R, t, width, height, scale, color)
        for g in geometries:
            self.add_geometry(g)

    def draw_points3D(self, points3D, color=None):
        geometries = draw_points3D(points3D, color)
        for g in geometries:
            self.add_geometry(g)    

    def save_rgbd_from_3d(self, pointclouds_list=None, meshes_list=None, intrinsic=None, extrinsic=None, save_path=dict(RGB=None, D=None)):
        '''
        pointclouds_list: [pc1, pc2, ..., pcN] in world frame
        meshes_list: [dict1(verts,faces), dict2(verts,faces), ..., dictN(verts,faces)] in world frame
        intrinsic: (3, 3)
        extrinsic: (4, 4)
        image_shape: (width, height)
        '''
        if save_path['D'] is not None and os.path.exists(save_path['D']):
            return 
        assert (pointclouds_list is not None) ^ (meshes_list is not None)
        # self.initialize(self.__width, self.__height, visible=False)
        if pointclouds_list is not None: # add point cloud
            self.add_pointcloud(np.concatenate(pointclouds_list, axis=0))
        
        elif meshes_list is not None: # add mesh
            for mesh_dict in meshes_list:
                self.add_mesh(mesh_dict) 
                   
        # update view
        if len(extrinsic) == 3:
            extrinsic = np.concatenate((extrinsic, np.array([[0, 0, 0, 1]])))
        self.update_view_point(intrinsic, extrinsic)

        # capture depth
        if save_path['D'] is not None:
            # render_depth = vis.capture_depth_float_buffer()   
            self.capture_depth_image(save_path['D'])  # uint16
        
        # capture RGB
        if save_path['RGB'] is not None:
            # render_RGB = vis.capture_screen_float_buffer(show=False)
            self.capture_screen_image(save_path['RGB'])  # uint8
        
        # clear all geometries
        self.__vis.clear_geometries()
        
        # to read the saved depth image file
        # depth = open3d.io.read_image(save_path['D'])
        # plt.imshow(depth)
        # plt.show()
        
    def render_pc_with_camera(self, pointclouds_list=None, meshes_list=None, intrinsic=None, extrinsic=None):
        '''
        pointclouds_list: [pc1, pc2, ..., pcN] in world frame
        meshes_list: [dict1(verts,faces), dict2(verts,faces), ..., dictN(verts,faces)] in world frame
        intrinsic: (3, 3)
        extrinsic: (4, 4)
        image_shape: (width, height)
        '''
        assert (pointclouds_list is not None) ^ (meshes_list is not None)
        # self.initialize(self.__width, self.__height, visible=False)
        if pointclouds_list is not None: # add point cloud
            self.add_pointcloud(np.concatenate(pointclouds_list, axis=0))
        
        elif meshes_list is not None: # add mesh
            for mesh_dict in meshes_list:
                self.add_geometry(mesh_dict) 
                    
        # update view
        if len(extrinsic) == 3:
            extrinsic = np.concatenate((extrinsic, np.array([[0, 0, 0, 1]])))

        # draw camera
        self.draw_camera(intrinsic, extrinsic, scale=0.5, color=[0.8, 0.2, 0.8])
        self.run()


def draw_camera(K, R, t, width, height, scale=1, color=None):
    """ Create axis, plane and pyramid geometries in Open3D format
    :   param K     : calibration matrix (camera intrinsics)
    :   param R     : rotation matrix
    :   param t     : translation
    :   param width : image width
    :   param height: image height
    :   param scale : camera model scale
    :   param color : color of the image plane and pyramid lines
    :   return      : camera model geometries (axis, plane and pyramid)
    """

    # default color
    if color is None:
        color = [0.8, 0.2, 0.8]

    # camera model scale
    s = 1 / scale

    # intrinsics
    Ks = np.array([[K[0, 0] * s,            0, K[0,2]],
                   [          0,  K[1, 1] * s, K[1,2]],
                   [          0,            0, K[2,2]]])
    Kinv = np.linalg.inv(Ks)

    # 4x4 transformation
    T = np.column_stack((R, t))
    T = np.vstack((T, (0, 0, 0, 1)))

    # axis
    axis = create_coordinate_frame(T, scale=scale*0.5)

    # points in pixel
    points_pixel = [
        [0, 0, 0],
        [0, 0, 1],
        [width, 0, 1],
        [0, height, 1],
        [width, height, 1],
    ]

    # pixel to camera coordinate system
    points = [scale * Kinv @ p for p in points_pixel]

    # image plane
    width = abs(points[1][0]) + abs(points[3][0])
    height = abs(points[1][1]) + abs(points[3][1])
    plane = open3d.geometry.TriangleMesh.create_box(width, height, depth=1e-6)
    plane.paint_uniform_color(color)
    plane.transform(T)
    plane.translate(R @ [points[1][0], points[1][1], scale])

    # pyramid
    points_in_world = [(R @ p + t) for p in points]
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
    ]
    colors = [color for i in range(len(lines))]
    line_set = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector(points_in_world),
        lines=open3d.utility.Vector2iVector(lines))
    line_set.colors = open3d.utility.Vector3dVector(colors)

    # return as list in Open3D format
    return [axis, plane, line_set]


def create_coordinate_frame(T, scale=0.25):
    frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
    frame.transform(T)
    return frame


def draw_points3D(points3D, color=None):
    # color: default value
    if color is None:
        color = [0.8, 0.2, 0.8]

    geometries = []
    for pt in points3D:
        sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01,
                                                            resolution=20)
        sphere.translate(pt)
        sphere.paint_uniform_color(np.array(color))
        geometries.append(sphere)

    return geometries




@torch.jit.script
def get_pc_from_world_to_camera(points_world: torch.Tensor, extrinsics: torch.Tensor) -> torch.Tensor:
    """
    Transform points from the world frame to the camera frame using the extrinsics matrix.

    Args:
        points_world: (T, N, 3) tensor of points in the world frame.
        extrinsics: (4, 4) or (T, 4, 4) tensor of world2camera matrices.

    Returns:
        points_camera: (T, N, 3) tensor of points in the camera frame.
    """
    if points_world.ndim == 2:
        points_world = points_world.unsqueeze(0)  # (1, N, 3)

    if extrinsics.ndim == 2:
        extrinsics = extrinsics.unsqueeze(0)  # (1, 4, 4)

    T, N, _ = points_world.shape
    if extrinsics.shape[0] == 1 and T > 1:
        extrinsics = extrinsics.expand(T, -1, -1)

    ones = torch.ones((T, N, 1), dtype=points_world.dtype, device=points_world.device)
    points_world_hom = torch.cat([points_world, ones], dim=-1)  # (T, N, 4)

    points_camera_hom = torch.bmm(points_world_hom, extrinsics.transpose(1, 2))  # (T, N, 4)

    points_camera = points_camera_hom[:, :, :3]  # (T, N, 3)

    return points_camera

@torch.jit.script
def get_pc_from_camera_to_world(points_camera: torch.Tensor, extrinsics: torch.Tensor) -> torch.Tensor:
    """
    Transform points from the camera frame to the world frame using the extrinsics matrix.

    Args:
        points_camera: (T, N, 3) tensor of points in the camera frame.
        extrinsics: (4, 4) or (T, 4, 4) tensor of world2camera matrices.

    Returns:
        points_world: (T, N, 3) tensor of points in the world frame.
    """
    if points_camera.ndim == 2:
        points_camera = points_camera.unsqueeze(0)  # (1, N, 3)

    if extrinsics.ndim == 2:
        extrinsics = extrinsics.unsqueeze(0)  # (1, 4, 4)

    T, N, _ = points_camera.shape

    if extrinsics.size(0) == 1 and T > 1:
        extrinsics = extrinsics.expand(T, -1, -1)

    # Compute inverse of extrinsics (world2cam -> cam2world)
    extrinsics_inv = torch.linalg.inv(extrinsics)  # (T, 4, 4)

    # Convert to homogeneous coordinates
    ones = torch.ones((T, N, 1), dtype=points_camera.dtype, device=points_camera.device)
    points_camera_hom = torch.cat([points_camera, ones], dim=-1)  # (T, N, 4)

    # Apply transformation
    points_world_hom = torch.bmm(points_camera_hom, extrinsics_inv.transpose(1, 2))  # (T, N, 4)

    points_world = points_world_hom[..., :3]  # (T, N, 3)

    return points_world

