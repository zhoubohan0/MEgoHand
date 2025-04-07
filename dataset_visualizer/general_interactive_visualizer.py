import argparse
import os.path as osp
import numpy as np
import torch
import smplx
from PIL import Image
import open3d as o3d
import importlib
from .viewer import ARCTICViewer, ViewerData, construct_viewer_meshes


class GeneralInteractiveVisualizer(ARCTICViewer):
    # constant
    MODEL_DIR=osp.join(osp.dirname(osp.abspath(__file__)), '..', "manopth/mano/models")
    SMPLX_MODEL_P = {
        "male": osp.join(osp.dirname(osp.abspath(__file__)), '..', "manopth/smplx/SMPLX_MALE.npz"),
        "female": osp.join(osp.dirname(osp.abspath(__file__)), '..', "manopth/smplx/SMPLX_FEMALE.npz"),
        "neutral": osp.join(osp.dirname(osp.abspath(__file__)), '..', "manopth/smplx/SMPLX_NEUTRAL.npz"),
    }
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __init__(
        self,
        interactive=True,
        render_types=["rgb", "depth", "mask"],
        size=(2024, 2024),
    ):
        super().__init__(render_types, interactive, size)

    
    def construct_meshes_in_camera_frame(
        self,
        image_filenames,
        cam_left_hand_verts,
        cam_right_hand_verts,
        cam_smplx_verts,
        cam_object_dict_list,
        intrinsic_matrix, 
        extrinsic_matrix=np.array([[ 1.,  0.,  0.,  0.],
                                [-0., -1., -0.,  0.],
                                [-0., -0., -1.,  0.]]),  # reflection for better view
        image_size=None,
        left_hand_faces=None,
        right_hand_faces=None,
        **kwargs  # filter unused args
    ):
        '''
        - image_filenames: (T), Optional can be None
        - image_size: (2), default (800, 640)

        - cam_left_hand_verts: (T, 778, 3) in camera frame, Optional can be None
        - cam_right_hand_verts: (T, 778, 3) in camera frame, Optional can be None
        - cam_smplx_verts: (T, 10475, 3) in camera frame, Optional can be None
        
        - cam_object_dict_list is a list of object_info_dict, Optional can be None
            1. object_model_file: str, cam_object_transforms: (T, 4, 4)
            2. object_model_file: str, cam_object_verts: (T, M, 3)
            3. cam_object_verts: (T, M, 3) in camera frame, object_faces: (N, 3)
        
        - intrinsic_matrix: (3, 3) or (T, 3, 3)
        - extrinsic_matrix: (3, 4) or (T, 3, 4)
        '''
        if image_size is None:
            image_size = Image.open(image_filenames[0]).size if image_filenames is not None and len(image_filenames) > 0 else (800, 640)

        ### construct meshes
        vis_dict = {}
        if cam_left_hand_verts is not None:  # left hand 
            self.left_hand = smplx.MANO(self.MODEL_DIR, create_transl=True, use_pca=False, flat_hand_mean=False, is_rhand=False).to(self.dev)
            vis_dict["left"] = {
                "v3d": cam_left_hand_verts,
                "f3d": self.left_hand.faces if left_hand_faces is None else left_hand_faces,
                "vc": None,
                "name": "left",
                "color": "white",
            }
        if cam_right_hand_verts is not None:  # right hand
            self.right_hand = smplx.MANO(self.MODEL_DIR, create_transl=True, use_pca=False, flat_hand_mean=False, is_rhand=True).to(self.dev)
            vis_dict["right"] = {
                "v3d": cam_right_hand_verts,
                "f3d": self.right_hand.faces if right_hand_faces is None else right_hand_faces,
                "vc": None,
                "name": "right",
                "color": "red",
            }
        if cam_smplx_verts is not None:  # smplx
            self.smplx_model = smplx.create(
                model_path=self.SMPLX_MODEL_P["neutral"],
                model_type="smplx",
                gender="neutral",
                num_pca_comps=45,
                v_template=None,
                flat_hand_mean=True,
                use_pca=False,
                batch_size=1,
            ).to(self.dev)
            vis_dict["smplx"] = {
                "v3d": cam_smplx_verts,
                "f3d": self.smplx_model.faces,
                "vc": None,
                "name": "smplx",
                "color": "rice",
            }
        if cam_object_dict_list is not None: 
            for i_obj, object_info_dict in enumerate(cam_object_dict_list, 1):
                '''
                [cam_object_verts, object_faces]
                '''
                cam_object_verts = object_info_dict['cam_object_verts']
                obj_faces = object_info_dict['object_faces']
                vis_dict[f"object{i_obj}"] = {
                    "v3d": cam_object_verts,
                    "f3d": obj_faces,
                    "vc": None,
                    "name": "object",
                    "color": "light-blue",
                }
                '''
                [visualize] 

                from scipy.spatial import procrustes
                p1 = obj_mesh.vertices
                p2 = data["cam_coord"]["verts.object"][0, view_idx]
                _, p1_aligned, disparity = procrustes(p1, p2)
                _, p2_aligned, disparity = procrustes(p2, p1)
                o3d.visualization.draw_geometries([o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(p1_aligned)), o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(p2_aligned))])
                '''

        meshes = construct_viewer_meshes(vis_dict, draw_edges=False, flat_shading=False)
        viewerdata = ViewerData(Rt=extrinsic_matrix, K=intrinsic_matrix, cols=image_size[0], rows=image_size[1], imgnames=image_filenames)
        batch = (meshes, viewerdata)
        self.check_format(batch)
        return batch


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

    # online render
    viewer = GeneralInteractiveVisualizer()    
    batch = viewer.construct_meshes_in_camera_frame(**data_sample)
    viewer.render_seq(batch)
