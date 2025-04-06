import numpy as np
import os, cv2
import trimesh
import torch
from manopth.manopth.manolayer import ManoLayer
from utils.pc import VisOpen3D
import imageio.v3 as iio
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mano_path = "manopth/mano/models"
mano_layer_dict = dict(
    left=ManoLayer(mano_root=mano_path, use_pca=False, flat_hand_mean=True, ncomps=45, side='left', center_idx=0).to(device),
    right=ManoLayer(mano_root=mano_path, use_pca=False, flat_hand_mean=True, ncomps=45, side='right', center_idx=0).to(device)
)

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def thing2dev(thing, dev):
    if hasattr(thing, "to"):
        thing = thing.to(dev)
        return thing
    if isinstance(thing, list):
        return [thing2dev(ten, dev) for ten in thing]
    if isinstance(thing, tuple):
        return tuple(thing2dev(list(thing), dev))
    if isinstance(thing, dict):
        return {k: thing2dev(v, dev) for k, v in thing.items()}
    if isinstance(thing, torch.Tensor):
        return thing.to(dev)
    return thing

def thing2np(obj):
    if isinstance(obj, dict):
        return {k: thing2np(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [thing2np(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(thing2np(v) for v in obj)
    else:
        return np.array(obj) if not isinstance(obj, np.ndarray) else obj

def pad_tensor_list(v_list: list):
    dev = v_list[0].device
    num_meshes = len(v_list)
    num_dim = 1 if len(v_list[0].shape) == 1 else v_list[0].shape[1]
    v_len_list = []
    for verts in v_list:
        v_len_list.append(verts.shape[0])

    pad_len = max(v_len_list)
    dtype = v_list[0].dtype
    if num_dim == 1:
        padded_tensor = torch.zeros(num_meshes, pad_len, dtype=dtype)
    else:
        padded_tensor = torch.zeros(num_meshes, pad_len, num_dim, dtype=dtype)
    for idx, (verts, v_len) in enumerate(zip(v_list, v_len_list)):
        padded_tensor[idx, :v_len] = verts
    padded_tensor = padded_tensor.to(dev)
    v_len_list = torch.LongTensor(v_len_list).to(dev)
    return padded_tensor, v_len_list

def transform_points(transformation: np.ndarray, points: np.ndarray):
    """
    Transform points using a 4x4 transformation matrix.  
    
    Args:
        transformation: (..., 4, 4) transformation matrix.
        points: (..., X, 3) points to transform.
        
    Returns:
        transformed_points: (..., X, 3) transformed points.    
    """
    leading_shape = points.shape[:-2]
    leading_dim = len(leading_shape)

    res = (
        np.swapaxes(
            np.matmul(
                transformation[..., :3, :3],
                np.swapaxes(points, leading_dim, leading_dim + 1),
            ),
            leading_dim,
            leading_dim + 1,
        )  # [..., X, 3]
        + transformation[..., :3, 3][..., np.newaxis, :]  # [..., 1, 3]
    )
    return res

def get_pc_from_world_to_camera(points_world, extrinsics):
    """
    Transform points from the world frame to the camera frame using the extrinsics matrix.

    Args:
        points_world: (T, N, 3) array of points in the world frame.
        extrinsics: (4, 4) or (T, 4, 4) array of world2camera matrices.

    Returns:
        points_camera: (N, 3) array of points in the camera frame.
    """
    if points_world.ndim == 2:
        points_world = np.expand_dims(points_world, axis=0)  # Add T dimension if missing

    if extrinsics.ndim == 2:
        extrinsics = np.expand_dims(extrinsics, axis=0)  # Add T dimension if missing

    T, N, _ = points_world.shape
    # Broadcast extrinsics to match the number of points if necessary
    if extrinsics.shape[0] == 1 and T > 1:
        extrinsics = np.tile(extrinsics, (T, 1, 1))  # Repeat extrinsics for all T

    # Convert points to homogeneous coordinates (T, N, 4)
    points_world_hom = np.concatenate([points_world, np.ones((T, N, 1))], axis=-1)

    # Apply the extrinsics transformation (T, N, 4) = (T, N, 4) @ (T, 4, 4)^T
    points_camera_hom = points_world_hom @ extrinsics.transpose(0,2,1)

    # Convert back to 3D coordinates by dropping the homogeneous coordinate (T, N, 3)
    points_camera = points_camera_hom[..., :3]

    return points_camera

def get_pc_from_camera_to_world(points_camera, extrinsics):
    """
    Transform points from the camera frame to the world frame using the extrinsics matrix.

    Args:
        points_camera: (T, N, 3) array of points in the camera frame.
        extrinsics: (4, 4) or (T, 4, 4) array of world2camera matrices.

    Returns:
        points_world: (N, 3) array of points in the world frame.
    """
    if points_camera.ndim == 2:
        points_camera = np.expand_dims(points_camera, axis=0)  # Add T dimension if missing

    if extrinsics.ndim == 2:
        extrinsics = np.expand_dims(extrinsics, axis=0)  # Add T dimension if missing

    T, N, _ = points_camera.shape
    # Broadcast extrinsics to match the number of points if necessary
    if extrinsics.shape[0] == 1 and T > 1:
        extrinsics = np.tile(extrinsics, (T, 1, 1))  # Repeat extrinsics for all T

    # Convert points to homogeneous coordinates (T, N, 4)
    points_camera_hom = np.concatenate([points_camera, np.ones((T, N, 1))], axis=-1)

    # Apply the inverse extrinsics transformation (T, N, 4) = (T, N, 4) @ (T, 4, 4)^T
    points_world_hom = points_camera_hom @ np.linalg.inv(extrinsics).transpose(0,2,1)

    # Convert back to 3D coordinates by dropping the homogeneous coordinate (T, N, 3)
    points_world = points_world_hom[..., :3]

    return points_world

def mp42imagefiles(video_path, save_rgb_dir, frame_format="%06d.jpg", way='ffmpeg'):
    """
    Save each frame in a sequence to an image file.

    Args:
        video_path (str): Path to the input MP4 video file.
        save_rgb_dir (str): Directory to save the extracted frames.
        frame_format (str): Format for saving frames (e.g., "%06d.jpg").
    """
    if os.path.exists(save_rgb_dir):
        return
    os.makedirs(save_rgb_dir, exist_ok=True)

    if way == 'cv2':
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_save = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break  
            frame_filename = os.path.join(save_rgb_dir, frame_format % frame_save)
            cv2.imwrite(frame_filename, frame)
            frame_save += 1
        cap.release()
    elif way == 'ffmpeg':
        os.system(f'ffmpeg -i "{video_path}" {os.path.join(save_rgb_dir,"%06d.jpg")}')  # -vf "rotate=PI:bilinear=0" -q:v 2
    # print(f"Saved {frame_save} frames to {save_rgb_dir}")

def ari2imagefiles(video_path, save_depth_dir, frame_format="%06d.png", way='ffmpeg'):
    if os.path.exists(save_depth_dir):
        return
    os.makedirs(save_depth_dir, exist_ok=True)
    if way == 'ffmpeg':
        os.system(f'ffmpeg -i "{video_path}" {os.path.join(save_depth_dir, frame_format)}')
    # depth_png = cv2.imread("depth_frame.png", cv2.IMREAD_ANYDEPTH)  # Critical for 16-bit

def compute_extrinsics(points_3d_world, points_3d_camera, K=None, method="Direct"):
    """
    Compute the extrinsic matrix [R | t] using Direct, DLT, EPnP, or PnP.

    Args:
        points_3d_world (np.ndarray): (N, 3) array of 3D points in the world frame.
        points_3d_camera (np.ndarray): (N, 3) array of corresponding 3D points in the camera frame.
        K (np.ndarray): (3, 3) camera intrinsic matrix (required for DLT, EPnP, PnP).
        method (str): Method to use for computation. Options: "Direct", "DLT", "EPnP", "PnP". Default is "Direct".

    Returns:
        np.ndarray: (3, 4) extrinsic matrix [R | t].
    """
    # Ensure inputs are numpy arrays
    points_3d_world = np.array(points_3d_world, dtype=np.float32)
    points_3d_camera = np.array(points_3d_camera, dtype=np.float32)

    if method == "Direct":
        # Direct rigid transformation (Procrustes analysis)
        # Center the points
        centroid_world = np.mean(points_3d_world, axis=0)
        centroid_camera = np.mean(points_3d_camera, axis=0)
        centered_world = points_3d_world - centroid_world
        centered_camera = points_3d_camera - centroid_camera

        # Compute the covariance matrix
        H = centered_world.T @ centered_camera

        # Perform SVD to find the rotation matrix
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute the translation vector
        t = centroid_camera - R @ centroid_world

        # Combine R and t into extrinsic matrix [R | t]
        extrinsic_matrix = np.hstack((R, t.reshape(3, 1)))

        return extrinsic_matrix
    elif method in ["DLT", "EPnP", "PnP"]:
        if K is None:
            raise ValueError("Intrinsic matrix K is required for DLT, EPnP, or PnP.")
        K = np.array(K, dtype=np.float32)

        # Project 3D camera points to 2D image plane using intrinsic matrix
        points_2d_camera, _ = cv2.projectPoints(points_3d_camera, np.zeros(3), np.zeros(3), K, None)
        points_2d_camera = points_2d_camera.reshape(-1, 2)

        if method == "DLT":
            # Direct Linear Transform (DLT)
            extrinsic_matrix = _dlt(points_3d_world, points_2d_camera, K)
        elif method == "EPnP":
            # Efficient PnP (EPnP)
            success, rvec, tvec = cv2.solvePnP(points_3d_world, points_2d_camera, K, None, flags=cv2.SOLVEPNP_EPNP)
            if not success:
                raise ValueError("Failed to solve EPnP. Check input data.")
            R, _ = cv2.Rodrigues(rvec)
            extrinsic_matrix = np.hstack((R, tvec))
        elif method == "PnP":
            # Standard PnP
            success, rvec, tvec = cv2.solvePnP(points_3d_world, points_2d_camera, K, None)
            if not success:
                raise ValueError("Failed to solve PnP. Check input data.")
            R, _ = cv2.Rodrigues(rvec)
            extrinsic_matrix = np.hstack((R, tvec))
    else:
        raise ValueError("Invalid method. Choose 'Direct', 'DLT', 'EPnP', or 'PnP'.")

    return extrinsic_matrix

def _dlt(points_3d_world, points_2d_camera, K):
    """
    Direct Linear Transform (DLT) for computing the extrinsic matrix.

    Args:
        points_3d_world (np.ndarray): (N, 3) array of 3D points in the world frame.
        points_2d_camera (np.ndarray): (N, 2) array of corresponding 2D points in the camera frame.
        K (np.ndarray): (3, 3) camera intrinsic matrix.

    Returns:
        np.ndarray: (3, 4) extrinsic matrix [R | t].
    """
    # Normalize 2D points using the intrinsic matrix
    K_inv = np.linalg.inv(K)
    points_2d_homogeneous = np.hstack((points_2d_camera, np.ones((points_2d_camera.shape[0], 1))))
    points_2d_normalized = (K_inv @ points_2d_homogeneous.T).T

    # Build the A matrix for DLT
    A = []
    for i in range(points_3d_world.shape[0]):
        X, Y, Z = points_3d_world[i]
        u, v = points_2d_normalized[i]
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z, -v])
    A = np.array(A)

    # Solve for the projection matrix P using SVD
    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)

    # Extract R and t from P
    R = P[:, :3]
    t = P[:, 3]

    # Ensure R is a valid rotation matrix by performing SVD and reconstructing
    U, S, Vt = np.linalg.svd(R)
    R = U @ Vt

    # Handle reflection case
    if np.linalg.det(R) < 0:
        R = -R
        t = -t

    # Combine R and t into extrinsic matrix [R | t]
    extrinsic_matrix = np.hstack((R, t.reshape(3, 1)))

    return extrinsic_matrix

class Mesh(trimesh.Trimesh):
    def __init__(
        self,
        filename=None,
        v=None,
        f=None,
        vc=None,
        fc=None,
        process=False,
        visual=None,
        **kwargs
    ):
        if filename is not None:
            mesh = trimesh.load(filename, process=process)
            v = mesh.vertices
            f = mesh.faces
            visual = mesh.visual

        super(Mesh, self).__init__(
            vertices=v, faces=f, visual=visual, process=process, **kwargs
        )

        self.v = self.vertices
        self.f = self.faces
        assert self.v is self.vertices
        assert self.f is self.faces

        if vc is not None:
            self.set_vc(vc)
            self.vc = self.visual.vertex_colors
            assert self.vc is self.visual.vertex_colors
        if fc is not None:
            self.set_fc(fc)
            self.fc = self.visual.face_colors
            assert self.fc is self.visual.face_colors

    def rot_verts(self, vertices, rxyz):
        return np.array(vertices * rxyz.T)

    def colors_like(self, color, array, ids):
        color = np.array(color)

        if color.max() <= 1.0:
            color = color * 255
        color = color.astype(np.int8)

        n_color = color.shape[0]
        n_ids = ids.shape[0]

        new_color = np.array(array)
        if n_color <= 4:
            new_color[ids, :n_color] = np.repeat(color[np.newaxis], n_ids, axis=0)
        else:
            new_color[ids, :] = color

        return new_color

    def set_vc(self, vc, vertex_ids=None):
        all_ids = np.arange(self.vertices.shape[0])
        if vertex_ids is None:
            vertex_ids = all_ids

        vertex_ids = all_ids[vertex_ids]
        new_vc = self.colors_like(vc, self.visual.vertex_colors, vertex_ids)
        self.visual.vertex_colors[:] = new_vc

    def set_fc(self, fc, face_ids=None):
        if face_ids is None:
            face_ids = np.arange(self.faces.shape[0])

        new_fc = self.colors_like(fc, self.visual.face_colors, face_ids)
        self.visual.face_colors[:] = new_fc

    @staticmethod
    def cat(meshes):
        return trimesh.util.concatenate(meshes)


def mano_forward(mano_theta, mano_beta, wrist_rot, wrist_trans, side):
    """
    Args:
        mano_theta: (N, 15*3) torch tensor
        mano_beta: (N, 10) or (10) torch tensor
        wrist_rot: (N, 3) torch tensor
        wrist_trans: (N, 3) torch tensor
        side: 'l' or 'r' for left or right hand
        
    Returns:
        hand_vertices: (N, 778, 3) torch tensor
        hand_keypoints: (N, 21, 3) torch tensor
        hand_faces: (1538, 3) torch tensor
    """

    theta = mano_theta.to(device)  # (N_frame, 15 * 3)
    beta = mano_beta.to(device) if len(mano_beta) == len(mano_theta) else mano_beta.view(-1,10).repeat(theta.shape[0], 1)  # (N_frame, 10)
    rot = wrist_rot.to(device)  # (N_frame, 3)
    trans = wrist_trans.to(device)  # (N_frame, 3)

    mano_layer = mano_layer_dict['left'] if 'l' in side else mano_layer_dict['right']
    hand_verts_pred, hand_joints_pred, full_pose = mano_layer(torch.cat([rot, theta], dim=-1), beta)  # full_pose==theta
    hand_verts_pred = hand_verts_pred / 1000.0
    hand_joints_pred = hand_joints_pred / 1000.0
    hand_verts_pred += trans.unsqueeze(1)  # (N_frame, 778, 3)
    hand_joints_pred += trans.unsqueeze(1)  # (N_frame, 21, 3)
    
    return hand_verts_pred, hand_joints_pred, mano_layer.th_faces.reshape(-1, 3), # theta.detach().cpu().numpy(), trans.detach().cpu().numpy()

def vis_mano(theta, beta, rot, trans, side, ref_keypoints=None, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    '''
    theta: (1, 45) cuda tensor
    beta: (1, 10) cuda tensor
    rot: (1,3) cuda tensor
    trans: (1,3) cuda tensor
    side: 'left' or 'right'
    keypoints: (1, 21, 3) cuda tensor
    '''
    # visualizer
    vis = VisOpen3D()
    vertices, keypoints, faces = mano_forward(theta, beta, rot, trans, side, device)
    vis.add_mesh(dict(verts=vertices[0].cpu().numpy(), faces=faces.cpu().numpy()))
    vis.draw_points3D(keypoints[0].cpu().numpy())
    if ref_keypoints is not None:
        vis.draw_points3D(ref_keypoints[0].cpu().numpy(), color=(0.0, 1.0, 0.0))
    vis.run()
    vis.destroy_window()
    return keypoints[0].cpu().numpy()

def normalize(x):
    mean = x.mean(0)[None,:]
    std = x.std(0)[None,:]
    x = (x - mean) / std
    return x, mean, std

def denormalize(x, mean, std):
    x = x * std + mean
    return x

def read_single_image(image_path):
    '''
    read RGB or depth png 
    '''
    return iio.imread(image_path)

def glance_dict(dic):
    for key, value in dic.items():
        if isinstance(value, np.ndarray):
            print(key, value.shape)
        elif isinstance(value, dict):
            print(key)
            glance_dict(value)
        elif isinstance(value, torch.Tensor):
            print(key, value.shape)
        else:
            print(key, value)

def find_lower_left(ls, x):
    return np.searchsorted(ls, x + 1) - 1