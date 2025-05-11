from huggingface_hub import snapshot_download
import numpy as np
import torch
import torch.nn.functional as F
import torch.jit as jit
import os, cv2
import matplotlib.pyplot as plt
from contextlib import contextmanager
import time

@contextmanager
def timer(name="Code block"):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{name} took {end - start:.4f} seconds")
    
class RunningMeanStdMinMax(jit.ScriptModule):
    def __init__(self, shape=(), device='cuda'):
        super(RunningMeanStdMinMax, self).__init__()
        self.register_buffer('mean', torch.zeros(shape, dtype=torch.float32).to(device))  # μ
        self.register_buffer('var', torch.ones(shape, dtype=torch.float32).to(device))    # σ^2, biased
        self.register_buffer('min', torch.inf * torch.ones(shape, dtype=torch.float32).to(device))    # min
        self.register_buffer('max', -torch.inf * torch.ones(shape, dtype=torch.float32).to(device))    # max
        self.register_buffer('n', torch.tensor(0, dtype=torch.float32).to(device))        # n

    @jit.script_method
    def update(self, x: torch.Tensor):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_n = x.shape[0]

        delta = batch_mean - self.mean
        total_n = self.n + batch_n

        self.mean = self.mean + delta * batch_n / total_n
        self.var = (self.var * self.n + batch_var * batch_n + torch.square(delta) * self.n * batch_n / total_n) / total_n
        self.n = total_n

        self.min = torch.minimum(self.min, torch.min(x, dim=0).values)
        self.max = torch.maximum(self.max, torch.max(x, dim=0).values)


    @jit.script_method
    def get(self,):
        return {
            'mean':self.mean, 
            'std':self.var.sqrt(),
            'min':self.min,
            'max':self.max,
            # ignore
            'q01':self.min + (self.max - self.min) * 0.01,
            'q99':self.min + (self.max - self.min) * 0.99,
        }


def resize(x, size=(256,256)):
    '''
    (B, H, W) -> (B, 1, H', W')
    '''
    if len(x.shape) == 3:
        x = x[:, None]  # (B, H, W) -> (B, 1, H, W)
    return F.interpolate(x, size, mode="bilinear", align_corners=True)

def default_collate_fn(features: list[dict], **kwargs) -> dict:
    '''
    list[dict] -> dict[Tensor]
    '''
    batch = {}
    keys = features[0].keys() if kwargs.get('keys', None) is None else kwargs['keys']
    for key in keys:  # ['video.ego_view', 'depth.ego_view', 'state.theta', 'state.beta', 'state.rot', 'state.trans', 'action.theta', 'action.rot', 'action.trans', 'annotation.human.action.task_description', 'video.depthmodel']
        values = [elem[key] for elem in features]
        # stack according to the type
        if isinstance(values[0], torch.Tensor):
            batch[key] = torch.stack(values, dim=0)
        elif isinstance(values[0], np.ndarray):
            batch[key] = np.stack(values, axis=0)
        elif isinstance(values[0], list):
            batch[key] = [elem for elem in values]
        else:
            batch[key] = values
    return batch


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

def download_from_hg(repo_id: str, repo_type: str) -> str:
    """
    Download the model/dataset from the hugging face hub.
    return the path to the downloaded
    """
    if '/' not in repo_id:
        return repo_id
    repo_path = snapshot_download(repo_id, repo_type=repo_type)
    return repo_path

def mse(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Mean Squared Error
    """
    return F.mse_loss(x1, x2, reduction='mean')

def l1(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    L1 Loss
    """
    return F.l1_loss(x1, x2, reduction='mean')

def smooth_l1(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Smooth L1 Loss
    """
    return F.smooth_l1_loss(x1, x2, reduction='mean')  

# depth related
def clip_depth(x, max_depth=2.0):  # m
    return x.clip(0, max_depth)

def minmaxscale(x: torch.Tensor) -> torch.Tensor:
    """
    MinMax Scale
    """
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return 2 * (x - x_min) / (x_max - x_min) - 1

# depth visualization
def depth2color(depth: np.ndarray, cmap='Spectral_r') -> np.ndarray:
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    cmap = plt.get_cmap(cmap)
    depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    return depth



def visualize_RGBD(
        depth: np.ndarray, depth_gt: np.ndarray, rgb: np.ndarray=None, 
        save_file='', tag='Pred', 
        way='matplotlib',  # 'cv2', 'matplotlib', 'plotly' 
        cmap='Spectral_r'  # 'magma', 'magma_r',
    ):
    """
    Concatenate depth, depth_gt, and rgb for visualization.
    depth: metric depth, \in [0,2]
    depth_gt: metric depth, \in [0,2]
    rgb: \in [0,255]
    """
    RGB_tag = 'RGB'
    depth_gt_tag = 'GT'
    if save_file:
        os.makedirs(os.path.dirname(save_file), exist_ok=True)

    if way == 'cv2':
        depth = depth2color(depth, cmap)
        depth_gt = depth2color(depth_gt, cmap)

        H, W = depth.shape[:2]
        split_region = np.ones((H, 50, 3), dtype=np.uint8) * 255
        scale = max(H, W) / 500  # Auto-scale factor
        depth = cv2.putText(depth, tag, (10, int(30*scale)), cv2.FONT_HERSHEY_DUPLEX, scale, (255,255,255), max(1, int(scale)))
        depth_gt = cv2.putText(depth_gt, depth_gt_tag, (10, int(30*scale)), cv2.FONT_HERSHEY_DUPLEX, scale, (255,255,255), max(1, int(scale)))
        combined_result = cv2.hconcat([depth, split_region, depth_gt])
        if rgb is not None:
            rgb = cv2.putText(rgb, RGB_tag, (10, int(30*scale)), cv2.FONT_HERSHEY_DUPLEX, scale, (255,255,255), max(1, int(scale)))
            combined_result = cv2.hconcat([rgb, split_region, combined_result])
        cv2.imwrite(save_file, combined_result)

    elif way == 'matplotlib':
        grid_size = 10
        num_plot = 2 if rgb is None else 3
        fig, ax = plt.subplots(1, num_plot, figsize=(grid_size * num_plot, grid_size))
        # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB) if rgb is not None else None
        ax[0].imshow(rgb)
        ax[0].set_title(RGB_tag)
        ax[0].axis('off')
        ax[1].imshow(depth, cmap=cmap)
        ax[1].set_title(tag)
        ax[1].axis('off')
        plt.colorbar(ax[1].imshow(depth, cmap=cmap), ax=ax[1], shrink=0.5)
        ax[2].imshow(depth_gt, cmap=cmap)
        ax[2].set_title(depth_gt_tag)
        ax[2].axis('off')
        plt.colorbar(ax[2].imshow(depth_gt, cmap=cmap), ax=ax[2], shrink=0.5)
        plt.savefig(save_file, bbox_inches='tight')
        plt.close()

    elif way == 'plotly':  # online, 3D
        import plotly.graph_objects as go
        fig = go.Figure(
            data=go.Surface(
                z=depth,  # Depth values as the height
                colorscale='Viridis',  # Choose a color scale like 'Viridis', 'Cividis', 'Jet', etc.
                colorbar=dict(title="Depth"),  # Add a colorbar with a label
            )
        )
        fig.update_layout(
            title=tag,
            scene=dict(
                xaxis_title="X-axis (pixels)",
                yaxis_title="Y-axis (pixels)",
                zaxis_title="Depth",
            ),
            margin=dict(l=0, r=0, t=50, b=0),  # Reduce margins for better fit
        )
        fig.show()
    else:
        raise NotImplementedError(f"Visualization method {way} is not implemented.")
