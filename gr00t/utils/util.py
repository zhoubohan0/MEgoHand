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
        inputs: dict[str, np.ndarray | torch.Tensor],
        save_file: str = '', 
        tag: str = 'Pred', 
        way: str = 'matplotlib',  # 'cv2', 'matplotlib', 'plotly' 
        cmap: str = 'Spectral_r',  # 'magma', 'magma_r',
        mask: np.ndarray | torch.Tensor | None = None,
    ):
    """
    Concatenate depth, depth_gt, and rgb for visualization.
    
    Args:
        inputs (dict): Dictionary containing visualization data with possible keys:
            - depth: metric depth map, numpy array, range [0,2] meters
            - depth_gt: ground truth depth map, numpy array, range [0,2] meters
            - rgb: RGB image, numpy array, range [0,255]
        save_file (str): Path to save the visualization result
        tag (str): Tag for the predicted depth visualization
        way (str): Visualization method, one of ['cv2', 'matplotlib', 'plotly']
        cmap (str): Colormap for depth visualization
        mask (np.ndarray | torch.Tensor | None): Optional mask for highlighting specific regions
        
    Returns:
        None: Saves visualization to file if save_file is provided
    """
    RGB_tag = 'RGB'
    depth_gt_tag = 'GT'
    if save_file:
        os.makedirs(os.path.dirname(save_file), exist_ok=True)

    # Convert torch tensors to numpy arrays if needed
    depth = inputs.get('depth', None)
    depth_gt = inputs.get('depth_gt', None) 
    rgb = inputs.get('rgb', None)

    if rgb is not None and rgb.shape[:2] != depth.shape[:2]:
        rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]))
    if depth_gt is not None and depth_gt.shape[:2] != depth.shape:
        depth_gt = resize(torch.Tensor(depth_gt)[None, ...], depth.shape[:2]).squeeze().numpy()
    # if depth is not None and isinstance(depth, torch.Tensor):
    #     depth = depth.detach().cpu().numpy()
    # if depth_gt is not None and isinstance(depth_gt, torch.Tensor):
    #     depth_gt = depth_gt.detach().cpu().numpy()
    # if rgb is not None and isinstance(rgb, torch.Tensor):
    #     rgb = rgb.detach().cpu().numpy()
    # if mask is not None and isinstance(mask, torch.Tensor):
    #     mask = mask.detach().cpu().numpy()

    # Validate input ranges
    # if depth is not None:
    #     assert depth.min() >= 0 and depth.max() <= 2, f"Depth range should be [0,2], got [{depth.min()}, {depth.max()}]"
    # if depth_gt is not None:
    #     assert depth_gt.min() >= 0 and depth_gt.max() <= 2, f"Depth GT range should be [0,2], got [{depth_gt.min()}, {depth_gt.max()}]"
    # if rgb is not None:
    #     assert rgb.min() >= 0 and rgb.max() <= 255, f"RGB range should be [0,255], got [{rgb.min()}, {rgb.max()}]"

    if way == 'cv2':
        # 只处理非None的输入
        vis_images = []
        if rgb is not None:
            rgb_vis = rgb.copy()
            H, W = rgb_vis.shape[:2]
            scale = max(H, W) / 200  
            rgb_vis = cv2.putText(rgb_vis, RGB_tag, (10, int(50*scale)), cv2.FONT_HERSHEY_DUPLEX, scale, (255,255,255), max(2, int(scale)))
            vis_images.append(rgb_vis)
            
        if depth is not None:
            depth_vis = depth2color(depth, cmap)
            H, W = depth_vis.shape[:2]
            scale = max(H, W) / 200  
            depth_vis = cv2.putText(depth_vis, tag, (10, int(50*scale)), cv2.FONT_HERSHEY_DUPLEX, scale, (255,255,255), max(2, int(scale)))
            vis_images.append(depth_vis)
            
        if depth_gt is not None:
            depth_gt_vis = depth2color(depth_gt, cmap)
            H, W = depth_gt_vis.shape[:2]
            scale = max(H, W) / 200  
            depth_gt_vis = cv2.putText(depth_gt_vis, depth_gt_tag, (10, int(50*scale)), cv2.FONT_HERSHEY_DUPLEX, scale, (255,255,255), max(2, int(scale)))
            vis_images.append(depth_gt_vis)
            
        if len(vis_images) > 0:
            H = vis_images[0].shape[0]
            split_region = np.ones((H, 50, 3), dtype=np.uint8) * 255
            
            combined_result = vis_images[0]
            for img in vis_images[1:]:
                combined_result = cv2.hconcat([combined_result, split_region, img])
                
            if save_file:
                cv2.imwrite(save_file, combined_result)

    elif way == 'matplotlib':
        num_plot = sum(x is not None for x in [rgb, depth, depth_gt])
        if num_plot == 0:
            return
            
        grid_size = 10
        fig, axes = plt.subplots(1, num_plot, figsize=(grid_size * num_plot, grid_size))
        if num_plot == 1:
            axes = [axes]
            
        plot_idx = 0
        if rgb is not None:
            axes[plot_idx].imshow(rgb)
            axes[plot_idx].set_title(RGB_tag, fontsize=24, pad=20) 
            axes[plot_idx].axis('off')
            plot_idx += 1
            
        if depth is not None:
            im = axes[plot_idx].imshow(depth, cmap=cmap)
            axes[plot_idx].set_title(tag, fontsize=24, pad=20) 
            axes[plot_idx].axis('off')
            plt.colorbar(im, ax=axes[plot_idx], shrink=0.5)
            plot_idx += 1
            
        if depth_gt is not None:
            im = axes[plot_idx].imshow(depth_gt, cmap=cmap)
            axes[plot_idx].set_title(depth_gt_tag, fontsize=24, pad=20) 
            axes[plot_idx].axis('off')
            plt.colorbar(im, ax=axes[plot_idx], shrink=0.5)
            
        if save_file:
            plt.savefig(save_file, bbox_inches='tight')
        plt.close()

    elif way == 'plotly':  # online, 3D
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        num_plots = sum(1 for x in [depth, depth_gt] if x is not None)
        if num_plots == 0:
            return
            
        fig = make_subplots(rows=1, cols=num_plots,
                           specs=[[{"type": "surface"}] * num_plots])
        
        plot_idx = 1
        if depth is not None:
            fig.add_trace(
                go.Surface(
                    z=depth,
                    colorscale='Viridis',
                    colorbar=dict(title="Depth"),
                ),
                row=1, col=plot_idx
            )
            plot_idx += 1
            
        if depth_gt is not None:
            fig.add_trace(
                go.Surface(
                    z=depth_gt,
                    colorscale='Viridis',
                    colorbar=dict(title="Depth GT"),
                ),
                row=1, col=plot_idx
            )
            
        fig.update_layout(
            title=tag,
            scene=dict(
                xaxis_title="X-axis (pixels)",
                yaxis_title="Y-axis (pixels)",
                zaxis_title="Depth",
            ),
            margin=dict(l=0, r=0, t=50, b=0),
        )
        fig.show()
    else:
        raise NotImplementedError(f"Visualization method {way} is not implemented.")
