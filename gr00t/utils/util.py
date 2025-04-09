from huggingface_hub import snapshot_download
import numpy as np
import torch
import torch.nn.functional as F

def depth_to1(x, max_depth=1500):
    return x.clip(0, max_depth) / max_depth

def resize(x, size=(256,256)):
    '''
    (B, H, W) -> (B, 1, H', W')
    '''
    return F.interpolate(x[:, None], size, mode="bilinear", align_corners=True)

def default_collate_fn(features: list[dict], **kwargs) -> dict:
    batch = {}
    keys = features[0].keys()
    for key in keys:  # ['video.ego_view', 'depth.ego_view', 'state.theta', 'state.beta', 'state.rot', 'state.trans', 'action.theta', 'action.rot', 'action.trans', 'annotation.human.action.task_description', 'video.depthanything']
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

def minmaxscale(x: torch.Tensor) -> torch.Tensor:
    """
    MinMax Scale
    """
    return (x - x.min()) / (x.max() - x.min())