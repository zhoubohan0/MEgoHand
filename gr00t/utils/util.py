from huggingface_hub import snapshot_download
import numpy as np
import torch
import torch.nn.functional as F
import torch.jit as jit


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

def clip_depth(x, max_depth=1.0):  # m
    return x.clip(0, max_depth)

def resize(x, size=(256,256)):
    '''
    (B, H, W) -> (B, 1, H', W')
    '''
    return F.interpolate(x[:, None], size, mode="bilinear", align_corners=True)

def default_collate_fn(features: list[dict], **kwargs) -> dict:
    '''
    list[dict] -> dict[Tensor]
    '''
    batch = {}
    keys = features[0].keys() if kwargs.get('keys', None) is None else kwargs['keys']
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
