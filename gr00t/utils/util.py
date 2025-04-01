from huggingface_hub import snapshot_download
import numpy as np
import torch

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