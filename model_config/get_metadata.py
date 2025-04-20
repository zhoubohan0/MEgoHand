import os
import json
import glob
import torch
from collections import defaultdict

# Setup
metadata_files = glob.glob('model_config/metadata/*_relative.json')
output_path = 'model_config/metadata.json'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pooling helpers
def pool_leaf_stats(stats_list):
    total_n = sum(d['n'] for d in stats_list)
    mean_accum = sum(d['mean'] * d['n'] for d in stats_list)
    total_mean = mean_accum / total_n

    var_within = sum((d['std']**2) * (d['n'] - 1) for d in stats_list)
    var_between = sum(d['n'] * (d['mean'] - total_mean)**2 for d in stats_list)
    total_var = (var_within + var_between) / (total_n - 1)
    total_std = total_var.sqrt()

    total_min = torch.min(torch.stack([d['min'] for d in stats_list]), dim=0).values
    total_max = torch.max(torch.stack([d['max'] for d in stats_list]), dim=0).values

    return {
        'mean': total_mean.tolist(),
        'std': total_std.tolist(),
        'min': total_min.tolist(),
        'max': total_max.tolist(),
        'q01': total_min.tolist(),
        'q99': total_max.tolist(),
    }

def pool_nested_stats(dicts, n_list):
    flat_data = defaultdict(list)

    for meta_dict, n in zip(dicts, n_list):
        for top_k in ['state', 'action']:
            for key, stat in meta_dict[top_k].items():
                flat_data[f'{top_k}/{key}'].append({
                    'mean': torch.tensor(stat['mean'], device=device),
                    'std': torch.tensor(stat['std'], device=device),
                    'min': torch.tensor(stat['min'], device=device),
                    'max': torch.tensor(stat['max'], device=device),
                    'n': n
                })

    pooled = {'state': {}, 'action': {}}
    for flat_key, stat_list in flat_data.items():
        section, key = flat_key.split('/')
        pooled[section][key] = pool_leaf_stats(stat_list)

    return pooled

# Load and process each metadata file
all_dicts = []
n_list = []
total_trajectories = 0
total_trajectory_length = 0

for file in metadata_files:
    with open(file, 'r', encoding='utf-8') as f:
        meta = json.load(f)
        all_dicts.append(meta["statistics"])

        n = meta["statistics"]['total_trajectory_length']
        n_list.append(n)
        total_trajectories += meta["statistics"]['num_trajectories']
        total_trajectory_length += n


# Pool statistics
pooled_result = dict(
    statistics={
        'total_trajectory_length': total_trajectory_length,
        'num_trajectories': total_trajectories,
        **pool_nested_stats(all_dicts, n_list)
    },
    modalities=meta['modalities'],
    embodiment_tag=meta["embodiment_tag"]
)

# Save to final output path
with open(output_path, 'w') as f:
    json.dump(pooled_result, f, indent=4)
