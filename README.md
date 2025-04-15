> To modify action horizon
> 
> gr00t/experiment/data_config.py L693
> 
> model_config/config.json action_horizon

Prepare Dataset
```bash
python -m mhmp_vis.dataset_loader.dataset_fpha
```

Calculate Statistics
```bash
bash bash/statistics.sh -dataset_list taco
```

Training
```bash
# MHMP
bash bash/train.sh -dataset_list taco -depthanything_path "" -incorporate_depth 0 -exp_name RGB 
# MHMP+

```