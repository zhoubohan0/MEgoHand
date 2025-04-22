# MHMP: Multimodal Hand Motion Pretraining
## Dataset
### Prepare Dataset
```bash
export dataset=taco
python -m mhmp_vis.dataset_loader.dataset_${dataset}
```

### Visualize Single Task in a Dataset
```bash
cd mhmp_vis
export dataset=fpha
export task="Subject_1/charge_cell_phone/1/106"

export dataset=hoi4d
export task="ZY20210800001/H1/C1/N19/S100/s02/T1/"

export dataset=taco
export task="(dust, roller, pan)/20230927_032"

...

python -m pdb -m dataset_visualizer.general_offline_visualizer -d ${dataset} -t ${task}
```

### Calculate Statistics for CascadedDataset
for each dataset:
```bash
for dataset in arctic fpha h2o hoi4d hot3d taco oakink2; do
    bash bash/statistics.sh -dataset_list "${dataset}"
done
python model_config/get_metadata.py  # final pooling
```

### Use CascadedDataset
Run: 
```bash
python -m scripts.dataloading_example
```
Minimal example codes
```python
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from mhmp_vis.dataset_loader import load_cascaded_dataset

data_config = DATA_CONFIG_MAP['hand_motion']
data_storage_kwargs = dict(window_size=16, side='right', is_render_depth=False)
# split whole train & evaluation dataset on server 
train_dataset, eval_dataset = load_cascaded_dataset(
    dataset_path='/mnt/vepfs/zbh/MHMP/datasets/',
    dataset_list='arctic,fpha,h2o,hoi4d,hot3d,taco'.split(','),
    split_portion=0.95,
    data_config=data_config,
    data_storage_kwargs=data_storage_kwargs,
    test=False,
    debug=False,
)
# load partial testing dataset locally
test_dataset = load_cascaded_dataset(
    dataset_path='/media/ps/zhan/DATASET/',
    dataset_list='fpha'.split(','),
    split_portion=0.99,
    data_config=data_config,
    data_storage_kwargs=data_storage_kwargs,
    test=True,
    debug=False,
)
```
> To modify action horizon
> 
> gr00t/experiment/data_config.py L693
> 
> model_config/config.json action_horizon

## Model
## Training
```bash
# MHMP-lora (unused)
# bash bash/train.sh -dataset_list taco -depthanything_path "" -incorporate_depth 0 -exp_name RGB_lora -lora_rank 32 -tune_llm 1 -device "4,5,6,7"  # -resume "logs/RGB_lora/checkpoint-v0" -debug 1

# MHMP
# on server:
bash bash/train.sh -batch_size 104 -depthanything_path "" -incorporate_depth 0 -exp_name RGB_notunellm_6 -lora_rank 0 -tune_llm 0 -device "0,1,2,3,4,5,6,7" -dataset_list arctic,fpha,h2o,hoi4d,hot3d,taco
# locally:
bash bash/train.sh -dataset_path /media/ps/zhan/DATASET/ -depthanything_path "" -incorporate_depth 0 -exp_name ZY_RGB_fpha -lora_rank 0 -tune_llm 0 -tune_visual 0 -device 0 -dataset_list fpha -debug 1 -batch_size 1 
# test and visualize
bash bash/test.sh -dataset_list taco -split_portion 0.95 -pretrain_ckpt /mnt/vepfs/zbh/MHMP/logs/RGB_notunellm/checkpoint-10000

# MHMP+

```
Visualize logs:
```bash
export exp_name=""
exp_dir=logs/$exp_name/runs/
tensorboard --logdir $exp_dir/*
```
The checkpoint will be saved in $exp_dir/*
