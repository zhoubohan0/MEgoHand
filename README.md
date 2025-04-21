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
export task="Subject_2/clean_glasses/1/256"
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
```python
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from mhmp_vis.dataset_loader import load_cascaded_dataset

data_config = DATA_CONFIG_MAP['hand_motion']
data_storage_kwargs = dict(window_size=16, side='right', is_render_depth=False)
train_dataset, eval_dataset = load_cascaded_dataset(
    dataset_path='/mnt/vepfs/zbh/MHMP/datasets/',
    dataset_list='arctic,fpha,h2o,hoi4d,hot3d,taco',
    split_portion=0.95,
    data_config=data_config,
    data_storage_kwargs=data_storage_kwargs,
    test=False,
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
bash bash/train.sh -batch_size 104 -depthanything_path "" -incorporate_depth 0 -exp_name RGB_notunellm_6 -lora_rank 0 -tune_llm 0 -device "0,1,2,3,4,5,6,7" -dataset_list arctic,fpha,h2o,hoi4d,hot3d,taco
bash bash/test.sh -dataset_list taco -split_portion 0.95 -pretrain_ckpt /mnt/vepfs/zbh/MHMP/logs/RGB_notunellm/checkpoint-10000

# MHMP+

```