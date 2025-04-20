> To modify action horizon
> 
> gr00t/experiment/data_config.py L693
> 
> model_config/config.json action_horizon

Prepare Dataset
```bash
python -m mhmp_vis.dataset_loader.dataset_fpha
```

Visualize Task
```bash
cd mhmp_vis
python -m pdb -m dataset_visualizer.general_offline_visualizer -d fpha -t "Subject_2/clean_glasses/1/256"
```

Calculate Statistics
for each dataset:
```bash
bash bash/statistics.sh -dataset_list taco
```
Final pooling:
```python
python model_config/get_metadata.py
```

Training
```bash
# MHMP
bash bash/train.sh -batch_size 104 -depthanything_path "" -incorporate_depth 0 -exp_name RGB_notunellm_6 -lora_rank 0 -tune_llm 0 -device "0,1,2,3,4,5,6,7" -dataset_list arctic,fpha,h2o,hoi4d,hot3d,taco
# bash bash/train.sh -dataset_list taco -depthanything_path "" -incorporate_depth 0 -exp_name RGB_lora -lora_rank 32 -tune_llm 1 -device "4,5,6,7"  # -resume "logs/RGB_lora/checkpoint-v0" -debug 1
bash bash/test.sh -dataset_list taco -split_portion 0.95 -pretrain_ckpt /mnt/vepfs/zbh/MHMP/logs/RGB_notunell
m/checkpoint-10000
# MHMP+

```