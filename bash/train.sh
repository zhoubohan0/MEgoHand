#!/bin/bash

multimodal="tv"
dataset_path="/mnt/vepfs/zbh/MHMP/datasets/"
depthanything_path="/mnt/vepfs/zbh/backup/pretrain/depth_anything_v2/"
dataset_list=""  # Initialize as empty array, separate by comma
exp_name="runs"   
debug=0
device="0"
lora_rank=0
tune_llm=0
tune_visual=1
batch_size=150
resume=""

while [[ $# -gt 0 ]]; do
  key="$1"
  case "$key" in
    -multimodal)
      multimodal="$2"
      shift 2
      ;;
    -dataset_path)
      dataset_path="$2"
      shift 2
      ;;
    -depthanything_path)
      depthanything_path="$2"
      shift 2
      ;;
    -dataset_list)
      # Split the comma-separated list into an array
      # IFS=',' read -ra dataset_list <<< "$2"
      dataset_list="$2"
      shift 2
      ;;
    -exp_name)
      exp_name="$2"
      shift 2
      ;;
    -debug)
      debug="$2"
      shift 2
      ;;
    -device)
      device="$2"
      shift 2
      ;;
    -lora_rank)
      lora_rank="$2"
      shift 2
      ;;
    -tune_llm)
      tune_llm="$2"
      shift 2
      ;;
    -tune_visual)
      tune_visual="$2"
      shift 2
      ;;
    -batch_size)
      batch_size="$2"
      shift 2
      ;;
    -resume)
      resume="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $key"
      exit 1
      ;;
  esac
done

echo "multimodal: $multimodal, dataset_path: $dataset_path, depthanything_path: $depthanything_path, dataset_list: ${dataset_list[@]}, exp_name: $exp_name, debug: $debug, device: $device, lora_rank: $lora_rank, tune_llm: $tune_llm, tune_visual: $tune_visual, batch_size: $batch_size, resume: $resume"

CUDA_VISIBLE_DEVICES=$device python -m scripts.train \
  --num_gpus $(echo $device | tr -cd ',' | wc -c | awk '{print $1 + 1}') \
  --multimodal "$multimodal" \
  --dataset_path "$dataset_path" \
  --depthanything_path "$depthanything_path" \
  --exp_name "$exp_name" \
  --debug "$debug" \
  --dataset_list "$dataset_list" \
  --lora_rank "$lora_rank" \
  --tune_llm "$tune_llm" \
  --tune_visual "$tune_visual" \
  --batch_size "$batch_size" \
  --resume "$resume" \
  # Uncomment the following line if you want to pass the dataset_list as a list
  # ${dataset_list[@]:+--dataset_list ${dataset_list[@]}}
