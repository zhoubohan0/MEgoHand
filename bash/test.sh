#!/bin/bash

pretrain_ckpt=""
dataset_path="/mnt/vepfs/zbh/MHMP/datasets/"
dataset_list=""  # Initialize as empty array, separate by comma
split_portion=1.0
batch_size=16
debug=0
device="0"
mode="vis"  #"test"

while [[ $# -gt 0 ]]; do
  key="$1"
  case "$key" in
    -dataset_path)
      dataset_path="$2"
      shift 2
      ;;
    -dataset_list)
      # Split the comma-separated list into an array
      # IFS=',' read -ra dataset_list <<< "$2"
      dataset_list="$2"
      shift 2
      ;;
    -split_portion)
      split_portion="$2"
      shift 2
      ;;
    -pretrain_ckpt)
      pretrain_ckpt="$2"
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
    -batch_size)
      batch_size="$2"
      shift 2
      ;;
    -mode)
      mode="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $key"
      exit 1
      ;;
  esac
done

echo "mode: $mode, dataset_path: $dataset_path, dataset_list: $dataset_list, split_portion: $split_portion, batch_size: $batch_size, debug: $debug, device: $device, pretrain_ckpt: $pretrain_ckpt"

CUDA_VISIBLE_DEVICES=$device python -m pdb -m scripts.test \
  --mode $mode \
  --dataset_path $dataset_path \
  --dataset_list $dataset_list \
  --split_portion $split_portion \
  --batch_size $batch_size \
  --debug $debug \
  --pretrain_ckpt $pretrain_ckpt \
