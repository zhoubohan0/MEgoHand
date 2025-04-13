#!/bin/bash

dataset_path="/mnt/vepfs/zbh/MHMP/datasets/"
depthanything_path="/mnt/vepfs/zbh/backup/pretrain/depth_anything_v2/"

while [[ $# -gt 0 ]]; do
  key="$1"
  case "$key" in
    -dataset_path)
      dataset_path="$2"
      shift 2
      ;;
    -depthanything_path)
      depthanything_path="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $key"
      exit 1
      ;;
  esac
done

# echo "dataset_path: $dataset_path, depthanything_path: $depthanything_path"

python -m pdb -m scripts.train \
  --dataset_path "$dataset_path" \
  --depthanything_path "$depthanything_path"
