#!/bin/bash

pretrain_ckpt=""
dataset_path="/mnt/vepfs/zbh/MHMP/datasets/"
testset="eval"
debug=0
device="0"
vis_ren_gen=0

while [[ $# -gt 0 ]]; do
  key="$1"
  case "$key" in
    -dataset_path)
      dataset_path="$2"
      shift 2
      ;;
    -testset)
      testset="$2"
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
    -vis_ren_gen)
      vis_ren_gen="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $key"
      exit 1
      ;;
  esac
done

echo "dataset_path: $dataset_path, testset: $testset, debug: $debug, device: $device, pretrain_ckpt: $pretrain_ckpt, vis_ren_gen: $vis_ren_gen"

CUDA_VISIBLE_DEVICES=$device python -m scripts.test \
  --dataset_path $dataset_path \
  --testset $testset \
  --debug $debug \
  --pretrain_ckpt $pretrain_ckpt \
  --vis_ren_gen $vis_ren_gen \
  --vis_ren_ori 