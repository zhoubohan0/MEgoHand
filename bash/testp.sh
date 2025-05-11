# bash bash/test.sh -testset test -pretrain_ckpt /mnt/vepfs/zbh/MHMP/logs/TID_l1_6_UD/checkpoint-50000 -device 0  &
bash bash/test.sh -testset test -pretrain_ckpt /mnt/vepfs/zbh/MHMP/logs/ID_l1_6_UD/checkpoint-50000 -device 1  & 
bash bash/test.sh -testset test -pretrain_ckpt /mnt/vepfs/zbh/MHMP/logs/T_l1_6/checkpoint-5000 -device 2  & 
# bash bash/test.sh -testset test -pretrain_ckpt /mnt/vepfs/zbh/MHMP/logs/TI_l1_6/checkpoint-25000 -device 3  & 
# bash bash/test.sh -testset test -pretrain_ckpt /mnt/vepfs/zbh/MHMP/logs/TID_l1_6_DA/checkpoint-45000 -device 4  & 
# bash bash/test.sh -testset test -pretrain_ckpt /mnt/vepfs/zbh/MHMP/logs/TID_l1_6_UD_rel/checkpoint-50000 -device 5  & 
# wait
