# stage 1
# python -m inv_mano.train_inv_mano --exp_name stage1 -s 1 
# stage 2
# python -m inv_mano.train_inv_mano --exp_name stage2 -s 2 -c inv_mano/logs/stage1/invmano_v1.pth
# test
python -m inv_mano.train_inv_mano --exp_name stage2 --mode test -c inv_mano/logs/stage2/invmano_240.pth