CUDA_VISIBLE_DEVICES=0,1 \
    python -m torch.distributed.launch --nproc_per_node=2 --master_port 12348 eval.py \
    --model psp \
    --backbone resnet18 \
    --dataset citys \
    --data /home/cvlab/datasets/cityscapes \
    --save-dir /home/cvlab/GSKD/result_log/CWD_CITY/cwd_ours_N_8_S_C_psp \
    --pretrained /home/cvlab/GSKD/chkpt/CWD_CITY/cwd_ours_N_8_S_C_psp/kd_psp_resnet18_citys_best_model.pth

