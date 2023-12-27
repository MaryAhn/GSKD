CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -m torch.distributed.launch --nproc_per_node=8 train_baseline.py \
    --model psp \
    --backbone resnet18 \
    --data /home/cvlab/datasets/cityscapes \
    --save-dir /home/cvlab/nfs_clientshare/keonhee/code/CIRKD/chkpt/Reproduce/student_psp_resnet18 \
    --log-dir /home/cvlab/nfs_clientshare/keonhee/code/CIRKD/log/student_psp_resnet18 \
    --teacher-pretrained /home/cvlab/nfs_clientshare/keonhee/code/CIRKD/pretrained/deeplabv3_resnet101_citys_best_model.pth \
    --student-pretrained-base /home/cvlab/nfs_clientshare/keonhee/CIRKD/pretrained/resnet18-imagenet.pth \
    --batch-size 8 \
    --max-iterations 80000 \


#CUDA_VISIBLE_DEVICES=0,1,2,3 \
#  python -m torch.distributed.launch --nproc_per_node=4 eval.py \
#  --model psp \
#  --backbone resnet18 \
#  --data [your dataset path]/cityscapes/ \
#  --save-dir [your directory path to store log files] \
#  --pretrained [your pretrained model path]