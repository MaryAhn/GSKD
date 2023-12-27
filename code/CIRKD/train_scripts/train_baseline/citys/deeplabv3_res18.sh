CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -m torch.distributed.launch --nproc_per_node=4 train_baseline.py \
    --model deeplabv3 \
    --backbone resnet18 \
    --data /home/nsml/datasets/cityscapes/ \
    --save-dir /home/nsml/nfs_clientshare/keonhee/code/CIRKD/chkpt/deeplabv3_resnet18 \
    --log-dir /home/nsml/nfs_clientshare/keonhee/code/CIRKD/log/deeplabv3_resnet18 \
    --pretrained-base pretrained/resnet18-imagenet.pth \
    --batch-size 8 \
    --max-iterations 80000