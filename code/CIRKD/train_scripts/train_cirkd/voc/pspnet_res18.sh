CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python -m torch.distributed.launch --nproc_per_node=8 \
    train_cirkd.py \
    --teacher-model deeplabv3 \
    --student-model psp \
    --teacher-backbone resnet101 \
    --student-backbone resnet18 \
    --dataset voc \
    --crop-size 512 512 \
    --data /home/cvlab/datasets/VOC2012 \
    --lambda-memory-pixel 0.01 \
	  --lambda-memory-region 0.01 \
    --save-dir /home/cvlab/nfs_clientshare/keonhee/code/CIRKD/chkpt/Reproduce/CIRKD_psp_voc_reproduce \
    --log-dir /home/cvlab/nfs_clientshare/keonhee/code/CIRKD/result/Reproduce/CIRKD_psp_voc_reproduce\
    --teacher-pretrained /home/cvlab/nfs_clientshare/keonhee/code/CIRKD/pretrained/deeplabv3_resnet101_pascal_aug_best_model.pth \
    --student-pretrained-base /home/cvlab/nfs_clientshare/keonhee/code/CIRKD/pretrained/resnet18-imagenet.pth \
    --batch-size 16 \
    --max-iterations 40000