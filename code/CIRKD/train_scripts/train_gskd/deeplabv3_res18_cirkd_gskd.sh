CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -m torch.distributed.launch --nproc_per_node=4 --master_port 15003 \
    train_cirkd_gskd.py \
    --teacher-model deeplabv3 \
    --student-model deeplabv3 \
    --teacher-backbone resnet101 \
    --student-backbone resnet18 \
    --data [your directory path to datasets]/cityscapes \
    --dataset ade20k \
    --save-dir [your directory path to save checkpoints] \
    --log-dir [your directory path to save log files] \
    --teacher-pretrained [your directory path to teacher pretrained checkpoint] \
    --student-pretrained-base [your directory path to student base pretrained checkpoint] \
    --batch-size 16 \
    --max-iterations 40000 \
    --lambda-image-desc 5.0 \
    --lambda-memory-pixel 0.01 \
	  --lambda-memory-region 0.01 \
    --N 8 \
    --S -1
