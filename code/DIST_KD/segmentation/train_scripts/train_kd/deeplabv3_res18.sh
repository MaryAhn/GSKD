python -m torch.distributed.launch --nproc_per_node=4 \
    train_kd.py \
    --teacher-model deeplabv3 \
    --student-model deeplabv3 \
    --teacher-backbone resnet101 \
    --student-backbone resnet18 \
    --data [your directory path to datasets]/cityscapes \
    --save-dir [your directory path to store checkpoints] \
    --log-dir [your directory path to store log files] \
    --gpu-id 0,1,2,3 \
    --teacher-pretrained [your directory path to teacher pretrained checkpoint] \
    --student-pretrained-base [your directory path to student base pretrained checkpoint]