python -m torch.distributed.launch --nproc_per_node=2 --master_port 15001 \
    train_gskd.py \
    --teacher-model deeplabv3 \
    --student-model deeplabv3 \
    --teacher-backbone resnet101 \
    --student-backbone resnet18 \
    --dataset citys \
    --data [your directory path to datasets]/cityscapes \
    --save-dir [your directory path to store checkpoints] \
    --log-dir [your directory path to store log files] \
    --gpu-id 0,1,2,3 \
    --teacher-pretrained [your directory path to teacher pretrained checkpoint] \
    --student-pretrained-base [your directory path to student base pretrained checkpoint] \
    --lambda-gskd 10.0 \
    --lambda-kd 1.0 \
    --gskd-temperature 1.0 \
    --kd-temperature 1.0 \
    --max-iterations 40000 \
    --N 8 \
    --S -1