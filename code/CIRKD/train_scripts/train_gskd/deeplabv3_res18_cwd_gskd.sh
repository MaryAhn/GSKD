CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -m torch.distributed.launch --nproc_per_node=4 --master_port 15003 \
    train_cwd_desc_ss_gt.py \
    --teacher-model deeplabv3 \
    --student-model deeplab_mobile \
    --teacher-backbone resnet101 \
    --student-backbone mobilenetv2 \
    --dataset citys \
    --data [your directory path to datasets]/cityscapes \
    --dataset ade20k \
    --save-dir [your directory path to save checkpoints] \
    --log-dir [your directory path to save log files] \
    --teacher-pretrained [your directory path to teacher pretrained checkpoint] \
    --student-pretrained-base [your directory path to student base pretrained checkpoint] \
    --batch-size 8 \
    --max-iterations 80000 \
    --lambda-desc 2.0 \
    --lambda-cwd-fea 50. \
    --lambda-cwd-logit 3. \
    --lambda-kd 1.0 \
    --lambda-d 0.1 \
    --lambda-adv 0.001 \
    --N 8 \
    --S -1