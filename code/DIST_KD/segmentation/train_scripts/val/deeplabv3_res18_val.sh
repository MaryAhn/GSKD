python -m torch.distributed.launch --nproc_per_node=2 --master_port 15002 eval.py \
  --model deeplabv3 \
  --backbone resnet101 \
  --dataset citys \
  --data [your directory path to datasets]/cityscapes \
  --save-dir [your directory path to save the result] \
  --pretrained [your directory path to load the checkpoint] \
  --gpu-id 0,1

