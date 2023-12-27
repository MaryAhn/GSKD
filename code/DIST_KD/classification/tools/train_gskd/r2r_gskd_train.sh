CUDA_VISIBLE_DEVICES=6,7 \
  python -m torch.distributed.launch --nproc_per_node=2  --master_port 15007 \
      train_gskd.py \
      -c configs/strategies/distill/resnet_kd_b1_2gpu.yaml \
      --model resnet18 \
      --dataset imagenet \
      --data-path [your directory path to datasets]/ImageNet \
      --gskd-loss-weight 5.0 \
      --gskd-temperature 1.0 \
      --N 16 \
      --S 10 \
      --distill-layer 2 \
      --experiment [your experiment name]
