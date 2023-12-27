CUDA_VISIBLE_DEVICES=4,5,6,7 \
  python -m torch.distributed.launch --nproc_per_node=4  --master_port 15007 \
      train_gskd.py \
      -c configs/strategies/distill/mbnet_dist_b1.yaml \
      --model mobilenet_v1 \
      --dataset imagenet \
      --data-path [your directory path to datasets]/ImageNet \
      --gskd-loss-weight 2.5 \
      --gskd-temperature 1.0 \
      --N 8 \
      --S 10 \
      --distill-layer 2 \
      --experiment [your experiment name]

