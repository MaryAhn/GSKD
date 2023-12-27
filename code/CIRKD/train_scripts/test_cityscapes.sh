CUDA_VISIBLE_DEVICES=0,1 \
  python -m torch.distributed.launch --nproc_per_node=2 --master_port 15002 test.py \
  --model deeplabv3 \
  --backbone resnet18 \
  --data /home/cvlab/datasets/cityscapes \
  --save-dir /home/cvlab/nfs_clientshare/keonhee/CIRKD/result_image/grayscale/CIRKD \
  --save-pred \
  --pretrained /home/cvlab/Downloads/CIRKD.pth