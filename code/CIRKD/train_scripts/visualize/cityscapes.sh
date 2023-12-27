
#'''generate color pallete segmentation mask'''
CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.5.0.4 --master_port 26501  \
    visualize_test.py \
    --model deeplabv3 \
    --backbone resnet18 \
    --dataset citys \
    --data /home/cvlab/datasets/cityscapes \
    --save-dir /home/cvlab/nfs_clientshare/keonhee/code/CIRKD/test_image \
    --save-pred \
    --pretrained /home/cvlab/nfs_clientshare/keonhee/code/CIRKD/chkpt/intrakd_desc_ss_8_80k_0.5/kd_deeplabv3_resnet18_citys_best_model.pth

#'''generate blend segmentation mask'''
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#python -m torch.distributed.launch --nproc_per_node=8 --master_addr 127.5.0.4 --master_port 26501  \
#    eval.py \
#    --model deeplabv3 \
#    --backbone resnet101 \
#    --dataset citys \
#    --data [your dataset path]/cityscapes/ \
#    --save-dir [your directory path to store segmentation mask files] \
#    --save-pred \
#    --blend \
#    --gpu-id 0,1,2,3,4,5,6,7 \
#    --pretrained [your pretrained model path]
