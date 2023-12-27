CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch --nproc_per_node=2 save_embeddings.py \
    --model deeplabv3 \
    --backbone resnet18 \
    --dataset citys \
    --data /home/cvlab/datasets/cityscapes \
    --save-dir /home/cvlab/DIST_KD/result/result_image/tsne \
    --pretrained /home/cvlab/CIRKD/chkpt/paper_cirkd/kd_deeplabv3_resnet18_citys_best_model.pth

python tsne_visual.py