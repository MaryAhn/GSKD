#!/bin/bash
ENTRY=1
GPUS=2
CONFIG=configs/strategies/distill/resnet_dist.yaml
MODEL=tv_resnet18
PY_ARGS=${@:5}

set -x

python -m torch.distributed.launch --nproc_per_node=${GPUS} \
    ${ENTRY} -c ${CONFIG} --model ${MODEL} ${PY_ARGS}
