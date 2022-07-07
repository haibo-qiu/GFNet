#!/usr/bin/env bash

python -u test.py  --dataset dataset/SemanticKITTI/ \
                   --arch_cfg configs/resnet_semantickitti.yaml \
                   --data_cfg configs/semantic-kitti.yaml \
                   --pretrained pretrained/gfnet_63.0_semantickitti.pth.tar \
                   --gpus '0' \
                   --test 0 \
                   --eval 1

# reproduce the results we submitted to the test server
# python -u test.py  --dataset dataset/SemanticKITTI/ \
#                    --arch_cfg configs/resnet_semantickitti.yaml \
#                    --data_cfg configs/semantic-kitti-trainval.yaml \
#                    --pretrained pretrained/gfnet_submit_semantickitti.pth.tar \
#                    --gpus '0' \
#                    --test 1 \
#                    --eval 1
