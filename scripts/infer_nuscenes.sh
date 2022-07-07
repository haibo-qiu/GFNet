#!/usr/bin/env bash

python -u test_nuscenes.py --dataset dataset/nuScenes/full/ \
                           --arch_cfg configs/resnet_nuscenes.yaml \
                           --data_cfg configs/nuscenes.yaml \
                           --pretrained pretrained/gfnet_76.8_nuscenes.pth.tar \
                           --gpus '0' \
                           --test 0 \
                           --eval 1

# reproduce the results we submitted to the test server
# python -u test_nuscenes.py --dataset dataset/nuScenes/full/ \
#                            --pkl_train dataset/nuScenes/nuscenes_trainval.pkl \
#                            --arch_cfg configs/resnet_nuscenes.yaml \
#                            --data_cfg configs/nuscenes.yaml \
#                            --pretrained pretrained/gfnet_submit_nuscenes.pth.tar \
#                            --gpus '0' \
#                            --test 1 \
#                            --eval 1
