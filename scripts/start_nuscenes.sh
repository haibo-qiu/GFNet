python -u train_nuscenes.py --dataset dataset/nuScenes/full/ \
                            --pkl_train dataset/nuScenes/nuscenes_train.pkl \
                            --arch_cfg configs/resnet_nuscenes.yaml \
                            --data_cfg configs/nuscenes.yaml \
                            --gpus '0,1,2,3' \
                            --debug 0
