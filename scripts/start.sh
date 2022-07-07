python -u train.py --dataset dataset/SemanticKITTI/ \
                   --arch_cfg configs/resnet_semantickitti.yaml \
                   --data_cfg configs/semantic-kitti.yaml \
                   --gpus '0,1,2,3' \
                   --debug 0
