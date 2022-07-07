#!/usr/bin/env bash

python -u dataset/utils_nuscenes/validate_submission.py --result_path logs/nuscenes/eval/2022-6-29-17-32-28/preds/ \
                                                        --eval_set test \
                                                        --dataroot dataset/nuScenes/full/ \
                                                        --version 'v1.0-test' \
                                                        --verbose True \
                                                        --zip_out dataset/nuScenes/
