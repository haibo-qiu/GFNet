#!/usr/bin/env python3

import os
import json
import shutil
import random
import argparse
import datetime
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm

from models.gfnet import GFNet
from libs.dataloader.nuScenes import Nuscenes
from libs.utils.training import validate
from libs.utils.sampler import DistributedEvalSampler
from libs.utils.tools import (create_eval_log, load_arch_cfg, find_free_port,
                              load_data_cfg, load_pretrained, recover_uint8_trick)

seed = 6
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

best_iou = 0.0

def parse_args():
    parser = argparse.ArgumentParser(description='Geometric Flow Network for 3D Point Clouds Semantic Segmentation')
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=False,
        default='dataset/nuScenes/full/',
        help='Dataset to train with.',
    )
    parser.add_argument(
        '--pkl_train',
        type=str,
        required=False,
        default='dataset/nuScenes/nuscenes_train.pkl',
        help='pkl file containing train info',
    )
    parser.add_argument(
        '--pkl_val',
        type=str,
        required=False,
        default='dataset/nuScenes/nuscenes_val.pkl',
        help='pkl file containing val info',
    )
    parser.add_argument(
        '--pkl_test',
        type=str,
        required=False,
        default='dataset/nuScenes/nuscenes_test.pkl',
        help='pkl file containing test info',
    )
    parser.add_argument(
        '--arch_cfg', '-ac',
        type=str,
        required=False,
        default='configs/resnet_nuscenes.yaml',
        help='Architecture yaml cfg file. See /config/arch for sample.',
    )
    parser.add_argument(
        '--data_cfg', '-dc',
        type=str,
        required=False,
        default='configs/nuscenes.yaml',
        help='Classification yaml cfg file. See /config/labels for sample.',
    )
    parser.add_argument(
        '--log', '-l',
        type=str,
        default='logs/' +
        datetime.datetime.now().strftime("%Y-%-m-%d-%H-%M-%S") + '/',
        help='Directory to put the log data. Default: ~/logs/date+time'
    )
    parser.add_argument(
        '--pretrained', '-p',
        type=str,
        required=False,
        default=None,
        help='Directory to get the pretrained model. If not passed, do from scratch!'
    )

    parser.add_argument(
        '--eval',
        type=int,
        required=False,
        default=1,
        help='whether eval'
    )

    parser.add_argument(
        '--test',
        type=int,
        required=False,
        default=0,
        help='whether test'
    )

    parser.add_argument(
        '--dist_backend',
        type=str,
        required=False,
        default='nccl',
        help='backend'
    )

    parser.add_argument(
        '--dist_url',
        type=str,
        required=False,
        default='tcp://127.0.0.1:8081',
        help='init method'
    )

    parser.add_argument(
        '--gpus', '-g',
        type=str,
        required=True,
        default='0',
        help='gpus to use'
    )
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS


def main():
    FLAGS = parse_args()

    ARCH = load_arch_cfg(FLAGS.arch_cfg)
    DATA = load_data_cfg(FLAGS.data_cfg)

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus
    torch.backends.cudnn.benchmark = True
    gpus = [int(i) for i in FLAGS.gpus.split(',')]

    port = find_free_port()
    FLAGS.dist_url = 'tcp://localhost:{}'.format(port)

    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    mp.set_sharing_strategy('file_system')
    nprocs = len(gpus)

    mp.spawn(main_worker, args=(FLAGS, ARCH, DATA, nprocs), nprocs=nprocs, join=True, daemon=False)

def main_worker(rank, FLAGS, ARCH, DATA, world_size):
    global best_iou
    dist.init_process_group(backend=FLAGS.dist_backend,
                            init_method=FLAGS.dist_url,
                            world_size=world_size,
                            rank=rank)
    torch.cuda.set_device(rank)

    log_path = None
    if rank == 0:
        global logger
        logger, log_path = create_eval_log(os.path.dirname(FLAGS.pretrained), FLAGS.data_cfg)

        # print summary of what we will do
        logger.info("----------")
        logger.info("INTERFACE:")
        logger.info("dataset: {}".format(FLAGS.dataset))
        logger.info("arch_cfg: {}".format(FLAGS.arch_cfg))
        logger.info("data_cfg: {}".format(FLAGS.data_cfg))
        logger.info("log: {}".format(log_path))
        logger.info("pretrained: {}".format(FLAGS.pretrained))
        logger.info("gpus: {}".format(FLAGS.gpus))
        logger.info('Configs: \n' + json.dumps(ARCH, indent=4, sort_keys=True))

    if FLAGS.test:  # eval on test set, and submit to test server
        set_name = 'test'
        pkl_path = FLAGS.pkl_test
    else: # eval on val set
        set_name = 'val'
        pkl_path = FLAGS.pkl_val

    # for submitting to the eval server
    preds_path = os.path.join(log_path, 'preds', 'lidarseg', set_name)
    os.makedirs(preds_path, exist_ok=True)
    subinfo_path =os.path.join(log_path, 'preds', set_name)
    os.makedirs(subinfo_path, exist_ok=True)
    shutil.copy('dataset/utils_nuscenes/submission.json', subinfo_path)

    dataset = Nuscenes(pkl_path=pkl_path,
                        data_path=FLAGS.dataset,
                        labels=DATA['labels_16'],
                        range_cfg=ARCH['range'],
                        polar_cfg=ARCH['polar'],
                        dataset_cfg=ARCH['dataset'],
                        color_map=DATA['color_map'],
                        learning_map=DATA['learning_map'],
                        split=set_name)
    data_sampler = DistributedEvalSampler(dataset)
    data_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=ARCH["train"]["batch_size"]*3,
                                             num_workers=ARCH["train"]["workers"],
                                             sampler=data_sampler,
                                             shuffle=False,
                                             pin_memory=True,
                                             drop_last=False)


    n_class = dataset.nclasses

    model = GFNet(ARCH,
                  layers=ARCH["backbone"]["layers"],
                  n_class=n_class-1,
                  flow=ARCH["train"]["flow"],
                  data_type=torch.float32)
    # add SyncBN
    if ARCH["train"]["syncbn"]:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # when the weight for range or polar is 0, then the network has unused params
    unused = not (ARCH["train"]["flow"])
    model = DDP(model.cuda(), device_ids=[rank], find_unused_parameters=unused)

    # load pretrained
    epoch = 0
    if FLAGS.pretrained:
        model, start_epoch, best_iou = load_pretrained(FLAGS.pretrained, model)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='mean').cuda()
    w_loss = ARCH["loss"]
    model.eval()
    with torch.no_grad():
        if FLAGS.eval and not FLAGS.test: # only for valid set
            if rank == 0:
                logger.info('start to evaluate with my own implement')
            validate(0, data_loader, model, criterion, w_loss, None)

        if rank == 0:
            logger.info('start to generate predicions to {} for official eval'.format(log_path))
        pbar = tqdm(total=len(data_loader))
        for i, (range_data, polar_data, r2p_matrix, p2r_matrix, knns) in enumerate(data_loader):
            batch_size = r2p_matrix.shape[0]
            in_vol, proj_mask, proj_labels, _, proj_xy, unproj_labels, path_seq, path_name, pxpy_range, _, _, _, _, _, points, _, _, _ = range_data
            _, vox_label, train_grid, full_labels, pt_fea, pxpypz_polar, num_pt = polar_data

            in_vol = in_vol.cuda(non_blocking=True)
            train_grid_2d = train_grid[:, :, :2].cuda(non_blocking=True)
            pt_fea = pt_fea.cuda(non_blocking=True)

            r2p_matrix = r2p_matrix.cuda(non_blocking=True)
            p2r_matrix = p2r_matrix[:, :, :, :2].cuda(non_blocking=True)

            pxpy_range = torch.flip(pxpy_range.cuda(non_blocking=True), dims=[-1])    # because for F.grid_sample, i,j,k index w,h,d (i.e., reverse order)
            pxpypz_polar = torch.flip(pxpypz_polar.cuda(non_blocking=True), dims=[-1])
            points = points.cuda(non_blocking=True)
            knns = knns.cuda(non_blocking=True)

            fusion_pred, range_pred, polar_pred, range_x, polar_x = model(
                    in_vol, pt_fea, train_grid_2d, num_pt, r2p_matrix, p2r_matrix, pxpy_range, pxpypz_polar, points, knns)

            pbar.update(1)
            if w_loss['fusion']:
                pred_to_test = fusion_pred
            elif (w_loss['range']+w_loss['polar']):
                pred_to_test = w_loss['range'] * range_pred + w_loss['polar'] * polar_pred
            elif (w_loss['range_proj']+w_loss['polar_proj']):
                pred_to_test = get_pred(range_x, polar_x, proj_xy, train_grid, full_labels, num_pt,
                        w_loss['range_proj'], w_loss['polar_proj'], data_loader.dataset.to_original,log_path, path_name, path_seq)

            for i, pred in enumerate(pred_to_test):
                final_pred = torch.argmax(pred, dim=0).squeeze()[:num_pt[i]]
                pred_np = final_pred.cpu().numpy()
                pred_np = recover_uint8_trick(pred_np)
                # pred_np = data_loader.dataset.to_original(pred_np)
                name = path_name[i]
                path = os.path.join(preds_path, name)
                pred_np.astype(np.uint8).tofile(path)
        pbar.close()

def get_pred(pred_range, pred_polar, range_proj, polar_proj, full_labels, num_pt, w_range, w_polar, to_original, log_path, path_name, path_seq):
    """
    pred_range: [B, 19, 64, 2048] predicions from range view
    pred_polar: [B, 19, 480, 360, 32] predicions from polar view
    range_proj: [B, max_points, 2] pixel location of 3d points
    polar_proj: [B, max_points, 3] vol location of 3d points
    full_labels: [B, max_points, 1] labels for 3d points
    num_pt: [B,] indicates specific number points for each sample
    """
    final_preds = []
    pred_range = pred_range.permute(0, 2, 3, 1)
    pred_polar = pred_polar.permute(0, 2, 3, 4, 1)
    for i, label in enumerate(full_labels):
        # fusion between range view and polar view
        length = label.shape[-1]
        proj_r = range_proj[i][:num_pt[i]]    # N x 2
        proj_p = polar_proj[i][:num_pt[i]]    # N x 3
        r_3d = pred_range[i, proj_r[:, 0], proj_r[:, 1], :]  # N x 19
        p_3d = pred_polar[i, proj_p[:, 0], proj_p[:, 1], proj_p[:, 2], :]  # N x 19

        r_3d = r_3d * int(w_range!=0)
        p_3d = p_3d * int(w_polar!=0)

        r_3d_pred = torch.argmax(r_3d, dim=-1).squeeze()
        p_3d_pred = torch.argmax(p_3d, dim=-1).squeeze()

        fused_3d = torch.cat((r_3d.unsqueeze(2), p_3d.unsqueeze(2)), dim=-1) # N x 19 x 2
        # fusing: max, mean or something else
        final_3d = torch.mean(fused_3d, dim=-1) # N x 19
        final_3d_pad = F.pad(input=final_3d, pad=(0, 0, 0, length-final_3d.shape[0]), mode='constant', value=0)
        final_preds.append(final_3d_pad[None, :, :])
    final_pred = torch.cat(final_preds, dim=0)
    final_pred = final_pred.permute(0, 2, 1)

    return final_pred

if __name__ == '__main__':
    main()
