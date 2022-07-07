#!/usr/bin/env python3

import os
import json
import argparse
import datetime
import random
import shutil
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from tensorboardX import SummaryWriter

from models.gfnet import GFNet
from libs.dataloader.SemanticKitti import SemanticKitti
from libs.utils.training import train_epoch, validate
from libs.utils.sampler import DistributedEvalSampler
from libs.utils.cosine_schedule import CosineAnnealingWarmUpRestarts
from libs.utils.ohem import OhemCrossEntropy
from libs.utils.tools import (create_log, load_arch_cfg, load_data_cfg,
                              load_pretrained, recording_cfg,
                              get_weight_per_class, save_checkpoint,
                              resume_training, find_free_port)

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
        required=True,
        default='dataset',
        help='Dataset to train with.',
    )
    parser.add_argument(
        '--arch_cfg', '-ac',
        type=str,
        required=True,
        default='configs/resnet.yaml',
        help='Architecture yaml cfg file. See /config/arch for sample.',
    )
    parser.add_argument(
        '--data_cfg', '-dc',
        type=str,
        required=False,
        default='configs/semantic-kitti.yaml',
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
        '--resume', '-r',
        type=str,
        required=False,
        default=None,
        help='Directory to resume the checkpoint.'
    )

    parser.add_argument(
        '--debug',
        type=int,
        required=False,
        default=1,
        help='whether debug'
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

    writer = tb_dir = log_path = None
    if rank == 0:
        global logger
        logger, log_path, tb_dir = create_log(FLAGS.log, FLAGS.data_cfg, FLAGS.debug)
        recording_cfg(FLAGS.arch_cfg, FLAGS.data_cfg, log_path)
        writer = SummaryWriter(log_dir=tb_dir, flush_secs=20)

        # print summary of what we will do
        logger.info("----------")
        logger.info("INTERFACE:")
        logger.info("pwd: {}".format(os.getcwd()))
        logger.info("dataset: {}".format(FLAGS.dataset))
        logger.info("arch_cfg: {}".format(FLAGS.arch_cfg))
        logger.info("data_cfg: {}".format(FLAGS.data_cfg))
        logger.info("dist_url: {}".format(FLAGS.dist_url))
        logger.info("log: {}".format(log_path))
        logger.info("pretrained: {}".format(FLAGS.pretrained))
        logger.info("gpus: {}".format(FLAGS.gpus))
        logger.info("debug: {}".format(FLAGS.debug))
        logger.info('Configs: \n' + json.dumps(ARCH, indent=4, sort_keys=True))
        logger.info('Args: \n' + json.dumps(vars(FLAGS), indent=4, sort_keys=True))
        # logger.info('Data: \n' + json.dumps(DATA, indent=4, sort_keys=True))

    train_dataset = SemanticKitti(root=FLAGS.dataset,
                                sequences=DATA["split"]["train"],
                                labels=DATA["labels"],
                                color_map=DATA["color_map"],
                                learning_map=DATA["learning_map"],
                                learning_map_inv=DATA["learning_map_inv"],
                                range_cfg=ARCH['range'],         # configs for range view (dict)
                                polar_cfg=ARCH['polar'],         # configs for polar view (dict)
                                dataset_cfg=ARCH['dataset'])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=ARCH["train"]["batch_size"],
                                               num_workers=ARCH["train"]["workers"],
                                               sampler=train_sampler,
                                               shuffle=(train_sampler is None),
                                               pin_memory=True,
                                               drop_last=True)

    val_dataset = SemanticKitti(root=FLAGS.dataset,
                            sequences=DATA["split"]["valid"],
                            labels=DATA["labels"],
                            color_map=DATA["color_map"],
                            learning_map=DATA["learning_map"],
                            learning_map_inv=DATA["learning_map_inv"],
                            range_cfg=ARCH['range'],         # configs for range view (dict)
                            polar_cfg=ARCH['polar'],         # configs for polar view (dict)
                            dataset_cfg=ARCH['dataset'])
    val_sampler = DistributedEvalSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=ARCH["train"]["batch_size"],
                                             num_workers=ARCH["train"]["workers"],
                                             sampler=val_sampler,
                                             shuffle=False,
                                             pin_memory=True,
                                             drop_last=False)

    n_class = len(DATA["learning_map_inv"])

    loss_w = get_weight_per_class(ARCH["train"]["epsilon_w"],
                                  n_class,
                                  DATA["content"],
                                  DATA["learning_map"],
                                  DATA["learning_ignore"],
                                  )

    ignore_class = []
    for i, w in enumerate(loss_w):
      if w < 1e-10:
        ignore_class.append(i)

    # for uint8 trick
    loss_w = loss_w[1:]

    model = GFNet(ARCH,
                  layers=ARCH["backbone"]["layers"],
                  n_class=n_class-1,
                  flow=ARCH["train"]["flow"],
                  data_type=torch.float32)

    if rank == 0:
        logger.info(model)

    # add SyncBN
    if ARCH["train"]["syncbn"]:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = torch.nn.CrossEntropyLoss(weight=loss_w, ignore_index=255, reduction='mean').cuda()
    criterion_3d = OhemCrossEntropy(ignore_index=255, thresh=0.9, min_kept=10000, weight=None).cuda()

    if ARCH["train"]["optimizer"] == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=ARCH["train"]["min_lr"], momentum=0.9, weight_decay=1e-4
        )
        lr_scheduler = CosineAnnealingWarmUpRestarts(
            optimizer, T_0=len(train_loader)*ARCH["train"]["max_epochs"], T_mult=10,
            eta_max=ARCH["train"]["max_lr"], T_up=len(train_loader)*ARCH["train"]["wup_epochs"], gamma=0.5
        )
    elif ARCH["train"]["optimizer"] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=ARCH["train"]["lr"])
        lr_step = []
        for step in ARCH["train"]["lr_step"]:
            lr_step.append(step * len(train_loader))
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                            milestones=lr_step,
                                                            gamma=ARCH["train"]["lr_factor"])
    else:
        raise ValueError('unsopported optimizer type!')

    # when the weight for range or polar is 0, then the network has unused params
    unused = not (ARCH["train"]["flow"])
    model = DDP(model.cuda(), device_ids=[rank], find_unused_parameters=unused)

    # load pretrained
    start_epoch = 0
    if FLAGS.pretrained:
        model, start_epoch, best_iou = load_pretrained(FLAGS.pretrained, model)

    if FLAGS.resume:
        model, optimizer, start_epoch, best_iou = resume_training(FLAGS.resume, model, optimizer)
        lr_scheduler.step(start_epoch*len(train_loader))

    for epoch in range(start_epoch, ARCH["train"]["max_epochs"]):
        train_sampler.set_epoch(epoch)
        train_epoch(epoch, train_loader, model,
                    criterion, criterion_3d, optimizer,
                    lr_scheduler,
                    ARCH["loss"],
                    ARCH["train"]["report_batch"],
                    ARCH["train"]["max_epochs"],
                    writer)
        if epoch % 5 == 0 or epoch+1 == ARCH["train"]["max_epochs"]:
            miou = validate(epoch, val_loader,
                            model, criterion,
                            ARCH["loss"],
                            writer)
            is_best = miou > best_iou
            if miou > best_iou:
                best_iou = miou

            if rank == 0:
                save_checkpoint(
                    {
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_iou': best_iou
                    }, log_path, is_best
                )
    if rank == 0:
        src = os.path.join(log_path, 'model_best.pth.tar')
        if os.path.isfile(src):
            shutil.move(src, os.path.join(log_path, 'model_best_{:.4f}.pth.tar'.format(best_iou)))

if __name__ == '__main__':
    main()
