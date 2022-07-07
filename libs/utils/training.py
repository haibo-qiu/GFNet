import time
import torch
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
from tensorboardX import SummaryWriter

from libs.utils.tools import AverageMeter, mp_logger, RPF_Hist, display_iou, fast_hist, uint8_trick
from libs.utils.lovasz_losses import lovasz_softmax

def get_hist(pred_range, pred_polar, range_proj, polar_proj, full_labels, num_pt, w_range, w_polar):
    """
    pred_range: [B, 19, 64, 2048] predicions from range view
    pred_polar: [B, 19, 480, 360, 32] predicions from polar view
    range_proj: [B, max_points, 2] pixel location of 3d points
    polar_proj: [B, max_points, 3] vol location of 3d points
    full_labels: [B, max_points, 1] labels for 3d points
    num_pt: [B,] indicates specific number points for each sample
    """
    hist = RPF_Hist()
    nclass = pred_range.shape[1]
    pred_range = pred_range.permute(0, 2, 3, 1)
    pred_polar = pred_polar.permute(0, 2, 3, 4, 1)
    for i, label in enumerate(full_labels):
        # fusion between range view and polar view
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
        # final_3d = torch.max(fused_3d, dim=-1)[0] # N x 19
        final_3d = torch.mean(fused_3d, dim=-1) # N x 19
        final_3d_pred = torch.argmax(final_3d, dim=-1).squeeze()

        label = uint8_trick(label.squeeze()[:num_pt[i]]).long().cuda()
        hist.append(fast_hist(r_3d_pred, label, nclass),
                    fast_hist(p_3d_pred, label, nclass),
                    fast_hist(final_3d_pred, label, nclass))

    return hist

def get_hist_frp(fusion_preds, range_preds, polar_preds, full_labels, num_pt):
    """
    pred_range: [B, max_points, 19] predicions from range view
    full_labels: [B, max_points, 1] labels for 3d points
    num_pt: [B,] indicates specific number points for each sample
    """
    hist = RPF_Hist()
    nclass = range_preds.shape[1]
    for i, label in enumerate(full_labels):
        f_pred = torch.argmax(fusion_preds[i], dim=0).squeeze()[:num_pt[i]]
        r_pred = torch.argmax(range_preds[i], dim=0).squeeze()[:num_pt[i]]
        p_pred = torch.argmax(polar_preds[i], dim=0).squeeze()[:num_pt[i]]

        label = uint8_trick(label.squeeze()[:num_pt[i]]).long().cuda()
        hist.append(fast_hist(r_pred, label, nclass),
                    fast_hist(p_pred, label, nclass),
                    fast_hist(f_pred, label, nclass))

    return hist

def train_epoch(epoch, train_loader, model, criterion, criterion_3d, optimizer, lr_scheduler, w_loss, report, max_epoch, writer):
    losses_range = AverageMeter()
    losses_polar = AverageMeter()
    losses_fusion = AverageMeter()
    losses_img = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ious = AverageMeter()
    ious_range = AverageMeter()
    ious_polar = AverageMeter()
    model.train()

    d1_time = AverageMeter()
    d2_time = AverageMeter()
    d3_time = AverageMeter()
    d4_time = AverageMeter()

    hist_list_img = RPF_Hist()
    hist_list = RPF_Hist()
    collect = False
    count = 0
    start = time.time()
    max_iter = max_epoch * len(train_loader)
    for i, (range_data, polar_data, r2p_matrix, p2r_matrix, knns) in enumerate(train_loader):
        data_time.update(time.time() - start)
        batch_size = r2p_matrix.shape[0]
        d1 = time.time()
        in_vol, proj_mask, proj_labels, _, proj_xy, unproj_labels, path_seq, path_name, pxpy_range, _, _, _, _, _, points, _, _, real_num_pt = range_data
        _, vox_label, train_grid, full_labels, pt_fea, pxpypz_polar, num_pt = polar_data

        proj_labels = uint8_trick(proj_labels)
        vox_label = uint8_trick(vox_label)
        unproj_labels = uint8_trick(unproj_labels)

        in_vol = in_vol.cuda(non_blocking=True)
        proj_labels = proj_labels.long().cuda(non_blocking=True)
        train_grid_2d = train_grid[:, :, :2].cuda(non_blocking=True)
        pt_fea = pt_fea.cuda(non_blocking=True)
        vox_label = vox_label.long().cuda(non_blocking=True)

        r2p_matrix = r2p_matrix.cuda(non_blocking=True)
        p2r_matrix = p2r_matrix[:, :, :, :2].cuda(non_blocking=True)

        unproj_labels = unproj_labels.long().cuda(non_blocking=True)
        pxpy_range = torch.flip(pxpy_range.cuda(non_blocking=True), dims=[-1])    # because for F.grid_sample, i,j,k index w,h,d (i.e., reverse order)
        pxpypz_polar = torch.flip(pxpypz_polar.cuda(non_blocking=True), dims=[-1])
        points = points.cuda(non_blocking=True)
        knns = knns.cuda(non_blocking=True)

        d1_time.update(time.time()-d1)
        d2 = time.time()
        optimizer.zero_grad()
        fusion_pred, range_pred, polar_pred, range_x, polar_x = model(
                in_vol, pt_fea, train_grid_2d, num_pt, r2p_matrix, p2r_matrix, pxpy_range, pxpypz_polar, points, knns)
        d2_time.update(time.time()-d2)
        d3 = time.time()
        loss_fusion = criterion_3d(fusion_pred, unproj_labels)
        loss_range = criterion_3d(range_pred, unproj_labels)
        loss_polar = criterion_3d(polar_pred, unproj_labels)

        loss_polar_img = criterion(polar_x, vox_label) + lovasz_softmax(torch.nn.functional.softmax(polar_x, dim=1), vox_label, ignore=255)
        loss_range_img = criterion(range_x, proj_labels) + lovasz_softmax(torch.nn.functional.softmax(range_x, dim=1), proj_labels, ignore=255)
        loss = w_loss['fusion']*loss_fusion + w_loss['range']*loss_range + \
               w_loss['polar']*loss_polar + w_loss['range_proj']*loss_range_img + \
               w_loss['polar_proj']*loss_polar_img

        d3_time.update(time.time()-d3)
        d4 = time.time()
        loss.backward()
        optimizer.step()
        d4_time.update(time.time()-d4)

        if collect:
            hist_list_img.update(get_hist(range_x, polar_x, proj_xy, train_grid, full_labels, num_pt, w_loss['range_proj'], w_loss['polar_proj']))
            hist_list.update(get_hist_frp(fusion_pred, range_pred, polar_pred, full_labels, num_pt))
            count += 1

        losses_range.update(w_loss['range'] * loss_range, batch_size)
        losses_polar.update(w_loss['polar'] * loss_polar, batch_size)
        losses_fusion.update(w_loss['fusion'] * loss_fusion, batch_size)
        losses_img.update(w_loss['range_proj']*loss_range_img + w_loss['polar_proj']*loss_polar_img, batch_size)

        batch_time.update(time.time() - start)
        start = time.time()

        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        iters = epoch * len(train_loader) + i
        if iters % report == 0:
            mp_logger(
                'Epoch: [{:03}/{}][{:04}/{}], Lr: {:.3e}, '
                'Data: {:.4f}({:.4f}), Batch: {:.3f}({:.3f}), '
                'Remain: {}, Fusion: {:.4f}, RV: {:.4f}, '
                'BEV: {:.4f}, Img: {:.4f}'.format(epoch, max_epoch,
                                                  i, len(train_loader)-1,
                                                  optimizer.param_groups[0]['lr'],
                                                  data_time.val, data_time.avg,
                                                  batch_time.val, batch_time.avg,
                                                  remain_time,
                                                  losses_fusion.avg, losses_range.avg,
                                                  losses_polar.avg, losses_img.avg)
            )
            if iters % (1*report) == 0 and dist.get_rank() == 0:
                writer.add_scalar('TRAIN/range_loss', losses_range.avg, iters)
                writer.add_scalar('TRAIN/polar_loss', losses_polar.avg, iters)
                writer.add_scalar('TRAIN/fusion_loss', losses_fusion.avg, iters)
                writer.add_scalar('TRAIN/img_loss', losses_img.avg, iters)

        if i % int(0.5*len(train_loader)) == 0 and i != 0:
            collect = False

        if count == report:
            catch_size = len(hist_list)
            iou, iou_range, iou_polar = hist_list.cal_iou()
            shows = display_iou([iou, iou_range, iou_polar],
                                ['fusion', 'range', 'polar'],
                                train_loader.dataset.get_xentropy_class_string,
                                avg='train_avg_iou_3d')
            mp_logger('\n' + shows)

            iou, iou_range, iou_polar = hist_list_img.cal_iou()
            shows = display_iou([iou, iou_range, iou_polar],
                                ['fusion', 'range', 'polar'],
                                train_loader.dataset.get_xentropy_class_string,
                                avg='train_avg_iou_proj')
            mp_logger('\n' + shows)
            ious.update(torch.mean(iou) * 100, catch_size)
            ious_range.update(torch.mean(iou_range) * 100, catch_size)
            ious_polar.update(torch.mean(iou_polar) * 100, catch_size)
            if dist.get_rank() == 0:
                writer.add_scalar('TRAIN/iou', ious.avg, iters)
                writer.add_scalar('TRAIN/iou_range', ious_range.avg, iters)
                writer.add_scalar('TRAIN/iou_polar', ious_polar.avg, iters)

            collect = False
            count = 0
            hist_list_img.reset()
            hist_list.reset()

        if lr_scheduler is not None:
            lr_scheduler.step()

def validate(epoch, val_loader, model, criterion, w_loss, writer):
    losses_range = AverageMeter()
    losses_polar = AverageMeter()
    losses_fusion = AverageMeter()
    losses_img = AverageMeter()
    model.eval()

    hist_list_img = RPF_Hist()
    hist_list = RPF_Hist()
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for i, (range_data, polar_data, r2p_matrix, p2r_matrix, knns) in enumerate(val_loader):
            batch_size = r2p_matrix.shape[0]
            in_vol, proj_mask, proj_labels, _, proj_xy, unproj_labels, path_seq, path_name, pxpy_range, _, _, _, _, _, points, _, _, _ = range_data
            _, vox_label, train_grid, full_labels, pt_fea, pxpypz_polar, num_pt = polar_data

            proj_labels = uint8_trick(proj_labels)
            vox_label = uint8_trick(vox_label)
            unproj_labels = uint8_trick(unproj_labels)

            in_vol = in_vol.cuda(non_blocking=True)
            proj_labels = proj_labels.long().cuda(non_blocking=True)
            train_grid_2d = train_grid[:, :, :2].cuda(non_blocking=True)
            pt_fea = pt_fea.cuda(non_blocking=True)
            vox_label = vox_label.long().cuda(non_blocking=True)

            r2p_matrix = r2p_matrix.cuda(non_blocking=True)
            p2r_matrix = p2r_matrix[:, :, :, :2].cuda(non_blocking=True)

            unproj_labels = unproj_labels.long().cuda(non_blocking=True)
            pxpy_range = torch.flip(pxpy_range.cuda(non_blocking=True), dims=[-1])    # because for F.grid_sample, i,j,k index w,h,d (i.e., reverse order)
            pxpypz_polar = torch.flip(pxpypz_polar.cuda(non_blocking=True), dims=[-1])
            points = points.cuda(non_blocking=True)
            knns = knns.cuda(non_blocking=True)

            fusion_pred, range_pred, polar_pred, range_x, polar_x = model(
                    in_vol, pt_fea, train_grid_2d, num_pt, r2p_matrix, p2r_matrix, pxpy_range, pxpypz_polar, points, knns)
            loss_fusion = criterion(fusion_pred, unproj_labels)
            loss_range = criterion(range_pred, unproj_labels)
            loss_polar = criterion(polar_pred, unproj_labels)

            loss_range_img = criterion(range_x, proj_labels) + lovasz_softmax(torch.nn.functional.softmax(range_x, dim=1), proj_labels, ignore=255)
            loss_polar_img = criterion(polar_x, vox_label) + lovasz_softmax(torch.nn.functional.softmax(polar_x, dim=1), vox_label, ignore=255)
            loss = w_loss['fusion']*loss_fusion + w_loss['range']*loss_range + \
                   w_loss['polar']*loss_polar + w_loss['range_proj']*loss_range_img + \
                   w_loss['polar_proj']*loss_polar_img


            hist_list_img.update(get_hist(range_x, polar_x, proj_xy, train_grid, full_labels, num_pt, w_loss['range_proj'], w_loss['polar_proj']))
            hist_list.update(get_hist_frp(fusion_pred, range_pred, polar_pred, full_labels, num_pt))

            losses_range.update(w_loss['range'] * loss_range, batch_size)
            losses_polar.update(w_loss['polar'] * loss_polar, batch_size)
            losses_fusion.update(w_loss['fusion'] * loss_fusion, batch_size)
            losses_img.update(w_loss['range_proj']*loss_range_img + w_loss['polar_proj']*loss_polar_img, batch_size)
            pbar.update(1)
        pbar.close()

        dist.all_reduce(losses_range.avg/dist.get_world_size())
        dist.all_reduce(losses_polar.avg/dist.get_world_size())
        dist.all_reduce(losses_fusion.avg/dist.get_world_size())
        dist.all_reduce(losses_img.avg/dist.get_world_size())
        mp_logger('epoch {:03}, fusion loss: {:.5f}, range loss: {:.5f}, polar loss: {:.5f}, img loss: {:.5f}'.format(
          epoch, losses_fusion.avg.item(), losses_range.avg.item(), losses_polar.avg.item(), losses_img.avg.item()))

        hist_list_img.all_reduce()
        iou_, iou_range, iou_polar = hist_list_img.get_iou()

        shows = display_iou([iou_, iou_range, iou_polar],
                            ['fusion', 'range', 'polar'],
                            val_loader.dataset.get_xentropy_class_string,
                            avg='val_avg_iou_proj')
        mp_logger('\n'+shows)

        hist_list.all_reduce()
        iou, iou_range, iou_polar = hist_list.get_iou()

        shows = display_iou([iou, iou_range, iou_polar],
                            ['fusion', 'range', 'polar'],
                            val_loader.dataset.get_xentropy_class_string,
                            avg='val_avg_iou_3d')
        mp_logger('\n'+shows)
        val_miou = max(torch.mean(iou), torch.mean(iou_), torch.mean(iou_range), torch.mean(iou_polar)) * 100
        val_miou_range = torch.mean(iou_range) * 100
        val_miou_polar = torch.mean(iou_polar) * 100
        if dist.get_rank() == 0 and (writer is not None):
            writer.add_scalar('VAL/range_loss', losses_range.avg, epoch)
            writer.add_scalar('VAL/polar_loss', losses_polar.avg, epoch)
            writer.add_scalar('VAL/fusion_loss', losses_fusion.avg, epoch)
            writer.add_scalar('VAL/img_loss', losses_img.avg, epoch)
            writer.add_scalar('VAL/IOU', val_miou, epoch)
            writer.add_scalar('VAL/IOU_Range', val_miou_range, epoch)
            writer.add_scalar('VAL/IOU_Polar', val_miou_polar, epoch)

    return val_miou
