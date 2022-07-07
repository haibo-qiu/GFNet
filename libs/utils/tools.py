import os
import sys
import yaml
import torch
import shutil
import random
import logging
import datetime
import numpy as np
import numba as nb
import torch.distributed as dist
from prettytable import PrettyTable
import pdb

def fast_hist(pred, label, n=19):
    k = (label >= 0) & (label < n)
    bin_count=torch.bincount(
        n * label[k] + pred[k], minlength=n ** 2)
    return bin_count[:n ** 2].reshape(n, n)

def uint8_trick(labels):
    # delete 0 label to make the training more easier
    labels = labels.type(torch.uint8)
    return labels - 1

def recover_uint8_trick(labels):
    labels = labels.reshape((-1)).astype(np.int32)
    return labels + 1

def mp_logger(meg, name='main-logger'):
    try:
        if dist.get_rank() == 0:
            logger = logging.getLogger(name)
            logger.info(meg)
    except:
        print(meg)

def load_arch_cfg(arch_cfg):
    # open arch config file
    try:
        print("Opening arch config file %s" % arch_cfg)
        ARCH = yaml.safe_load(open(arch_cfg, 'r'))
        return ARCH
    except Exception as e:
        print(e)
        print("Error opening arch yaml file.")
        quit()

def load_data_cfg(data_cfg):
    # open data config file
    try:
        print("Opening data config file %s" % data_cfg)
        DATA = yaml.safe_load(open(data_cfg, 'r'))
        return DATA
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()

def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def create_eval_log(log_path, data_cfg, name='main-logger'):
    assert os.path.exists(log_path)

    if 'logs' not in log_path:
        if 'kitti' in data_cfg:
            data_name = 'semantickitti'
        elif 'nuscenes' in data_cfg:
            data_name = 'nuscenes'
        else:
            raise NotImplemented
        log_path = os.path.join('logs', data_name, 'eval', datetime.datetime.now().strftime("%Y-%-m-%d-%H-%M-%S"))
        os.makedirs(log_path, exist_ok=True)

    # create logger
    log_file = os.path.join(log_path, 'eval.log')
    fmt = "[%(asctime)s] %(message)s"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # add file handler to save the log to file
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter(fmt))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # add console handler to output log on screen
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(fmt))
    sh.setLevel(logging.DEBUG)
    logger.addHandler(sh)
    logger.propagate = False

    return logger, log_path

def create_log(log_path, data_cfg, debug, name='main-logger'):
    if debug:
        log_path = log_path.replace('logs', 'debug')
        log_path = os.path.join('logs', log_path)

    if 'kitti' in data_cfg:
        data_name = 'semantickitti'
    elif 'nuscenes' in data_cfg:
        data_name = 'nuscenes'
    else:
        raise NotImplemented

    log_path = log_path.replace('logs', data_name)
    log_path = os.path.join('logs', log_path)

    # create log folder
    try:
        if os.path.isdir(log_path):
            shutil.rmtree(log_path)
        os.makedirs(log_path)
    except Exception as e:
        print(e)
        print("Error creating log directory. Check permissions!")
        quit()

    # create logger
    log_file = os.path.join(log_path, 'training.log')
    fmt = "[%(asctime)s] %(message)s"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # add file handler to save the log to file
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter(fmt))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # add console handler to output log on screen
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(fmt))
    sh.setLevel(logging.DEBUG)
    logger.addHandler(sh)
    logger.propagate = False

    # create tensorboard folder
    tb_dir = os.path.join(log_path, 'tensorboard')
    os.makedirs(tb_dir, exist_ok=True)

    return logger, log_path, tb_dir

def save_checkpoint(state, save_path, is_best, filename='checkpoint.pth.tar', test=True):
    # if state['epoch'] in [280, 290] and test:
        # filename = '{}_{}'.format(state['epoch'], filename)
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        state_info = {
            'epoch': state['epoch'],
            'best_iou': state['best_iou'],
            'model': state['model']
        }
        torch.save(state_info, os.path.join(save_path, 'model_best.pth.tar'))
        mp_logger('save model with current best iou:{:.4f}'.format(state['best_iou']))

def load_part_params(model, trained_params):
    params = dict(model.state_dict())

    for k, v in params.items():
        new_k = 'backbone.' + k
        if new_k in trained_params:
            params[k].data.copy_(trained_params[new_k].data)
    model.load_state_dict(params)
    return model

def load_pretrained(pretrained, model):
    epoch = best_iou = 0
    # does model folder exist?
    if os.path.isfile(pretrained):
        mp_logger("pretrained model exists! Using model from %s" % (pretrained))
        checkpoint = torch.load(pretrained, map_location=lambda storage, loc: storage.cuda())
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['model'])
        # epoch = checkpoint['epoch']
        best_iou = checkpoint['best_iou']
    else:
        mp_logger("pretrained model doesnt exist! Start with random weights...")
    return model, epoch, best_iou

def resume_training(resume, model, optimizer):
    epoch = best_iou = 0
    # does model folder exist?
    if os.path.isfile(resume):
        mp_logger("model exists! Resuming model from %s" % (resume))
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage.cuda())
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch'] + 1
        best_iou = checkpoint['best_iou']
        mp_logger("start from epoch: {}, with current best iou: {}".format(epoch, best_iou))
    else:
        mp_logger("model doesnt exist! Start with random weights...")
    return model, optimizer, epoch, best_iou

def recording_cfg(arch_cfg, data_cfg, log_path):

    # copy all files to log folder (to remember what we did, and make inference
    # easier). Also, standardize name to be able to open it later
    pwd = os.getcwd()
    ignores = ['.git', '.gitignore', 'debug', 'dataset', 'logs', 'pretrained', '__pycache__']
    valid_ext = ['.py', '.sh', '.ply', '.yaml']
    try:
        print("Copying files to %s for further reference." % log_path)
        shutil.copyfile(arch_cfg, os.path.join(log_path, "arch_cfg.yaml"))
        shutil.copyfile(data_cfg, os.path.join(log_path, "data_cfg.yaml"))

        log_path = os.path.join(log_path, "codes")
        for v in sorted(os.listdir(pwd)):
            if v not in ignores:
                if not os.path.isdir(os.path.join(pwd, v)):
                    if any(v.endswith(ext) for ext in valid_ext) and 'debug' not in v:
                        os.makedirs(log_path, exist_ok=True)
                        shutil.copyfile(os.path.join(pwd, v), os.path.join(log_path, v))
                else:
                    for dp, dn, fn in os.walk(os.path.join(pwd, v)):
                        for f in fn:
                            if any(f.endswith(ext) for ext in valid_ext):
                                filename = os.path.join(dp, f)
                                os.makedirs(os.path.dirname(filename.replace(pwd, log_path)), exist_ok=True)
                                shutil.copyfile(filename, filename.replace(pwd, log_path))
    except Exception as e:
        print(e)
        print("Error copying files, check permissions. Exiting...")
        quit()

def vis_diff_projection(FLAGS, DATA, ARCH, name):
    import parser
    from PIL import Image
    Dataset = parser.Parser(root=FLAGS.dataset,
                            train_sequences=DATA["split"]["train"],
                            valid_sequences=DATA["split"]["valid"],
                            test_sequences=None,
                            labels=DATA["labels"],
                            color_map=DATA["color_map"],
                            learning_map=DATA["learning_map"],
                            learning_map_inv=DATA["learning_map_inv"],
                            sensor=ARCH["dataset"]["sensor"],
                            max_points=ARCH["dataset"]["max_points"],
                            batch_size=ARCH["train"]["batch_size"],
                            workers=ARCH["train"]["workers"],
                            gt=True,
                            shuffle_train=True)
    train_loader = Dataset.get_train_set()
    for i, (proj, proj_mask, proj_labels, proj_color, _, _, _, _, _, _, _, _, _, _, _, _) in enumerate(train_loader):
        proj_color = proj_color.numpy().squeeze()
        proj_img = Image.fromarray((proj_color * 255).astype(np.uint8))
        os.makedirs(name, exist_ok=True)
        save_name = os.path.join(name, '{:02}.jpg'.format(i))
        print(save_name)
        proj_img.save(save_name)

        if i >= 49:
            break

def vis_proj_v1_v2(root, loader, phase='val'):
    from PIL import Image
    os.makedirs(os.path.join(root, 'images'), exist_ok=True)
    valid_rates = []
    for i, (range_data, polar_data, r2p_matrix, p2r_matrix, knns) in enumerate(loader):
        print(i, '/', len(loader))
        proj_mask = range_data[1]
        proj_color = range_data[3]
        proj_color = proj_color.numpy().squeeze()
        # PDB_MP().set_trace()
        rate = proj_mask.sum() / proj_mask.numel()
        valid_rates.append(rate.item())

        if phase == 'val':
            proj_img = Image.fromarray((proj_color * 255).astype(np.uint8))
            save_name = os.path.join(os.path.join(root, 'images'), '{:04}.jpg'.format(i))
            # print(save_name)
            proj_img.save(save_name)
        # in_vol, proj_mask, proj_labels, proj_color, proj_xy, unproj_labels, path_seq, path_name, pxpy_range, _, _, _, _, _, points, _, _, real_num_pt = range_data
    mean_rate = np.mean(valid_rates)
    print(mean_rate)

def vis_range_view(data_loader, root='temp/' + 'v2_1hres_0flip_1trans_0rot'):
    from PIL import Image
    for i, (range_data, polar_data, r2p_matrix, p2r_matrix, knns) in enumerate(data_loader):
        proj_color = range_data[3]
        proj_color = proj_color[0].numpy().squeeze()
        # in_vol, proj_mask, proj_labels, proj_color, proj_xy, _, path_seq, path_name, _, _, _, _, _, _, _, _, _ = range_data
        # PDB_MP().set_trace()
        proj_img = Image.fromarray((proj_color * 255).astype(np.uint8))
        os.makedirs(root, exist_ok=True)
        save_name = os.path.join(root, '{:02}.jpg'.format(i))
        print(save_name)
        proj_img.save(save_name)

        if i >= 49:
            break

def whether_aug(train, condition=None):
    if condition is not None:
        aug = train and condition and (random.random() > 0.5)
    else:
        aug = train and (random.random() > 0.5)
    return aug

@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])',nopython=True,cache=True,parallel = False)
def nb_process_label(processed_label,sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,),dtype = np.uint16)
    counter[sorted_label_voxel_pair[0,3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0,:3]
    for i in range(1,sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i,:3]
        if not np.all(np.equal(cur_ind,cur_sear_ind)):
            processed_label[cur_sear_ind[0],cur_sear_ind[1],cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,),dtype = np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i,3]] += 1
    processed_label[cur_sear_ind[0],cur_sear_ind[1],cur_sear_ind[2]] = np.argmax(counter)
    return processed_label

# transformation between Cartesian coordinates and polar coordinates
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:,0]**2 + input_xyz[:,1]**2)
    phi = np.arctan2(input_xyz[:,1],input_xyz[:,0])
    return np.stack((rho,phi,input_xyz[:,2]),axis=1)

def polar2cat(input_xyz_polar):
    x = input_xyz_polar[0]*np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0]*np.sin(input_xyz_polar[1])
    return np.stack((x,y,input_xyz_polar[2]),axis=0)

# weights for loss
def get_weight_per_class(epsilon_w, n_class, contents, mapping, ignore):
    content = torch.zeros(n_class, dtype=torch.float)
    for cl, freq in contents.items():
      x_cl = mapping[cl]  # map actual class to xentropy class
      content[x_cl] += freq
    loss_w = 1 / (content + epsilon_w)   # get weights
    for x_cl, w in enumerate(loss_w):  # ignore the ones necessary to ignore
      if ignore[x_cl]:
        # don't weigh
        loss_w[x_cl] = 0
    mp_logger("Loss weights from content: {}".format(loss_w.data))
    return loss_w

def display_iou(ious, names, class_string, avg='val_avg_iou'):
    shows = PrettyTable()
    info_dict = {}
    mean_ious = []
    for n, v in zip(names, ious):
        mean_ious.append(np.round(100*torch.mean(v).item(), 6))
        if n not in info_dict:
            info_dict[n] = []
        info_dict['classes'] = []
        for i, class_iou in enumerate(v):
            class_name = class_string(i+1)
            info_dict[n].append(np.round(100*class_iou.item(), 6))
            info_dict['classes'].append(class_name)
    for v in ['classes'] + names:
        shows.add_column(v, info_dict[v])
    shows.add_row([avg] + mean_ious)
    shows.align = 'r'
    return shows.get_string()

def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class RPF_Hist(object):
    """
    simultaneously store range-view polar-view and fusion view hist
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.range = []
        self.polar = []
        self.fusion = []

    def append(self, val_r, val_p, val_f):
        self.range.append(val_r)
        self.polar.append(val_p)
        self.fusion.append(val_f)

    def update(self, hist):
        self.range += hist.range
        self.polar += hist.polar
        self.fusion += hist.fusion

    def all_reduce(self):
        self.range_sum = sum(self.range)
        self.polar_sum = sum(self.polar)
        self.fusion_sum = sum(self.fusion)
        dist.all_reduce(self.range_sum)
        dist.all_reduce(self.polar_sum)
        dist.all_reduce(self.fusion_sum)

    def get_iou(self):
        iou = self.per_class_iou(self.fusion_sum)
        iou_range = self.per_class_iou(self.range_sum)
        iou_polar = self.per_class_iou(self.polar_sum)
        return iou, iou_range, iou_polar

    def per_class_iou(self, h_list):
        return torch.diag(h_list) / (h_list.sum(1) + h_list.sum(0) - torch.diag(h_list) + 1e-15)

    def cal_iou(self):
        iou = self.per_class_iou(sum(self.fusion))
        iou_range = self.per_class_iou(sum(self.range))
        iou_polar = self.per_class_iou(sum(self.polar))
        return iou, iou_range, iou_polar

    def __len__(self):
        assert len(self.range) == len(self.polar) == len(self.fusion)
        return len(self.fusion)

class PDB_MP(pdb.Pdb):
    """
    Borrowed from https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess
    """

    _original_stdin_fd = sys.stdin.fileno()
    _original_stdin = None

    def __init__(self):
        pdb.Pdb.__init__(self, nosigint=True)

    def _cmdloop(self):
        current_stdin = sys.stdin
        try:
            if not self._original_stdin:
                self._original_stdin = os.fdopen(self._original_stdin_fd)
            sys.stdin = self._original_stdin
            self.cmdloop()
        finally:
            sys.stdin = current_stdin
