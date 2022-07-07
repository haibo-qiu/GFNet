import os
import sys
sys.path.append(os.getcwd())

import random
import pickle
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from scipy.spatial.ckdtree import cKDTree as kdtree
from libs.utils.tools import cart2polar, polar2cat, nb_process_label, mp_logger, whether_aug, vis_range_view
from libs.utils.laserscan import LaserScan, SemLaserScan

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits

class Nuscenes(Dataset):
    def __init__(self, pkl_path,
                 data_path,
                 labels,
                 range_cfg,         # configs for range view (dict)
                 polar_cfg,         # configs for polar view (dict)
                 dataset_cfg,
                 color_map,
                 learning_map,
                 max_volume_space=[50, np.pi, 3],
                 min_volume_space=[0, -np.pi, -5],
                 ignore_label=0,
                 version='v1.0-trainval',
                 split='train',
                 return_ref=True,
                 knn=True):
        assert version in ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
        if version == 'v1.0-trainval':
            train_scenes = splits.train
            val_scenes = splits.val
        elif version == 'v1.0-test':
            train_scenes = splits.test
            val_scenes = []
        elif version == 'v1.0-mini':
            train_scenes = splits.mini_train
            val_scenes = splits.mini_val
        else:
            raise NotImplementedError
        self.split = split
        self.train = (split=='train')
        self.data_path = data_path
        self.return_ref = return_ref
        self.polar = polar_cfg
        self.range = range_cfg
        self.dataset = dataset_cfg
        self.color_map = color_map
        self.max_volume_space=max_volume_space
        self.min_volume_space=min_volume_space
        self.ignore_label = ignore_label
        self.knn = knn
        self.neighbor = 7
        self.cal_valid = []

        self.labels = labels
        self.nclasses = len(labels)

        self.learning_map = learning_map

        with open(pkl_path, 'rb') as fp:
            self.samples_annots = pickle.load(fp)

        self.samples = [v.split('**')[0] for v in self.samples_annots]
        self.annotations = [v.split('**')[1] for v in self.samples_annots]

        mp_logger('loading {} set with {} samples'.format(split, len(self.samples)))

    def get_xentropy_class_string(self, idx):
        return self.labels[idx]

    def change_split(self, s):
        assert s in ['train', 'val']
        self.split = s
        mp_logger('change split to {}'.format(s))

    def write_txt(self):
        # for sample
        mp_logger('write to train.pkl...')
        with open('train.pkl', 'wb') as fp:
            pickle.dump(self.samples_annots, fp)

    def reset(self, s=None):
        if s:
            self.change_split(s)

        if self.split == 'train':
            self.token_list = self.train_token_list
        elif self.split == 'val':
            self.token_list = self.val_token_list
        elif self.split == 'test':
            self.token_list = self.train_token_list

    def range_dataset(self, scan_points, scan_file, label, label_file):
        self.sensor_img_means = torch.tensor(self.range['sensor_img_means'],
                                             dtype=torch.float32)
        self.sensor_img_stds = torch.tensor(self.range['sensor_img_stds'],
                                            dtype=torch.float32)
        scan = SemLaserScan(sem_color_dict=self.color_map,
                            train=self.train,
                            project=True,
                            H=self.range['sensor_img_H'],
                            W=self.range['sensor_img_W'],
                            fov_up=self.range['sensor_fov_up'],
                            fov_down=self.range['sensor_fov_down'],
                            proj_version=self.range['proj'],
                            hres=self.range['hres'],
                            factor=self.range['factor'],
                            points_to_drop=self.points_to_drop,
                            flip=self.range['flip'],
                            trans=self.range['trans'],
                            rot=self.range['rot'])

        # open and obtain scan
        scan.open_scan(scan_points)
        if self.split != 'test':
            scan.set_label_nuscenes(label[scan.valid].reshape((-1)))

        scan.max_points = self.range['max_points']
        unproj_n_points = scan.points.shape[0]
        unproj_xyz = torch.full((scan.max_points, 3), -1.0, dtype=torch.float)
        unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)
        unproj_range = torch.full([scan.max_points], -1.0, dtype=torch.float)
        unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
        unproj_remissions = torch.full([scan.max_points], -1.0, dtype=torch.float)
        unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)

        if self.split != 'test':
            unproj_labels = torch.full([scan.max_points], 0, dtype=torch.int32)
            unproj_labels[:unproj_n_points] = torch.from_numpy(scan.sem_label)
            unproj_labels = unproj_labels.unsqueeze(0)
        else:
            unproj_labels = []

        # get points and labels
        proj_range = torch.from_numpy(scan.proj_range).clone()
        proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
        proj_remission = torch.from_numpy(scan.proj_remission).clone()
        proj_mask = torch.from_numpy(scan.proj_mask)
        if self.split != 'test':
            proj_labels = torch.from_numpy(scan.proj_sem_label).clone()
            proj_labels = proj_labels * proj_mask
            proj_color = torch.from_numpy(scan.proj_sem_color).clone()
        else:
            proj_labels = []
            proj_color = []
        if scan.crop and self.train:
            new_w = scan.new_w
        else:
            new_w = scan.sizeWH[0]

        proj_x = torch.full([scan.max_points], 0, dtype=torch.float)
        proj_x[:unproj_n_points] = torch.from_numpy(2 * (scan.proj_x/(new_w-1) - 0.5))
        proj_y = torch.full([scan.max_points], 0, dtype=torch.float)
        proj_y[:unproj_n_points] = torch.from_numpy(2 * (scan.proj_y/(scan.sizeWH[1]-1) - 0.5))
        proj_yx = torch.stack([proj_y, proj_x], dim=1)[None, :, :]

        proj = torch.cat([proj_range.unsqueeze(0).clone(),
                        proj_xyz.clone().permute(2, 0, 1),
                        proj_remission.unsqueeze(0).clone()])
        proj = (proj - self.sensor_img_means[:, None, None]
                        ) / self.sensor_img_stds[:, None, None]
        proj = proj * proj_mask.float()

        # get name and sequence, to be changed
        path_norm = os.path.normpath(label_file)
        path_split = path_norm.split(os.sep)
        path_seq = path_split[-2]
        # path_name = path_split[-1].replace(".bin", ".label")
        path_name = path_split[-1]

        proj_idx = torch.from_numpy(scan.proj_idx).clone()

        points2pixel = np.concatenate((scan.proj_y.reshape(-1, 1), scan.proj_x.reshape(-1, 1)), axis=1)
        points2pixel = torch.from_numpy(points2pixel).long()
        full_p2p = torch.full((scan.max_points, points2pixel.shape[-1]), -1, dtype=torch.long)
        full_p2p[:unproj_n_points] = points2pixel

        points2pixel = proj_yx[0, :unproj_n_points, :].float()

        data_tuple = (proj, proj_mask, proj_labels, proj_color, full_p2p,
                      unproj_labels, path_seq, path_name, proj_yx, proj_x, proj_y, proj_range, unproj_range,
                      proj_xyz, unproj_xyz, proj_remission, unproj_remissions, unproj_n_points)

        return data_tuple, points2pixel, proj_idx, scan.valid, scan.max_points

    def polar_dataset(self, index, scan, labels):
        'Generates one sample of data'

        # put in attribute
        xyz = scan[:, 0:3]    # get xyz
        remissions = scan[:, 3]  # get remission
        # remissions = np.squeeze(remissions)
        num_pt = xyz.shape[0]

        # data aug
        if whether_aug(self.train):
            xyz = self.polar_data_aug(xyz)

        # if self.points_to_drop is not None:
            # labels = np.delete(labels, self.points_to_drop, axis=0)
        if self.valid.sum() == 0:
            labels = np.zeros_like(labels)[:10000]
        else:
            labels = labels[self.valid]

        # convert coordinate into polar coordinates
        xyz_pol = cart2polar(xyz)

        if self.polar['fixed_volume_space']:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)
        else:
            max_bound_r = np.percentile(xyz_pol[:,0],100,axis = 0)
            min_bound_r = np.percentile(xyz_pol[:,0],0,axis = 0)
            max_bound = np.max(xyz_pol[:,1:],axis = 0)
            min_bound = np.min(xyz_pol[:,1:],axis = 0)
            max_bound = np.concatenate(([max_bound_r],max_bound))
            min_bound = np.concatenate(([min_bound_r],min_bound))

        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = np.array(self.polar['grid_size'])
        intervals = crop_range/(cur_grid_size-1)

        if (intervals==0).any():
            mp_logger("Zero interval!")
        grid_ind = (np.floor((np.clip(xyz_pol,min_bound,max_bound)-min_bound)/intervals)).astype(np.int)
        pxpypz = (np.clip(xyz_pol,min_bound,max_bound)-min_bound) / crop_range
        pxpypz = 2 * (pxpypz- 0.5)

        # process voxel position
        voxel_position = np.zeros(self.polar['grid_size'],dtype = np.float32)
        dim_array = np.ones(len(self.polar['grid_size'])+1,int)
        dim_array[0] = -1
        voxel_position = np.indices(self.polar['grid_size'])*intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        # voxel_position = polar2cat(voxel_position)

        # process labels
        processed_label = np.ones(self.polar['grid_size'],dtype = np.uint8)*self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind,labels],axis = 1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:,0],grid_ind[:,1],grid_ind[:,2])),:]
        processed_label = nb_process_label(np.copy(processed_label),label_voxel_pair)
        # data_tuple = (voxel_position,processed_label)

        # prepare visiblity feature
        # find max distance index in each angle,height pair
        valid_label = np.zeros_like(processed_label,dtype=bool)
        valid_label[grid_ind[:,0],grid_ind[:,1],grid_ind[:,2]] = True
        valid_label = valid_label[::-1]
        max_distance_index = np.argmax(valid_label,axis=0)
        max_distance = max_bound[0]-intervals[0]*(max_distance_index)
        distance_feature = np.expand_dims(max_distance, axis=2)-np.transpose(voxel_position[0],(1,2,0))
        distance_feature = np.transpose(distance_feature,(1,2,0))
        # convert to boolean feature
        distance_feature = (distance_feature>0)*-1.
        distance_feature[grid_ind[:,2],grid_ind[:,0],grid_ind[:,1]]=1.

        distance_feature = torch.from_numpy(distance_feature)
        processed_label = torch.from_numpy(processed_label)

        data_tuple = (distance_feature,processed_label)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5)*intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = np.concatenate((return_xyz,xyz_pol,xyz[:,:2]),axis = 1)

        return_fea = np.concatenate((return_xyz,remissions[...,np.newaxis]),axis = 1)

        # order in decreasing z
        pixel2point = np.full(shape=self.polar['grid_size'], fill_value=-1, dtype=np.float32)
        indices = np.arange(xyz_pol.shape[0])
        order = np.argsort(xyz_pol[:, 2])
        grid_ind_order = grid_ind[order].copy()
        indices = indices[order]
        pixel2point[grid_ind_order[:, 0], grid_ind_order[:, 1], grid_ind_order[:, 2]] = indices

        pixel2point = torch.from_numpy(pixel2point)

        return_fea = torch.from_numpy(return_fea)
        grid_ind = torch.from_numpy(grid_ind)
        labels = torch.from_numpy(labels)

        full_return_fea = torch.full((self.max_points, return_fea.shape[-1]), -1.0, dtype=torch.float)
        full_return_fea[:return_fea.shape[0]] = return_fea

        full_grid_ind = torch.full((self.max_points, grid_ind.shape[-1]), -1, dtype=torch.long)
        full_grid_ind[:grid_ind.shape[0]] = grid_ind

        full_pxpypz = torch.full((self.max_points, pxpypz.shape[-1]), 0, dtype=torch.float)
        full_pxpypz[:pxpypz.shape[0]] = torch.from_numpy(pxpypz)
        full_pxpypz = full_pxpypz[None, None, :, :]

        full_labels = torch.full((self.max_points, labels.shape[-1]), -1, dtype=torch.long)
        full_labels[:labels.shape[0]] = labels
        full_labels = full_labels.permute(1, 0)

        if self.polar['return_test']:
            data_tuple += (full_grid_ind,full_labels,full_return_fea,index, full_pxpypz, num_pt)
        else:
            data_tuple += (full_grid_ind,full_labels,full_return_fea, full_pxpypz, num_pt)
        return data_tuple, torch.from_numpy(pxpypz).float(), pixel2point

    def __getitem__(self, index):

        lidar_path = os.path.join(self.data_path, self.samples[index])
        raw_data = np.fromfile(lidar_path, dtype = np.float32).reshape((-1, 5))
        scan = raw_data[:, :4]

        if self.split == 'test':
            lidarseg_path = os.path.join(self.data_path, self.annotations[index]) # not exist, only for prediction
            label = np.expand_dims(np.zeros_like(scan[:,0],dtype=int),axis=1)
        else:
            lidarseg_path = os.path.join(self.data_path, self.annotations[index])
            annotated_data = np.fromfile(lidarseg_path, dtype=np.uint8).reshape((-1,1))
            label = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        # whether drop some points
        self.points_to_drop = None
        if whether_aug(self.train, self.dataset['drop']):
            self.drop_points = random.uniform(0, self.dataset['drop_rate'])
            self.points_to_drop = np.random.randint(0, len(scan)-1, int(len(scan) * self.drop_points))
            scan = np.delete(scan, self.points_to_drop, axis=0)
            label = np.delete(label, self.points_to_drop, axis=0)
        # scale aug
        if whether_aug(self.train, self.dataset['scale']):
            factor = np.random.uniform(1-self.dataset['scale_rate'], 1+self.dataset['scale_rate'])
            scan[:, :3] *= factor

        range_data, range_point2pixel, range_pixel2point, valid, max_points = self.range_dataset(scan, lidar_path, label, lidarseg_path)

        self.cal_valid.append(valid.sum())

        if valid.sum() == 0:
            scan = np.zeros_like(scan)[:10000]
        else:
            scan = scan[valid]
        # when training due to the order of index changed, we need to re-cal knn
        if self.knn:
            tree = kdtree(scan[:, :3])
            _, knns = tree.query(scan[:, :3], k=self.neighbor)
        self.valid = valid
        self.max_points = max_points

        polar_data, polar_point2pixel, polar_pixel2point = self.polar_dataset(index, scan, label)

        r2p_flow_matrix = self.r2p_flow_matrix(polar_pixel2point, range_point2pixel)
        p2r_flow_matrix = self.p2r_flow_matrix(range_pixel2point, polar_point2pixel)

        knns_full = torch.full((max_points, self.neighbor), 0, dtype=torch.long)
        knns_full[:knns.shape[0]] = torch.from_numpy(knns).long()

        return range_data, polar_data, r2p_flow_matrix, p2r_flow_matrix, knns_full

    def p2r_flow_matrix(self, range_idx, polar_idx):
        """
        range_idx: [H, W] indicates the location of each range pixel on point clouds
        polar_idx: [N, 3] indicates the location of each points on polar grids
        """
        H, W = range_idx.shape
        N, K = polar_idx.shape
        flow_matrix = torch.full(size=(H, W, K), fill_value=-10, dtype=torch.float)
        if self.valid.sum() == 0:
            return flow_matrix

        valid_idx = torch.nonzero(range_idx+1).transpose(0, 1)
        valid_value = range_idx[valid_idx[0], valid_idx[1]].long()
        flow_matrix[valid_idx[0], valid_idx[1], :] = polar_idx[valid_value, :]
        return flow_matrix

    def r2p_flow_matrix(self, polar_idx, range_idx):
        """
        polar_idx: [H, W, C] indicates the location of each range pixel on point clouds
        range_idx: [N, 2] indicates the location of each points on polar grids
        """
        H, W, C = polar_idx.shape
        N, K = range_idx.shape
        flow_matrix = torch.full(size=(H, W, C, K), fill_value=-10, dtype=torch.float) # smaller than -1 to trigger the zero padding of grid_sample
        if self.valid.sum() == 0:
            return flow_matrix

        valid_idx = torch.nonzero(polar_idx+1).transpose(0, 1)
        valid_value = polar_idx[valid_idx[0], valid_idx[1], valid_idx[2]].long()
        flow_matrix[valid_idx[0], valid_idx[1], valid_idx[2], :] = range_idx[valid_value, ]
        return flow_matrix

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.samples)

    def polar_data_aug(self, xyz):
        # random data augmentation by rotation
        if whether_aug(self.train, self.polar['rotate_aug']):
            rotate_rad = np.deg2rad(np.random.random()*360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:,:2] = np.dot( xyz[:,:2],j)

        # random data augmentation by flip x , y or x+y
        if whether_aug(self.train, self.polar['flip_aug']):
            flip_type = np.random.choice(4,1)
            if flip_type==1:
                xyz[:,0] = -xyz[:,0]
            elif flip_type==2:
                xyz[:,1] = -xyz[:,1]
            elif flip_type==3:
                xyz[:,:2] = -xyz[:,:2]

        # z axis noise
        if whether_aug(self.train, self.polar['noise_aug']):
            noise = np.random.normal(0, 0.2, size=xyz.shape[0])
            xyz[:, -1] += noise

        return xyz

if __name__ == '__main__':
    import yaml
    import pdb
    data_path = 'dataset/nuScenes/full/'

    arch_cfg = 'configs/resnet_nuscenes.yaml'
    ARCH = yaml.safe_load(open(arch_cfg, 'r'))

    data_cfg = 'configs/nuscenes.yaml'
    DATA = yaml.safe_load(open(data_cfg, 'r'))
    dataset = Nuscenes(pkl_path='dataset/nuScenes/nuscenes_train.pkl',
                       data_path=data_path,
                       labels=DATA['labels_16'],
                       range_cfg=ARCH['range'],
                       polar_cfg=ARCH['polar'],
                       dataset_cfg=ARCH['dataset'],
                       color_map=DATA['color_map'],
                       learning_map=DATA['learning_map'],
                      )
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=4,
                                               num_workers=8,
                                               shuffle=False,
                                               pin_memory=True,
                                               drop_last=True)

    # for i, (range_data, polar_data, r2p_matrix, p2r_matrix, knns) in enumerate(train_loader):
        # print(i, len(train_loader))
        # in_vol, proj_mask, proj_labels, _, proj_xy, unproj_labels, path_seq, path_name, pxpy_range, _, _, _, _, _, points, _, _, real_num_pt = range_data
        # _, vox_label, train_grid, full_labels, pt_fea, pxpypz_polar, num_pt = polar_data
        # if i > 20:
            # break

    vis_range_view(train_loader, root='temp/vis_nuscenes')
