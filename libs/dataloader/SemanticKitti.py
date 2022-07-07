import os
import sys
sys.path.append(os.getcwd())

import random
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.ckdtree import cKDTree as kdtree
from libs.utils.tools import cart2polar, polar2cat, nb_process_label, mp_logger, whether_aug 
from libs.utils.laserscan import LaserScan, SemLaserScan

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']

def is_scan(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)

def is_label(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)


class SemanticKitti(Dataset):

    def __init__(self, root,        # directory where data is
                 sequences,         # sequences for this data (e.g. [1,3,4,6])
                 labels,                # label dict: (e.g 10: "car")
                 color_map,         # colors dict bgr (e.g 10: [255, 0, 0])
                 learning_map,  # classes to learn (0 to N-1 for xentropy)
                 learning_map_inv,      # inverse of previous (recover labels)
                 range_cfg,         # configs for range view (dict)
                 polar_cfg,         # configs for polar view (dict)
                 dataset_cfg,
                 max_volume_space=[50, np.pi, 1.5],
                 min_volume_space=[3, -np.pi, -3],
                 ignore_label=0,
                 gt=True,
                 knn=True):          # send ground truth?
        # save deats
        self.root = os.path.join(root, "sequences")
        self.sequences = sequences
        self.train = (0 in sequences)
        self.labels = labels
        self.color_map = color_map
        self.polar = polar_cfg
        self.range = range_cfg
        self.dataset = dataset_cfg
        self.learning_map = learning_map
        self.learning_map_inv = learning_map_inv
        self.max_volume_space=max_volume_space
        self.min_volume_space=min_volume_space
        self.ignore_label = ignore_label
        self.gt = gt
        self.knn = knn
        self.neighbor = 7

        self.nclasses = len(self.learning_map_inv)

        # make sure directory exists
        if os.path.isdir(self.root):
            mp_logger("Sequences folder exists! Using sequences from %s" % self.root)
        else:
            raise ValueError("Sequences folder doesn't exist! Exiting...")

        # make sure sequences is a list
        assert(isinstance(self.sequences, list))

        # placeholder for filenames
        self.scan_files = []
        self.label_files = []
        self.knn_files = []

        # fill in with names, checking that all sequences are complete
        for seq in self.sequences:
            # to string
            seq = '{0:02d}'.format(int(seq))

            mp_logger("parsing seq {}".format(seq))

            # get paths for each
            scan_path = os.path.join(self.root, seq, "velodyne")
            label_path = os.path.join(self.root, seq, "labels")
            knn_path = os.path.join(self.root, seq, "knns")

            # get files
            scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                    os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
            label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                    os.path.expanduser(label_path)) for f in fn if is_label(f)]
            knn_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                    os.path.expanduser(knn_path)) for f in fn if is_scan(f)]

            # check all scans have labels
            if self.gt:
                assert(len(scan_files) == len(label_files))
            if self.knn:
                assert(len(scan_files) == len(knn_files))

            # extend list
            self.scan_files.extend(scan_files)
            self.label_files.extend(label_files)
            self.knn_files.extend(knn_files)

        # sort for correspondance
        self.scan_files.sort()
        self.label_files.sort()
        self.knn_files.sort()

        mp_logger("Using {} scans from sequences {}".format(len(self.scan_files),
                                                        self.sequences))
    def __len__(self):
        return len(self.scan_files)
    
    def range_dataset(self, scan_points, scan_file, label_file):
        # open a semantic laserscan
        self.sensor_img_means = torch.tensor(self.range['sensor_img_means'],
                                             dtype=torch.float32)
        self.sensor_img_stds = torch.tensor(self.range['sensor_img_stds'],
                                            dtype=torch.float32)
        if self.gt:
            scan = SemLaserScan(self.color_map,
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
        else:
            scan = LaserScan(train=self.train,
                             project=True,
                             H=self.range['sensor_img_H'],
                             W=self.range['sensor_img_W'],
                             fov_up=self.range['sensor_fov_up'],
                             fov_down=self.range['sensor_fov_down'],
                             proj_version=self.range['proj'],
                             hres=self.range['hres'],
                             factor=self.range['factor'],
                             flip=self.range['flip'],
                             trans=self.range['trans'],
                             rot=self.range['rot'])

        # open and obtain scan
        scan.open_scan(scan_points)
        if self.gt:
            scan.open_label(label_file)
            # map unused classes to used classes (also for projection)
            # scan.sem_label = scan.sem_label[scan.valid]
            scan.sem_label = self.map(scan.sem_label, self.learning_map)
            scan.proj_sem_label = self.map(scan.proj_sem_label, self.learning_map)

        # make a tensor of the uncompressed data (with the max num points)
        unproj_n_points = scan.points.shape[0]
        unproj_xyz = torch.full((scan.max_points, 3), -1.0, dtype=torch.float)
        unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)
        unproj_range = torch.full([scan.max_points], -1.0, dtype=torch.float)
        unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
        unproj_remissions = torch.full([scan.max_points], -1.0, dtype=torch.float)
        unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)
        if self.gt:
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
        if self.gt:
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

        # get name and sequence
        path_norm = os.path.normpath(scan_file)
        path_split = path_norm.split(os.sep)
        path_seq = path_split[-3]
        path_name = path_split[-1].replace(".bin", ".label")

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

    def polar_dataset(self, index, scan, label_file):
        'Generates one sample of data'

        # put in attribute
        xyz = scan[:, 0:3]    # get xyz
        remissions = scan[:, 3]  # get remission
        # remissions = np.squeeze(remissions)
        num_pt = xyz.shape[0]

        # data aug
        if whether_aug(self.train):
            xyz = self.polar_data_aug(xyz)

        if self.gt:
            labels = np.fromfile(label_file, dtype=np.int32).reshape((-1, 1))
            labels = labels & 0xFFFF  # semantic label in lower half
            labels = self.map(labels, self.learning_map)
        else:
            labels = np.expand_dims(np.zeros_like(scan[:,0],dtype=int),axis=1)

        if self.points_to_drop is not None:
            labels = np.delete(labels, self.points_to_drop, axis=0)
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

    def __getitem__(self, index):
        # get item in tensor shape
        label_file = None
        if self.gt:
            label_file = self.label_files[index]

        scan_file = self.scan_files[index]
        scan = np.fromfile(scan_file, dtype=np.float32).reshape((-1, 4))
        # scan = np.zeros_like(scan)

        if self.knn:
            try:
                knn_file = self.knn_files[index]
                knns = np.fromfile(knn_file, dtype=np.int32).reshape((-1, self.neighbor))
            except:
                tree = kdtree(scan[:, :3])
                _, knns = tree.query(scan[:, :3], k=self.neighbor)

        # whether drop some points
        self.points_to_drop = None
        if whether_aug(self.train, self.dataset['drop']):
            self.drop_points = random.uniform(0, self.dataset['drop_rate'])
            self.points_to_drop = np.random.randint(0, len(scan)-1, int(len(scan) * self.drop_points))
            scan = np.delete(scan, self.points_to_drop, axis=0)
            knns = np.delete(knns, self.points_to_drop, axis=0)
        # scale aug
        if whether_aug(self.train, self.dataset['scale']):
            factor = np.random.uniform(1-self.dataset['scale_rate'], 1+self.dataset['scale_rate'])
            scan[:, :3] *= factor

        range_data, range_point2pixel, range_pixel2point, valid, max_points = self.range_dataset(scan, scan_file, label_file)

        if valid.sum() == 0:
            scan = np.zeros_like(scan)[:10000]
        else:
            scan = scan[valid]
        # when training due to the order of index changed, we need to re-cal knn
        if self.train:
            tree = kdtree(scan[:, :3])
            _, knns = tree.query(scan[:, :3], k=self.neighbor)
        self.valid = valid
        self.max_points = max_points

        polar_data, polar_point2pixel, polar_pixel2point = self.polar_dataset(index, scan, label_file)
        r2p_flow_matrix = self.r2p_flow_matrix(polar_pixel2point, range_point2pixel)
        p2r_flow_matrix = self.p2r_flow_matrix(range_pixel2point, polar_point2pixel)

        knns_full = torch.full((max_points, self.neighbor), 0, dtype=torch.long)
        knns_full[:knns.shape[0]] = torch.from_numpy(knns).long()

        return range_data, polar_data, r2p_flow_matrix, p2r_flow_matrix, knns_full

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

    @staticmethod
    def map(label, mapdict):
        # put label from original values to xentropy
        # or vice-versa, depending on dictionary values
        # make learning map a lookup table
        maxkey = 0
        for key, data in mapdict.items():
            if isinstance(data, list):
                nel = len(data)
            else:
                nel = 1
            if key > maxkey:
                maxkey = key
        # +100 hack making lut bigger just in case there are unknown labels
        if nel > 1:
            lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
        else:
            lut = np.zeros((maxkey + 100), dtype=np.int32)
        for key, data in mapdict.items():
            try:
                lut[key] = data
            except IndexError:
                mp_logger("Wrong key: {}".format(key))
        # do the mapping
        return lut[label]

    def get_original_class_string(self, idx):
        return self.labels[idx]

    def get_xentropy_class_string(self, idx):
        return self.labels[self.learning_map_inv[idx]]

    def to_original(self, label):
        # put label in original values
        return self.map(label, self.learning_map_inv)

    def to_xentropy(self, label):
        # put label in xentropy values
        return self.map(label, self.learning_map)

    def to_color(self, label):
        # put label in original values
        label = self.map(label, self.learning_map_inv)
        # put label in color
        return self.map(label, self.color_map)

if __name__ == '__main__':
    import yaml
    from tqdm import tqdm
    ARCH = yaml.safe_load(open('configs/resnet_semantickitti.yaml', 'r'))
    DATA = yaml.safe_load(open('configs/semantic-kitti.yaml', 'r'))

    dataset = SemanticKitti(root='dataset/SemanticKITTI/',
                            sequences=DATA["split"]["train"],
                            labels=DATA["labels"],
                            color_map=DATA["color_map"],
                            learning_map=DATA["learning_map"],
                            learning_map_inv=DATA["learning_map_inv"],
                            range_cfg=ARCH['range'],         # configs for range view (dict)
                            polar_cfg=ARCH['polar'],         # configs for polar view (dict)
                            dataset_cfg=ARCH['dataset'])
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
