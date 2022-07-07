import os
import sys
sys.path.append(os.getcwd())

import pickle
from pathlib import Path

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits

def get_pkl_info(nusc, version):
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

    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in train_scenes])
    val_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in val_scenes])

    train_token_list, val_token_list = get_path_infos(nusc,train_scenes,val_scenes)

    print('%s: train scene(%d), val scene(%d)' % (version, len(train_scenes), len(val_scenes)))

    if version == 'v1.0-test':
        write_pkl_test(nusc, train_token_list, 'test')
    else:
        write_pkl(nusc, train_token_list, 'train')
        write_pkl(nusc, val_token_list, 'val')
        write_pkl(nusc, train_token_list + val_token_list, 'trainval')

def write_pkl_test(nusc, token_list, split):
    samples_annots = [
        '{}**{}'.format(nusc.get('sample_data', token)['filename'],
                        'lidarseg/v1.0-test-fake/{}_lidarseg.bin'.format(token)) \
        for token in token_list
    ]
    write_file = 'dataset/nuScenes/nuscenes_{}.pkl'.format(split)
    print('write {} samples to {}'.format(len(samples_annots), write_file))
    with open(write_file, 'wb') as fp:
        pickle.dump(samples_annots, fp)

def write_pkl(nusc, token_list, split):
    samples_annots = [
        '{}**{}'.format(nusc.get('sample_data', token)['filename'],
                        nusc.get('lidarseg', token)['filename']) \
        for token in token_list
    ]
    write_file = 'dataset/nuScenes/nuscenes_{}.pkl'.format(split)
    print('write {} samples to {}'.format(len(samples_annots), write_file))
    with open(write_file, 'wb') as fp:
        pickle.dump(samples_annots, fp)

def get_available_scenes(nusc):
    available_scenes = []
    print('total scene num:', len(nusc.scene))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            if not Path(lidar_path).exists():
                scene_not_exist = True
                break
            else:
                break

        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num:', len(available_scenes))
    return available_scenes

def get_path_infos(nusc,train_scenes,val_scenes):
    train_token_list = []
    val_token_list = []
    for sample in nusc.sample:
        scene_token = sample['scene_token']
        data_token = sample['data']['LIDAR_TOP']
        if scene_token in train_scenes:
            train_token_list.append(data_token)
        else:
            val_token_list.append(data_token)
    return train_token_list, val_token_list

if __name__ == '__main__':
    data_path = 'dataset/nuScenes/full/'
    for version in ['v1.0-test', 'v1.0-trainval']:
        nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
        get_pkl_info(nusc, version)
