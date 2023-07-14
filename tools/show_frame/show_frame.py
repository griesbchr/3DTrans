import argparse
from pathlib import Path
import glob
import orjson
from scipy.spatial.transform import Rotation

import open3d
from tools.visual_utils import open3d_vis_utils as vis

import numpy as np



def get_lidar(root_path, idx):
    lidar_file = root_path / idx

    np_lidar_file = lidar_file.with_suffix('.npy')
    if np_lidar_file.is_file(
    ):  # all preprocessing done in `convert_json_to_numpy`
        points = np.load(np_lidar_file, allow_pickle=False)
    else:  # fallback to json (very slow)
        assert lidar_file.exists()
        with open(str(lidar_file), 'r') as f:
            data = orjson.loads(f.read())
        points = np.array([[p['x'], p['y'], p['z'], p['i']]
                            for p in data['points']])
        dev_loc, dev_heading = data['device_position'], data[
            'device_heading']
        dev_loc = np.array([dev_loc['x'], dev_loc['y'], dev_loc['z']])
        points[:, 0:3] -= dev_loc[None]
        rot_matrix = Rotation.from_quat([
            dev_heading['x'], dev_heading['y'], dev_heading['z'],
            dev_heading['w']
        ]).as_matrix()
        points[:, 0:3] = points[:, 0:3] @ rot_matrix

    points[:, -1] = np.clip(points[:, -1], a_min=0, a_max=1.)
    return points


def get_labels(root_path, idx):
    label_file = (root_path / Path(idx)).parent.parent / 'label'
    label_file = [lf for lf in label_file.glob('*.json')]
    assert label_file.__len__() == 1, 'Multiple label files are not expected'
    label_file = label_file[0]

    with open(str(root_path / idx), 'r') as f:
        data = orjson.loads(f.read())
    dev_loc, dev_rot = data['device_position'], data['device_heading']
    dev_loc = np.array([dev_loc['x'], dev_loc['y'], dev_loc['z']])
    dev_q = Rotation.from_quat(
        [dev_rot['x'], dev_rot['y'], dev_rot['z'], dev_rot['w']])

    with open(str(label_file), 'r') as f:
        labels = orjson.loads(f.read())

    #boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    boxes3d_list = []
    for l in labels['labels']:
        if Path(idx).stem in l['file_id']:
            bb = l.get('three_d_bbox', None)
            name = l.get('label_category_id', None)
            q = l.get('three_d_bbox', {}).get('quaternion', None)
            if bb is None or name is None or q is None:
                continue
            yaw_q = Rotation.from_quat(
                np.array([q['x'], q['y'], q['z'], q['w']]))
            loc = (np.array([bb['cx'], bb['cy'], bb['cz']]) -
                    dev_loc) @ dev_q.as_matrix()
            yaw = (dev_q.inv() * yaw_q).as_euler('xyz', degrees=False)[-1]
            boxes3d_list.append()
    boxes3d = np.array(boxes3d_list)
    return boxes3d


def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dataset', type=str, default="avltruck", help='the data path')
    parser.add_argument('--frame_idx', type=str, default="sequences/CityStreet_dgt_2021-07-08-15-24-00_0_s0/dataset/logical_frame_000005", help='the frame_idx in form "sequences/SEQUENCENAME/dataset/logical_frame_FRAMEID"')
    args = parser.parse_args()

    if (args.dataset == "avltruck"):
        rootpath = Path("/data/AVLTruck/")
    else:
        raise NotImplementedError("Please specify the dataset path")

    points = get_lidar(rootpath, args.frame_idx)
    labels = get_labels(rootpath, args.frame_idx)
    vis.draw_scenes(points,)

    return


if __name__ == '__main__':

    main()