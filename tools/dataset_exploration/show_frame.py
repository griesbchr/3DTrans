import argparse
from pathlib import Path
import glob
import pickle
import orjson
import copy
from scipy.spatial.transform import Rotation
import yaml
from easydict import EasyDict

import open3d 
from tools.visual_utils import open3d_vis_utils as vis

import numpy as np


def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dataset', type=str, default="avltruck", help='the dataset name')
    parser.add_argument('--frame-idx', type=str, default="sequences/CityStreet_dgt_2021-11-09-09-08-59_0_s0/dataset/logical_frame_000016.json", help='the frame_idx in form "sequences/SEQUENCENAME/dataset/logical_frame_FRAMEID.json"')
    args = parser.parse_args()

    if (args.dataset == "avltruck"):
        from pcdet.datasets.avltruck.avl_dataset import AVLDataset
        cfg_path =  "/home/cgriesbacher/thesis/3DTrans/tools/cfgs/dataset_configs/avltruck/OD/avltruck_dataset.yaml"
        dataset_cfg = EasyDict(yaml.safe_load(open(cfg_path)))
        data_dir = Path("/") / 'data' / 'AVLTruck'

        dataset = AVLDataset(dataset_cfg, class_names=None)
        #get annos from info files
        dataset.set_split('train')
        sample_id_list = dataset.sample_id_list
        list_index = sample_id_list.index(args.frame_idx)
        info = copy.deepcopy(dataset.avl_infos[list_index])

        gt_boxes_lidar = info["annos"]['gt_boxes_lidar']  

    elif (args.dataset == "zod"):
        from pcdet.datasets.zod.zod_dataset import ZODDataset
        cfg_path =  "tools/cfgs/dataset_configs/zod/OD/zod_dataset.yaml"
        dataset_cfg = EasyDict(yaml.safe_load(open(cfg_path)))
        data_dir = Path("/") / 'data' / 'zod'
        class_names = None

        dataset = ZODDataset(dataset_cfg, class_names=class_names)
        annos = dataset.get_label(args.frame_idx)
        gt_boxes_lidar = annos['gt_boxes_lidar'] 

        #get annos from info files
        #dataset.set_split('train')
        #sample_id_list = dataset.sample_id_list
        #list_index = sample_id_list.index(args.frame_idx)
        #info = copy.deepcopy(dataset.avl_infos[list_index])
        #gt_boxes_lidar = info["annos"]['gt_boxes_lidar']   
        
    else:
        raise NotImplementedError("Please specify the dataset path")
    

    points = dataset.get_lidar(args.frame_idx)
  
    vis.draw_scenes(points, gt_boxes_lidar, fit_ground_plane=False)

    return


if __name__ == '__main__':

    main()