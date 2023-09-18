import argparse
from pathlib import Path
import glob
import pickle
import orjson
import copy
from scipy.spatial.transform import Rotation
import yaml
from easydict import EasyDict

from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import build_dataloader
from pcdet.config import cfg, cfg_from_yaml_file
import torch
from pcdet.utils import common_utils

import open3d 
from tools.visual_utils import open3d_vis_utils as vis

import numpy as np


def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dataset', type=str, help='the dataset name')
    parser.add_argument('--frame-idx', type=str, default=None)
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model path')

    args = parser.parse_args()

    logger = common_utils.create_logger()

    if (args.dataset == "avltruck"):
        from pcdet.datasets.avltruck.avltruck_dataset import AVLTruckDataset
        cfg_path =  "/home/cgriesbacher/thesis/3DTrans/tools/cfgs/dataset_configs/avltruck/OD/avltruck_dataset.yaml"
        dataset_cfg = EasyDict(yaml.safe_load(open(cfg_path)))
        data_dir = Path("/") / 'data' / 'AVLTruck'

        if args.frame_idx is None:
            #args.frame_idx = "sequences/CityStreet_dgt_2021-07-06-09-56-53_0_s0/dataset/logical_frame_000006.json"
            args.frame_idx = "sequences/CityStreet_dgt_2021-07-12-11-45-52_0_s0/dataset/logical_frame_000006.json"
        dataset = AVLTruckDataset(dataset_cfg, class_names=None, training=False)
        
        annos = dataset.get_label(args.frame_idx)
        gt_boxes_lidar = annos['gt_boxes_lidar'] 

        ##get annos from info files
        #sample_id_list = dataset.sample_id_list
        #list_index = sample_id_list.index(args.frame_idx)
        #info = copy.deepcopy(dataset.avl_infos[list_index])
        #gt_boxes_lidar = info["annos"]['gt_boxes_lidar']  

    elif (args.dataset == "zod"):
        from pcdet.datasets.zod.zod_dataset import ZODDataset
        cfg_path =  "cfgs/dataset_configs/zod/OD/zod_dataset.yaml"
        dataset_cfg = EasyDict(yaml.safe_load(open(cfg_path)))
        data_dir = Path("/") / 'data' / 'zod'

        if args.frame_idx is None:
            args.frame_idx = "055820"
        
        class_names = ["Vehicle_Car", 
                       "Vehicle_Van", 
                       "Vehicle_Truck", 
                       "Vehicle_Trailer", 
                       "Vehicle_Bus", 
                       "Vehicle_HeavyEquip", 
                       "Vehicle_TramTrain",
                       "VulnerableVehicle_Bicycle",
                       "VulnerableVehicle_Motorcycle",
                       "Pedestrian"]
        
        #dataset = ZODDataset(dataset_cfg, class_names=class_names)
        dataset, train_loader, train_sampler = build_dataloader(dataset_cfg=dataset_cfg,
                                        class_names=class_names,
                                        batch_size=1,
                                        dist=False,
                                        workers=0,
                                        logger=logger,
                                        training=False)
        annos = dataset.get_label(args.frame_idx, class_names)
        
        #filter annos dict for class names
        mask = np.isin(annos['name'], class_names)
        annos["name"] = annos["name"][mask]
        annos["gt_boxes_lidar"] = annos["gt_boxes_lidar"][mask]
        annos["truncated"] = annos["truncated"][mask]


        gt_boxes_lidar = annos['gt_boxes_lidar'] 

        points = dataset.get_lidar(args.frame_idx)
        from pcdet.utils import box_utils
        corners = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
        truncated = dataset.get_object_truncation(corners, dataset.zod_frames[args.frame_idx].calibration)
        corners_alt = np.array([obj_anno.box3d.corners for obj_anno in annos["obj_annos"]])
        print(truncated)
        #get annos from info files
        #dataset.set_split('train')
        #sample_id_list = dataset.sample_id_list
        #list_index = sample_id_list.index(args.frame_idx)
        #info = copy.deepcopy(dataset.avl_infos[list_index])
        #gt_boxes_lidar = info["annos"]['gt_boxes_lidar']   
    elif (args.dataset == "avlrooftop"):
        from pcdet.datasets.avlrooftop.avlrooftop_dataset import AVLRooftopDataset
        cfg_path =  "cfgs/dataset_configs/avlrooftop/OD/avlrooftop_dataset.yaml"
        dataset_cfg = EasyDict(yaml.safe_load(open(cfg_path)))
        data_dir = Path("/") / 'data' / 'AVLRooftop'
        
        if args.frame_idx is None:
            args.frame_idx = "sequences/CITY_Sunny_junction_20200319140600/unpacked/lidar/0026.pkl"
            args.frame_idx = "sequences/CITY_Rain_street_20200511124709/unpacked/lidar/0020.pkl" #no bike human label
        class_names = None

        dataset = AVLRooftopDataset(dataset_cfg, 
                                    class_names=class_names, 
                                    training=True)
        gt_boxes_lidar = dataset.get_label(args.frame_idx)['gt_boxes_lidar']
        
        #get annos from info files
        #sample_id_list = dataset.sample_id_list
        #list_index = sample_id_list.index(args.frame_idx)
        #info = copy.deepcopy(dataset.avl_infos[list_index])
        #gt_boxes_lidar = info["annos"]['gt_boxes_lidar']  

    else:
        raise NotImplementedError("Please specify the dataset path")
    
    points = dataset.get_lidar(args.frame_idx)

    #load model if specified
    if args.ckpt is not None:

        

        #get model config 
        ckpt_path = Path(args.ckpt)
        cfg_path = [file for file in ckpt_path.parent.parent.glob('*.yaml')]
        assert len(cfg_path) == 1, "More of less than one config file found"
        cfg_path = cfg_path[0]

        #parse config
        cfg_from_yaml_file(cfg_path, cfg)


        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
        model.cuda()
        model.eval()
        with torch.no_grad():
            #find sample index for frame
            sample_id_list = dataset.sample_id_list
            list_index = sample_id_list.index(args.frame_idx)
            #get data dict for frame
            data_dict = dataset.__getitem__(list_index)
            data_dict = dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
       
        vis.draw_scenes(points, gt_boxes_lidar, pred_dicts[0]["pred_boxes"].detach().cpu().numpy())
    else:
        vis.draw_scenes(points, gt_boxes_lidar)

    return


if __name__ == '__main__':

    main()