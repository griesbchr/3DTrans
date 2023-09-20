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
    parser.add_argument('--dataset', type=str, default=None, help='the dataset name')
    parser.add_argument('--frame-idx', type=str, default=None)
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model path')

    args = parser.parse_args()

    logger = common_utils.create_logger()

    dataset = "zod"
    
    #avlrooftop
    #checkpoint_path = "/home/cgriesbacher/thesis/3DTrans/output/avlrooftop_models/second/full_80epochs/ckpt/checkpoint_epoch_80.pth"

    #zod 
    #checkpoint_path = "/home/cgriesbacher/thesis/3DTrans/output/zod_models/second/full_2epochs_notrunc/ckpt/checkpoint_epoch_2.pth"

    #avltruck
    checkpoint_path = "/home/cgriesbacher/thesis/3DTrans/output/avltruck_models/second/full_10epochs/ckpt/checkpoint_epoch_10.pth"
    

    
    if (args.dataset == None):
        args.dataset = dataset

    if (args.ckpt == None):
        args.ckpt = checkpoint_path


    if (args.dataset == "avltruck"):
        from pcdet.datasets.avltruck.avltruck_dataset import AVLTruckDataset
        cfg_path =  "/home/cgriesbacher/thesis/3DTrans/tools/cfgs/dataset_configs/avltruck/OD/avltruck_dataset.yaml"
        dataset_cfg = EasyDict(yaml.safe_load(open(cfg_path)))

        class_names = ['Vehicle_Drivable_Car',
                       'Vehicle_Drivable_Van', 
                       'Vehicle_Ridable_Motorcycle', 
                       'Vehicle_Ridable_Bicycle', 
                       'Human', 
                       'LargeVehicle_Bus', 
                       'LargeVehicle_TruckCab', 
                       'LargeVehicle_Truck', 
                       'Trailer']

        if args.frame_idx is None:
            args.frame_idx = 'sequences/CityThoroughfare_dgt_2021-08-20-10-48-41_0_s0/dataset/logical_frame_000020.json'
        
    elif (args.dataset == "zod"):
        from pcdet.datasets.zod.zod_dataset import ZODDataset
        cfg_path =  "cfgs/dataset_configs/zod/OD/zod_dataset.yaml"
        dataset_cfg = EasyDict(yaml.safe_load(open(cfg_path)))

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
        

    elif (args.dataset == "avlrooftop"):
        from pcdet.datasets.avlrooftop.avlrooftop_dataset import AVLRooftopDataset
        cfg_path =  "cfgs/dataset_configs/avlrooftop/OD/avlrooftop_dataset.yaml"
        dataset_cfg = EasyDict(yaml.safe_load(open(cfg_path)))
        
        if args.frame_idx is None:
            args.frame_idx = "sequences/CITY_Sunny_junction_20200319140600/unpacked/lidar/0026.pkl"
        
        class_names = ["Vehicle_Drivable_Car",
                        "Vehicle_Drivable_Van",
                        "LargeVehicle_Truck",
                        "LargeVehicle_TruckCab",
                        "LargeVehicle_Bus",
                        "LargeVehicle_Bus_Bendy",
                        "Trailer",
                        "Vehicle_Ridable_Motorcycle",
                        "Vehicle_Ridable_Bicycle",
                        "Human",
                        "PPObject_Stroller"]

    else:
        raise NotImplementedError("Please specify the dataset path")
    

    dataset, train_loader, train_sampler = build_dataloader(dataset_cfg=dataset_cfg,
                                    class_names=class_names,
                                    batch_size=1,
                                    dist=False,
                                    workers=0,
                                    logger=logger,
                                    training=False)
    annos = dataset.get_label(args.frame_idx)
    gt_boxes_lidar = annos['gt_boxes_lidar'] 

    points = dataset.get_lidar(args.frame_idx)

    #filter out gt boxes that are out of range
    gt_boxes_lidar = gt_boxes_lidar[gt_boxes_lidar[:,0] < dataset_cfg.POINT_CLOUD_RANGE[3]]
    gt_boxes_lidar = gt_boxes_lidar[gt_boxes_lidar[:,0] > dataset_cfg.POINT_CLOUD_RANGE[0]]
    gt_boxes_lidar = gt_boxes_lidar[gt_boxes_lidar[:,1] < dataset_cfg.POINT_CLOUD_RANGE[4]]
    gt_boxes_lidar = gt_boxes_lidar[gt_boxes_lidar[:,1] > dataset_cfg.POINT_CLOUD_RANGE[1]]
    gt_boxes_lidar = gt_boxes_lidar[gt_boxes_lidar[:,2] < dataset_cfg.POINT_CLOUD_RANGE[5]]
    gt_boxes_lidar = gt_boxes_lidar[gt_boxes_lidar[:,2] > dataset_cfg.POINT_CLOUD_RANGE[2]]


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