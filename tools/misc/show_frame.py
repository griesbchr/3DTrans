import argparse
from pathlib import Path
import yaml
from easydict import EasyDict
import numpy as np

from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import build_dataloader
from pcdet.config import cfg, cfg_from_yaml_file
import torch
from pcdet.utils import common_utils

from tools.visual_utils import open3d_vis_utils as vis

from tools.misc.calc_num_of_params import count_parameters  #, calc_flops
import os

def parse_args():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dataset', type=str, default=None, help='the dataset name')
    parser.add_argument('--frame-idx', type=str, default=None)
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model path')

    args = parser.parse_args()

    return args

def main():
    os.chdir("/home/cgriesbacher/thesis/3DTrans/tools")
    args = parse_args()
    
    save_image = True

    fov=True

    dataset = "zod"
    checkpoint_path = None
    
    #avlrooftop
    #checkpoint_path = "/home/cgriesbacher/thesis/3DTrans/output_okeanos/output/avltruck_models/pvrcnnpp_STrooftop/D1_5epochs_STrooftop_ft_D6_50epochs_ros_06_015_thresh_high_lr/ckpt/checkpoint_epoch_4.pth"
    #checkpoint_path = "/home/cgriesbacher/thesis/3DTrans/output_okeanos/output/avltruck_models/pvrcnnpp_ros/D6_50epochs/ckpt/checkpoint_epoch_50.pth"
    #checkpoint_path = "/home/cgriesbacher/thesis/3DTrans/output_okeanos/output/avlrooftop_models/pvrcnnpp/D1_50epochs/ckpt/checkpoint_epoch_50.pth"

    #zod 
    #checkpoint_path = "/home/cgriesbacher/thesis/3DTrans/output/zod_models/dsvt_pillar/D16_100epochs/ckpt/checkpoint_epoch_100.pth"
    checkpoint_path = "/home/cgriesbacher/thesis/3DTrans/output_okeanos/output/zod_models/pvrcnnpp/D16_50epochs/ckpt/checkpoint_epoch_50.pth"

    #avltruck
    #checkpoint_path = "/home/cgriesbacher/thesis/3DTrans/output/avltruck_models/centerpoint/D6_100epochs_4classes/ckpt/checkpoint_epoch_100.pth"
    #checkpoint_path = "/home/cgriesbacher/thesis/3DTrans/output_okeanos/output/avltruck_models/pvrcnnpp_ros/D6_50epochs/ckpt/checkpoint_epoch_50.pth"
    #checkpoint_path = "/home/cgriesbacher/thesis/3DTrans/output_okeanos/output/avltruck_models/pvrcnnpp_STzod/D6_5epochs_STzod_ft_D16_50epochs_ros/ckpt/checkpoint_epoch_3.pth"

    if (args.dataset == None):
        args.dataset = dataset

    if (args.ckpt == None):
        args.ckpt = checkpoint_path
    
    if (args.dataset == "avltruck"):
        cfg_path =  "/home/cgriesbacher/thesis/3DTrans/tools/cfgs/dataset_configs/avltruck/OD/avltruck_dataset.yaml"
        dataset_cfg = EasyDict(yaml.safe_load(open(cfg_path)))

        dataset_class_names = ['Vehicle_Drivable_Car',
                       'Vehicle_Drivable_Van', 
                       'Vehicle_Ridable_Motorcycle', 
                       'Vehicle_Ridable_Bicycle', 
                       'Human', 
                       'LargeVehicle_Bus', 
                       'LargeVehicle_TruckCab', 
                       'LargeVehicle_Truck', 
                       'Trailer']

        if args.frame_idx is None:
            args.frame_idx = 'sequences/CityStreet_dgt_2021-07-23-12-56-14_0_s0/dataset/logical_frame_000008.json'
        
        image_path_frame = args.frame_idx.split("/")[1] + "_" + args.frame_idx.split("/")[-1].split(".")[0] 

    elif (args.dataset == "zod"):
        cfg_path =  "/home/cgriesbacher/thesis/3DTrans/tools/cfgs/dataset_configs/zod/OD/zod_dataset.yaml"
        dataset_cfg = EasyDict(yaml.safe_load(open(cfg_path)))
        
        dataset_class_names = ["Vehicle_Car", 
                       "Vehicle_Van", 
                       "Vehicle_Truck", 
                       "Vehicle_Trailer", 
                       "Vehicle_Bus", 
                       "Vehicle_HeavyEquip", 
                       "Vehicle_TramTrain",
                       "VulnerableVehicle_Bicycle",
                       "VulnerableVehicle_Motorcycle",
                       "Pedestrian"]
        if args.frame_idx is None:
            args.frame_idx = "055261"
        
        image_path_frame = args.frame_idx

    elif (args.dataset == "avlrooftop"):
        cfg_path =  "/home/cgriesbacher/thesis/3DTrans/tools/cfgs/dataset_configs/avlrooftop/OD/avlrooftop_dataset.yaml"
        dataset_cfg = EasyDict(yaml.safe_load(open(cfg_path)))
    
        dataset_class_names = ["Vehicle_Drivable_Car",
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

    
        if args.frame_idx is None:
            args.frame_idx = 'sequences/CITY_Normal_roundabout_20200320100220_1/unpacked/lidar/0019.pkl'
        
        image_path_frame = args.frame_idx.split("/")[1] + "_" + args.frame_idx.split("/")[-1].split(".")[0]
    elif (args.dataset == "kitti"):
        cfg_path =  "/home/cgriesbacher/thesis/3DTrans/tools/cfgs/dataset_configs/kitti/OD/kitti_dataset.yaml"
        dataset_cfg = EasyDict(yaml.safe_load(open(cfg_path)))
    
        dataset_class_names = ["Car",
                        "Van",
                        "Truck",
                        "Pedestrian",
                        "Person_sitting",
                        "Cyclist",
                        "Tram"]

    
        if args.frame_idx is None:
            args.frame_idx = '000063'
        
        image_path_frame = args.frame_idx

    else:
        raise NotImplementedError("Please specify the dataset path")

    if fov is not None:
        dataset_cfg.FOV_POINTS_ONLY = fov
        dataset_cfg.EVAL_FOV_ONLY = fov
        dataset_cfg.TRAIN_FOV_ONLY = fov
        dataset_cfg.LIDAR_HEADING = 0
        dataset_cfg.LIDAR_FOV = 120

    image_path = "viz/" + args.dataset

    if args.ckpt != None:
        train_dataset = args.ckpt.split("/")[-5]
        train_detector = args.ckpt.split("/")[-4]
        train_name = args.ckpt.split("/")[-3]
        train_epoch = args.ckpt.split("/")[-1].split(".")[0].split("_")[-1]

        image_path += "/" + train_dataset + "_" + train_detector + "_" + train_name + "_" + train_epoch

    image_path += "/" + image_path_frame + ".png"

    logger = common_utils.create_logger()

    #load model if specified
    if args.ckpt is not None:

        #get model config 
        ckpt_path = Path(args.ckpt)
        cfg_path = [file for file in ckpt_path.parent.parent.glob('*.yaml')]
        assert len(cfg_path) == 1, "More or less than one config file found in "+ckpt_path.parent.parent.__str__()
        cfg_path = cfg_path[0]

        #parse config
        cfg_from_yaml_file(cfg_path, cfg)

        # some detectors use different range or voxel preprocessing
        if hasattr(cfg.DATA_CONFIG, "POINT_CLOUD_RANGE"):
            dataset_cfg.POINT_CLOUD_RANGE = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
        if hasattr(cfg.DATA_CONFIG, "DATA_PROCESSOR"):
            dataset_cfg.DATA_PROCESSOR = cfg.DATA_CONFIG.DATA_PROCESSOR

        #build dataset
        dataset, train_loader, train_sampler = build_dataloader(dataset_cfg=dataset_cfg,
                                    class_names=cfg.CLASS_NAMES,
                                    batch_size=1,
                                    dist=False,
                                    workers=0,
                                    logger=logger,
                                    training=False) 
        
        #build model
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
        model.cuda()
        model.eval()

        #calc model params
        #count_parameters(model)

        #find sample index for frame
        sample_id_list = dataset.sample_id_list
        list_index = sample_id_list.index(args.frame_idx)
        #get data from info files -> is in detector class name space
        data_dict = dataset.__getitem__(list_index)


        data_dict = dataset.collate_batch([data_dict])

        #get eval frame        
        with torch.no_grad():
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

        annos = dataset.generate_prediction_dicts(
                data_dict, pred_dicts, dataset.class_names, None)
        
        eval_metric = "waymo"
        result_str, _, eval_gt_annos, eval_det_annos= dataset.evaluation(det_annos=annos, class_names=dataset.class_names, eval_metric=eval_metric, return_annos=True)
        
        print(result_str)

        print("remaining gt boxes for nontruncated method:", dataset.extract_fov_gt_nontruncated(data_dict["gt_boxes"][0,:,:7].cpu().numpy(), 120, 0).sum())
        print("remaining gt boxes for center method:", dataset.extract_fov_gt(data_dict["gt_boxes"][0,:,:7].cpu().numpy(), 120, 0).sum())
        
        points4 = data_dict['points'].detach().cpu().numpy()[:,1:]
        gt_boxes = eval_gt_annos[0]["gt_boxes_lidar"]
        det_boxes = eval_det_annos[0]["boxes_lidar"]

        points4 = dataset.extract_fov_data(points4, 120, 0)

        scores = eval_det_annos[0]["score"]
        points = points4[:,:3]
        gt_boxes_lidar = gt_boxes
        color = points4[:,-1]
    else:
        dataset, train_loader, train_sampler = build_dataloader(dataset_cfg=dataset_cfg,
                                        class_names=dataset_class_names,
                                        batch_size=1,
                                        dist=False,
                                        workers=0,
                                        logger=logger,
                                        training=False) 
        points4 = dataset.get_lidar(args.frame_idx)
        gt_boxes_lidar = dataset.get_label(args.frame_idx)["gt_boxes_lidar"]
        #filter out gt boxes that are out of range
        gt_boxes_lidar = gt_boxes_lidar[gt_boxes_lidar[:,0] < dataset_cfg.POINT_CLOUD_RANGE[3]]
        gt_boxes_lidar = gt_boxes_lidar[gt_boxes_lidar[:,0] > dataset_cfg.POINT_CLOUD_RANGE[0]]
        gt_boxes_lidar = gt_boxes_lidar[gt_boxes_lidar[:,1] < dataset_cfg.POINT_CLOUD_RANGE[4]]
        gt_boxes_lidar = gt_boxes_lidar[gt_boxes_lidar[:,1] > dataset_cfg.POINT_CLOUD_RANGE[1]]
        gt_boxes_lidar = gt_boxes_lidar[gt_boxes_lidar[:,2] < dataset_cfg.POINT_CLOUD_RANGE[5]]
        gt_boxes_lidar = gt_boxes_lidar[gt_boxes_lidar[:,2] > dataset_cfg.POINT_CLOUD_RANGE[2]]

        #filter out boxes that are not in fov
        if fov:
            gt_boxes_lidar_mask = dataset.extract_fov_gt(gt_boxes_lidar, dataset_cfg.LIDAR_FOV, dataset_cfg.LIDAR_HEADING)
            gt_boxes_lidar = gt_boxes_lidar[gt_boxes_lidar_mask]

        color=points4[:,-1]
        points=points4[:,:3]
        det_boxes=None
        scores = None      
    if save_image:
        #insert image view here, can be copied by pressing Ctrl+C in open3d window and paste in editor file
        
        view_control = {
			"boundingbox_max" : [ 176.625, 143.04679870605469, 15.683642387390137 ],
			"boundingbox_min" : [ -0.059999999999999998, -142.8343505859375, -4.8461880683898926 ],
			"field_of_view" : 60.0,
			"front" : [ -0.9691411056871565, -0.084253242249075211, 0.23166119320679146 ],
			"lookat" : [ 26.938944167003275, 0.13884025881358295, -1.5513422083739699 ],
			"up" : [ 0.23550074544658964, -0.038778553809448474, 0.97110021246962386 ],
			"zoom" : 0.079999999999999613
		}

        vis.draw_scenes(points, gt_boxes_lidar,det_boxes ,point_colors=color, view_control=view_control, image_path=image_path, ref_scores=scores)
    else:
        vis.draw_scenes(points, gt_boxes_lidar,det_boxes ,point_colors=color, ref_scores=scores)

    return


if __name__ == '__main__':

    main()