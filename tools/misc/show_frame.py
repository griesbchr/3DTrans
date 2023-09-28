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


def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dataset', type=str, default=None, help='the dataset name')
    parser.add_argument('--frame-idx', type=str, default=None)
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model path')

    args = parser.parse_args()

    logger = common_utils.create_logger()

    dataset = "avltruck"
    
    #avlrooftop
    #checkpoint_path = "/home/cgriesbacher/thesis/3DTrans/output/avlrooftop_models/second/full_80epochs/ckpt/checkpoint_epoch_80.pth"

    #zod 
    #checkpoint_path = "/home/cgriesbacher/thesis/3DTrans/output/output/zod_models/second/full_30epochs_fusesingletrack/ckpt/checkpoint_epoch_30.pth"

    #avltruck
    checkpoint_path = "/home/cgriesbacher/thesis/3DTrans/output/avltruck_models/second/D6_80epochs_fusesingletrack/ckpt/checkpoint_epoch_80.pth"

    
    if (args.dataset == None):
        args.dataset = dataset

    if (args.ckpt == None):
        args.ckpt = checkpoint_path


    if (args.dataset == "avltruck"):
        from pcdet.datasets.avltruck.avltruck_dataset import AVLTruckDataset
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
            args.frame_idx = 'sequences/CityStreet_dgt_2021-08-19-11-46-54_0_s0/dataset/logical_frame_000012.json'
        
    elif (args.dataset == "zod"):
        from pcdet.datasets.zod.zod_dataset import ZODDataset
        cfg_path =  "cfgs/dataset_configs/zod/OD/zod_dataset.yaml"
        dataset_cfg = EasyDict(yaml.safe_load(open(cfg_path)))

        if args.frame_idx is None:
            args.frame_idx = "022786"
        
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
        

    elif (args.dataset == "avlrooftop"):
        from pcdet.datasets.avlrooftop.avlrooftop_dataset import AVLRooftopDataset
        cfg_path =  "cfgs/dataset_configs/avlrooftop/OD/avlrooftop_dataset.yaml"
        dataset_cfg = EasyDict(yaml.safe_load(open(cfg_path)))
        
        if args.frame_idx is None:
            args.frame_idx = 'sequences/INTERURBAN_Normal_roundabout_20200505103429/unpacked/lidar/0007.pkl'
        
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

    else:
        raise NotImplementedError("Please specify the dataset path")
    

    #load model if specified
    if args.ckpt is not None:

        #get model config 
        ckpt_path = Path(args.ckpt)
        cfg_path = [file for file in ckpt_path.parent.parent.glob('*.yaml')]
        assert len(cfg_path) == 1, "More of less than one config file found"
        cfg_path = cfg_path[0]

        #parse config
        cfg_from_yaml_file(cfg_path, cfg)

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
        
        result_str, result_dict, eval_gt_annos, eval_det_annos= dataset.evaluation(det_annos=annos, class_names=dataset.class_names, eval_metric="kitti", return_annos=True)
        
        print(result_str)

        print("remaining gt boxes for nontruncated method:", dataset.extract_fov_gt_nontruncated(data_dict["gt_boxes"][0,:,:7].cpu().numpy(), 120, 0).sum())
        print("remaining gt boxes for center method:", dataset.extract_fov_gt(data_dict["gt_boxes"][0,:,:7].cpu().numpy(), 120, 0).sum())
        
        points4 = data_dict['points'].detach().cpu().numpy()[:,1:]
        gt_boxes = eval_gt_annos[0]["gt_boxes_lidar"]
        det_boxes = eval_det_annos[0]["boxes_lidar"]

        points4 = dataset.extract_fov_data(points4, 120, 0)

        vis.draw_scenes(points4[:,:3], gt_boxes, det_boxes,point_colors=points4[:,-1])
    else:
        dataset, train_loader, train_sampler = build_dataloader(dataset_cfg=dataset_cfg,
                                        class_names=dataset_class_names,
                                        batch_size=1,
                                        dist=False,
                                        workers=0,
                                        logger=logger,
                                        training=False) 
        points = dataset.get_lidar(args.frame_idx)
        gt_boxes_lidar = dataset.get_label(args.frame_idx)["gt_boxes_lidar"]
        #filter out gt boxes that are out of range
        gt_boxes_lidar = gt_boxes_lidar[gt_boxes_lidar[:,0] < dataset_cfg.POINT_CLOUD_RANGE[3]]
        gt_boxes_lidar = gt_boxes_lidar[gt_boxes_lidar[:,0] > dataset_cfg.POINT_CLOUD_RANGE[0]]
        gt_boxes_lidar = gt_boxes_lidar[gt_boxes_lidar[:,1] < dataset_cfg.POINT_CLOUD_RANGE[4]]
        gt_boxes_lidar = gt_boxes_lidar[gt_boxes_lidar[:,1] > dataset_cfg.POINT_CLOUD_RANGE[1]]
        gt_boxes_lidar = gt_boxes_lidar[gt_boxes_lidar[:,2] < dataset_cfg.POINT_CLOUD_RANGE[5]]
        gt_boxes_lidar = gt_boxes_lidar[gt_boxes_lidar[:,2] > dataset_cfg.POINT_CLOUD_RANGE[2]]

        vis.draw_scenes(points, gt_boxes_lidar)

    return


if __name__ == '__main__':

    main()