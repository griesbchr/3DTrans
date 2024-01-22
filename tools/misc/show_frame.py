import argparse
from pathlib import Path
import yaml
from easydict import EasyDict
import numpy as np
import matplotlib.pyplot as plt

from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import build_dataloader
from pcdet.config import cfg, cfg_from_yaml_file
import torch
from pcdet.utils import common_utils, downsample_utils


from tools.misc.calc_num_of_params import count_parameters  #, calc_flops
import os
import random


def parse_args():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dataset', type=str, default=None, help='the dataset name')
    parser.add_argument('--frame-idx', type=str, default=None)
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model path')

    args = parser.parse_args()

    return args

def label_point_cloud_beam(polar_image, beam, method='dbscan'):
    if polar_image.shape[0] <= beam:
        print("too small point cloud!")
        return np.arange(polar_image.shape[0])
    beam_label = downsample_utils.beam_label(polar_image[:,1], beam, method=method)
    return beam_label

def get_polar_image(points):
    theta, phi = downsample_utils.compute_angles(points[:,:3])
    r = np.sqrt(np.sum(points[:,:3]**2, axis=1))
    polar_image = points.copy()
    polar_image[:,0] = phi 
    polar_image[:,1] = theta
    polar_image[:,2] = r
    return polar_image
    
def segment_lidar_to_beams(orig_points, num_beams):
    points = orig_points.copy()
    #add z offset 
    points[:,2] += -3.4097

    polar_image = get_polar_image(points)

    ##reduce range for more reliable clustering
    polar_image = polar_image[polar_image[:,1] < np.pi/16]
    polar_image = polar_image[polar_image[:,1] > -np.pi/16]
    polar_image_plot = polar_image.copy()

    ##shift angle to 0 to 180 degree
    polar_image_plot[:,0] += np.pi/2
    #mod to 180 degree
    polar_image_plot[:,0] = polar_image_plot[:,0] % np.pi


    plt.figure()
    plt.scatter(polar_image_plot[:,0], polar_image_plot[:,1], c=polar_image_plot[:,2], s=0.1)
    plt.xlabel("phi")
    plt.ylabel("theta")
    plt.title("range image")
    #save fig
    plt.savefig("viz/range_image.png")

    beam_label = downsample_utils.beam_label(polar_image[:,1], num_beams, method='kmeans++')

    beam_label_plot =  downsample_utils.beam_label(polar_image_plot[:,1], num_beams,  method='kmeans++')
    #beam_label_plot_ransac = downsample_utils.beam_label_ransac(polar_image_plot, num_beams, inlier_threshold=0.01)

    color_arr = [0, 0.5, 1]
    beam_colors = np.zeros((num_beams,1))
    for i in range(num_beams):
        beam_colors[i,:]  = np.array([color_arr[i%3]])
    #map beam labels to colors
    beam_label_plot = beam_colors[beam_label_plot]

    plt.figure()
    plt.scatter(polar_image_plot[:,0], polar_image_plot[:,1]*180/np.pi, c=beam_label_plot, s=0.1)
    plt.xlabel("phi")
    plt.ylabel("theta")
    plt.title("range image")
    plt.ylim(-10, 10)
    #save fig
    plt.savefig("viz/range_image_beams.png")

    return beam_label


def main():
    os.chdir("/home/cgriesbacher/thesis/3DTrans/tools")
    args = parse_args()
    
    save_image = True

    fov=True
    training = True             #enable augmentations
    no_detection = False
    dataset = "avlrooftop"
    checkpoint_path = None
    select_random_frame = False
    
    #avlrooftop
    #checkpoint_path = "/home/cgriesbacher/thesis/3DTrans/output_okeanos/output/avltruck_models/pvrcnnpp_STrooftop/D1_5epochs_STrooftop_ft_D6_50epochs_ros_06_015_thresh_high_lr/ckpt/checkpoint_epoch_4.pth"
    #checkpoint_path = "/home/cgriesbacher/thesis/3DTrans/output_okeanos/output/avltruck_models/pvrcnnpp_ros/D6_50epochs/ckpt/checkpoint_epoch_50.pth"
    #checkpoint_path = "/home/cgriesbacher/thesis/3DTrans/output_okeanos/output/avlrooftop_models/pvrcnnpp/D1_50epochs/ckpt/checkpoint_epoch_50.pth"

    #zod 
    #checkpoint_path = "/home/cgriesbacher/thesis/3DTrans/output/zod_models/dsvt_pillar/D16_100epochs/ckpt/checkpoint_epoch_100.pth"
    checkpoint_path = "/home/cgriesbacher/thesis/3DTrans/output_okeanos/output/zod_models/pvrcnnpp/D16_50epochs/ckpt/checkpoint_epoch_50.pth"
    #checkpoint_path = "/home/cgriesbacher/thesis/3DTrans/output/zod_models/pvrcnnpp_ros_rbds/D6_50epochs_rbds0.25/ckpt/checkpoint_epoch_50.pth"
    #checkpoint_path = "/home/cgriesbacher/thesis/3DTrans/output_okeanos/output/zod_models/pvrcnnpp_ros_ubds/D16_50epochs_ubds4/ckpt/checkpoint_epoch_50.pth"
    #avltruck
    #checkpoint_path = "/home/cgriesbacher/thesis/3DTrans/output/avltruck_models/centerpoint/D6_100epochs_4classes/ckpt/checkpoint_epoch_100.pth"
    #checkpoint_path = "/home/cgriesbacher/thesis/3DTrans/output_okeanos/output/avltruck_models/pvrcnnpp_ros/D6_50epochs/ckpt/checkpoint_epoch_50.pth"
    #checkpoint_path = "/home/cgriesbacher/thesis/3DTrans/output_okeanos/output/avltruck_models/pvrcnnpp_STzod/D6_5epochs_STzod_ft_D16_50epochs_ros/ckpt/checkpoint_epoch_3.pth"

    # ST avltruck -> zod
    #checkpoint_path = "/home/cgriesbacher/thesis/3DTrans/output_okeanos/output/avltruck_models/pvrcnnpp_STzod/D16_20epochs_STzod_ft_D6_50epochs_fov_gace_labelupdate_nogaceretrain_1/ckpt/checkpoint_epoch_1.pth"
    
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
            if training:
                args.frame_idx = 'sequences/CityStreet_dgt_2021-08-31-14-43-12_0_s0/dataset/logical_frame_000020.json'
            else:
                args.frame_idx = 'sequences/PrimaryHighway_dgt_2021-07-30-11-04-31_0_s0/dataset/logical_frame_000008.json'
        
        image_path_frame = args.frame_idx.split("/")[1] + "_" + args.frame_idx.split("/")[-1].split(".")[0] 

    elif (args.dataset == "zod"):
        cfg_path =  "/home/cgriesbacher/thesis/3DTrans/tools/cfgs/dataset_configs/zod/DA/zod_dataset.yaml"
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
            if training:
                args.frame_idx = '009375'
            else:
                args.frame_idx = "067893"
        
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
            if training:
                args.frame_idx = 'sequences/HIGHWAY_Normal_road_20200427122209_1/unpacked/lidar/0003.pkl'
            else:
                args.frame_idx = 'sequences/HIGHWAY_Rain_road_20200528094334_2/unpacked/lidar/0041.pkl'
        
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
                                    training=training) 
        
        #build model
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
        model.cuda()
        model.eval()

        #calc model params
        #count_parameters(model)

        #find sample index for frame
        sample_id_list = dataset.sample_id_list
        if select_random_frame:
            args.frame_idx = random.choice(sample_id_list)
            print("selected random frame: ", args.frame_idx)

        list_index = sample_id_list.index(args.frame_idx)
        #get data from info files -> is in detector class name space
        data_dict = dataset.__getitem__(list_index)
        if no_detection:
            import tools.visual_utils.open3d_vis_utils as vis
            vis.draw_scenes(data_dict["points"][:,:3], point_colors=data_dict["points"][:,3], gt_boxes=data_dict["gt_boxes"])
            return
        data_dict = dataset.collate_batch([data_dict])

        #get eval frame        
        with torch.no_grad():
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

        annos = dataset.generate_prediction_dicts(
                data_dict, pred_dicts, dataset.class_names, None)
        
        points4 = data_dict['points'].detach().cpu().numpy()[:,1:]  #get rid of batch dim

        eval_metric = "waymo"
        #if not training:
        if True:
            result_str, _, eval_gt_annos, eval_det_annos= dataset.evaluation(det_annos=annos, class_names=dataset.class_names, eval_metric=eval_metric, return_annos=True)
        
            print(result_str)

            print("remaining gt boxes for nontruncated method:", dataset.extract_fov_gt_nontruncated(data_dict["gt_boxes"][0,:,:7].cpu().numpy(), 120, 0).sum())
            print("remaining gt boxes for center method:", dataset.extract_fov_gt(data_dict["gt_boxes"][0,:,:7].cpu().numpy(), 120, 0).sum())
        
            gt_boxes = eval_gt_annos[0]["gt_boxes_lidar"]
            det_boxes = eval_det_annos[0]["boxes_lidar"]

            points4 = dataset.extract_fov_data(points4, 120, 0)

            scores = eval_det_annos[0]["score"]
            gt_boxes_lidar = gt_boxes
        else:
            gt_boxes_lidar = data_dict["gt_boxes"][0].cpu().numpy()
            det_boxes = annos[0]["boxes_lidar"]
            scores = annos[0]["score"]

        points = points4[:,:3]
        color = points4[:,-1]

        #num_beams = 128
        
        #dataset.beam_label_num_beams = num_beams
        #dataset.beam_label_vfov = [-np.pi/16, np.pi/16]
#
        #points_with_beams = dataset.add_beam_labels(points4)
        #label_beams = points_with_beams[:,-1].astype(np.int64)

        #label_beams = segment_lidar_to_beams(points, num_beams)
        #

        #map each beam to a pseudo random color
        #init beam colors with [0, 0.5, 1, 0, 0.5, 1, ...]
        #color_arr = [0, 0.5, 1]
        #beam_colors = np.zeros((num_beams,1))
        #for i in range(num_beams):
        #    beam_colors[i,:]  = np.array([color_arr[i%3]])
        ##map beam labels to colors
        #color = beam_colors[label_beams]

    else:
        dataset, train_loader, train_sampler = build_dataloader(dataset_cfg=dataset_cfg,
                                        class_names=dataset_class_names,
                                        batch_size=1,
                                        dist=False,
                                        workers=0,
                                        logger=logger,
                                        training=training) 
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
        from tools.visual_utils import open3d_vis_utils as vis
        vis.draw_scenes(points, gt_boxes_lidar,det_boxes ,point_colors=color, view_control=view_control, image_path=image_path, ref_scores=scores)
    else:
        from tools.visual_utils import open3d_vis_utils as vis
        vis.draw_scenes(points, gt_boxes_lidar,det_boxes ,point_colors=color, ref_scores=scores)

    return


if __name__ == '__main__':

    main()