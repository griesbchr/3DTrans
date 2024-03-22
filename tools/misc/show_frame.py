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

def plot_pr_curve(result_dict, result_dir, filename="pr_curve.png"):
    import matplotlib.pyplot as plt
    import os
    import seaborn as sns

    #find class names
    class_names = list(set([key.split('_')[0] for key in result_dict if 'AP' in key]))

    #sort class names alphabetically inverse
    class_names.sort(reverse=True)

    #do a subplot for each class in sns for the precision recall curve
    #to two seperate plots side by side for level 1 and level 2
    #the dimentionality of the plot grid thus is 2 x len(class_names)q
    fig, axs = plt.subplots(len(class_names),2, figsize=(20, 10))
    #fig.suptitle('Precision Recall for ' + cfg.TAG + "-" + cfg.EXP_GROUP_PATH + "-" + cfg.extra_tag + "@" + cfg.DATA_CONFIG.DATASET, fontsize=16)
    for i, class_name in enumerate(class_names):
        for j, level in enumerate(['1', '2']):
            #get all keys that contain the class name and level
            keys = [key for key in result_dict if class_name in key and level in key and "PR" in key]
            
            #get first value that contain the class name and level
            values = [result_dict[key] for key in keys][0]

            #get recall and precision values
            recalls = values[:,1]
            precisions = values[:,0]

            #plot recall vs. precision
            axs[i, j].plot(recalls, precisions)
            axs[i, j].set_title(class_name + ' Level ' + level)
            axs[i, j].set_xlabel('Recall')
            axs[i, j].set_ylabel('Precision')
            axs[i, j].set_ylim([0, 1.01])
            axs[i, j].set_xlim([0, 1.01])
            axs[i, j].grid(True)

            #todo plot confusion matrix for new waymo eval

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig_path = os.path.join(result_dir, filename)
    plt.savefig(fig_path)
    print('Saved pr_curve.png to %s' % fig_path)
    plt.close(fig)   
    
def main():
    os.chdir("/home/cgriesbacher/thesis/3DTrans/tools")
    args = parse_args()
    
    save_image = True

    fov=True
    training = False             #enable augmentations
    no_detection = False
    dataset = "zod"
    checkpoint_path = None
    
    select_random_frame = True
    frame_keyword = None

    #avlrooftop
    #checkpoint_path = "/home/cgriesbacher/thesis/3DTrans/output_okeanos/output/avltruck_models/pvrcnnpp_STrooftop/D1_5epochs_STrooftop_ft_D6_50epochs_ros_06_015_thresh_high_lr/ckpt/checkpoint_epoch_4.pth"
    #checkpoint_path = "/home/cgriesbacher/thesis/3DTrans/output/avlrooftop_models/pvrcnnpp/D1_50epochs/ckpt/checkpoint_epoch_50.pth"
    #checkpoint_path = "/home/cgriesbacher/thesis/3DTrans/output/avlrooftop_models/pvrcnnpp_ros_ubus2/D1_50epochs_R2/ckpt/checkpoint_epoch_50.pth"
    #checkpoint_path1 = "/home/cgriesbacher/thesis/3DTrans/output/avlrooftop_models/iassd/D1_50epochs/ckpt/checkpoint_epoch_50.pth"

    #zod 
    #checkpoint_path = "/home/cgriesbacher/thesis/3DTrans/output/zod_models/dsvt_pillar/D16_100epochs/ckpt/checkpoint_epoch_100.pth"
    checkpoint_path = "/home/cgriesbacher/thesis/3DTrans/output/zod_models/pvrcnnpp_ros/D16_50epochs/ckpt/checkpoint_epoch_50.pth"
    #checkpoint_path = "/home/cgriesbacher/thesis/3DTrans/output/zod_models/pvrcnnpp_ros_rbds/D6_50epochs_rbds0.25/ckpt/checkpoint_epoch_50.pth"
    #checkpoint_path = "/home/cgriesbacher/thesis/3DTrans/output_okeanos/output/zod_models/pvrcnnpp_ros_ubds/D16_50epochs_ubds4/ckpt/checkpoint_epoch_50.pth"
    
    #avltruck
    #checkpoint_path = "/home/cgriesbacher/thesis/3DTrans/output/avltruck_models/centerpoint/D6_100epochs_4classes/ckpt/checkpoint_epoch_100.pth"
    #checkpoint_path = "/home/cgriesbacher/thesis/3DTrans/output/avltruck_models/pvrcnnpp/D6_50epochs/ckpt/checkpoint_epoch_50.pth"
    #checkpoint_path = "/home/cgriesbacher/thesis/3DTrans/output_okeanos/avltruck_models/pvrcnnpp_sn2rooftop/D6_50epochs/ckpt/checkpoint_epoch_50.pth"
    #checkpoint_path = "/home/cgriesbacher/thesis/3DTrans/output_okeanos/output/avltruck_models/pvrcnnpp_STzod/D6_5epochs_STzod_ft_D16_50epochs_ros/ckpt/checkpoint_epoch_3.pth"
    
    # ST avltruck -> zod
    #checkpoint_path = "/home/cgriesbacher/thesis/3DTrans/output_okeanos/output/avltruck_models/pvrcnnpp_STzod/D16_20epochs_STzod_ft_D6_50epochs_fov_gace_labelupdate_nogaceretrain_1/ckpt/checkpoint_epoch_1.pth"
    
    checkpoint_path = [checkpoint_path]

    if (args.dataset == None):
        args.dataset = dataset

    if (args.ckpt == None):
        args.ckpt = checkpoint_path
    
    #convert checkpoint path to list
    if not isinstance(args.ckpt, list):
        args.ckpt = [args.ckpt]

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
                args.frame_idx = 'sequences/CityStreet_dgt_2021-07-08-15-18-50_0_s0/dataset/logical_frame_000008.json'
            else:
                args.frame_idx = 'sequences/Motorway_dgt_2021-09-14-14-21-32_0_s0/dataset/logical_frame_000011.json'
        
        image_path_frame = args.frame_idx.split("/")[1] + "_" + args.frame_idx.split("/")[-1].split(".")[0] 

    elif (args.dataset == "zod"):
        cfg_path =  "/home/cgriesbacher/thesis/3DTrans/tools/cfgs/dataset_configs/zod/OD/zod16_dataset_nogtsampling.yaml"
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
                args.frame_idx = "018409"
        
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
                args.frame_idx = 'sequences/CITY_Rain_junction_20200511124748_1/unpacked/lidar/0019.pkl'
            else:
                args.frame_idx = 'sequences/CITY_Normal_roundabout_20200320100220_2/unpacked/lidar/0045.pkl'
        
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

    logger = common_utils.create_logger()
    
    if args.ckpt == None:
        image_path += "/" + image_path_frame + ".png"

    if args.ckpt != None:
        if not isinstance(args.ckpt, list):
            args.ckpt = [args.ckpt]



        det_boxes = []
        det_labels = []
        det_scores = []
        random_frame_set = False
        for i, ckpt in enumerate(args.ckpt):
            train_dataset = ckpt.split("/")[-5]
            train_detector = ckpt.split("/")[-4]
            train_name = ckpt.split("/")[-3]
            train_epoch = ckpt.split("/")[-1].split(".")[0].split("_")[-1]
            #get model config 
            ckpt_path = Path(ckpt)
            cfg_path = [file for file in ckpt_path.parent.parent.glob('*.yaml')]
            assert len(cfg_path) == 1, "More or less than one config file found in "+ckpt_path.parent.parent.__str__()
            cfg_path = cfg_path[0]

            #parse config
            cfg_from_yaml_file(cfg_path, cfg)

            image_path += "/" + train_dataset + "_" + train_detector + "_" + train_name + "_" + train_epoch

            # some detectors use different range or voxel preprocessing
            if hasattr(cfg.DATA_CONFIG, "POINT_CLOUD_RANGE"):
                dataset_cfg.POINT_CLOUD_RANGE = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
            if hasattr(cfg.DATA_CONFIG, "DATA_PROCESSOR"):
                dataset_cfg.DATA_PROCESSOR = cfg.DATA_CONFIG.DATA_PROCESSOR

            #select frame and get data dict
            #build dataset
            dataset, train_loader, train_sampler = build_dataloader(dataset_cfg=dataset_cfg,
                                        class_names=cfg.CLASS_NAMES,
                                        batch_size=1,
                                        dist=False,
                                        workers=0,
                                        logger=logger,
                                        training=training) 
            #find sample index for frame
            sample_id_list = list(dataset.sample_id_list.copy())
            if select_random_frame and not random_frame_set:
                #filter frame ids that contain a keyword
                if frame_keyword is not None:
                    sample_id_list = [sample_id for sample_id in sample_id_list if frame_keyword in sample_id]
                
                args.frame_idx = random.choice(sample_id_list)
                print("\nselected random frame: ", args.frame_idx, "\n")
                random_frame_set = True

            list_index = sample_id_list.index(args.frame_idx)

            #get data from info files -> is in detector class name space
            data_dict = dataset.__getitem__(list_index)
            if no_detection:
                import tools.visual_utils.open3d_vis_utils as vis
                vis.draw_scenes(data_dict["points"][:,:3], point_colors=data_dict["points"][:,3], gt_boxes=data_dict["gt_boxes"])
                return
            data_dict = dataset.collate_batch([data_dict])


            #get model config 
            ckpt_path = Path(ckpt)
            cfg_path = [file for file in ckpt_path.parent.parent.glob('*.yaml')]
            assert len(cfg_path) == 1, "More or less than one config file found in "+ckpt_path.parent.parent.__str__()
            cfg_path = cfg_path[0]

            #parse config
            cfg_from_yaml_file(cfg_path, cfg)

            #build model
            model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
            model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=False)
            model.cuda()
            model.eval()

            #get eval frame        
            with torch.no_grad():
                load_data_to_gpu(data_dict)
                pred_dicts, _ = model.forward(data_dict)

            annos = dataset.generate_prediction_dicts(
                    data_dict, pred_dicts, dataset.class_names, None)
            
            points4 = data_dict['points'].detach().cpu().numpy()[:,1:]  #get rid of batch dim

            eval_metric = "waymo"
            if not training:
                result_str, results_dict, eval_gt_annos, eval_det_annos= dataset.evaluation(det_annos=annos, class_names=dataset.class_names, eval_metric=eval_metric, return_annos=True)
                plot_pr_curve(results_dict, "./", filename="pr_curve"+str(i)+".png")
                print(result_str)

                print("remaining gt boxes for nontruncated method:", dataset.extract_fov_gt_nontruncated(data_dict["gt_boxes"][0,:,:7].cpu().numpy(), 120, 0).sum())
                print("remaining gt boxes for center method:", dataset.extract_fov_gt(data_dict["gt_boxes"][0,:,:7].cpu().numpy(), 120, 0).sum())
            
                gt_boxes = eval_gt_annos[0]["gt_boxes_lidar"]
                det_boxes.append(eval_det_annos[0]["boxes_lidar"])
                det_labels.append(np.ones(eval_det_annos[0]["boxes_lidar"].shape[0])*i)
                points4 = dataset.extract_fov_data(points4, 120, 0)

                det_scores.append(eval_det_annos[0]["score"])
                gt_boxes_lidar = gt_boxes
            else:
                gt_boxes_lidar = data_dict["gt_boxes"][0].cpu().numpy()
                det_boxes.append(annos[0]["boxes_lidar"])
                det_labels.append(np.ones(det_boxes[i].shape[0])*i)
                det_scores.append(annos[0]["score"])

            points = points4[:,:3]
            color = points4[:,-1]

        det_boxes = np.concatenate(det_boxes, axis=0)
        det_labels = np.concatenate(det_labels, axis=0)
        det_labels = det_labels.astype(np.int64)
        det_scores = np.concatenate(det_scores, axis=0)

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
        det_labels=None
        det_scores = None

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
        vis.draw_scenes(points, gt_boxes_lidar, det_boxes, det_labels=det_labels, point_colors=color, view_control=view_control, image_path=image_path, det_scores=det_scores)
    else:
        from tools.visual_utils import open3d_vis_utils as vis
        vis.draw_scenes(points, gt_boxes_lidar,det_boxes ,point_colors=color, det_scores=det_scores)

    return


if __name__ == '__main__':

    main()