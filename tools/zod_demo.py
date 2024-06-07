from pathlib import Path
import yaml
from easydict import EasyDict
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from pcdet.models.detectors.pv_rcnn_plusplus import PVRCNNPlusPlus
from pcdet.datasets.zod.zod_dataset import ZODDataset

from pcdet.utils import common_utils
from pcdet.config import cfg, cfg_from_yaml_file

import os
    
def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib']:
            continue
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        elif key in ['db_flag']:
            continue
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()

def main():

    #3DTrans/tools as working directory
    os.chdir("/home/cgriesbacher/thesis/3DTrans/tools")
 
    no_detection=False
    checkpoint_path = "../ckpts/zod/pvrcnnpp/checkpoint_epoch_50.pth"
    frame_idx = "018409"
    cfg_path =  "cfgs/dataset_configs/zod/OD/zod_dataset.yaml"
    model_cfg = "cfgs/zod_models/pvrcnnpp.yaml"
    
    cfg_path = Path(cfg_path).resolve()
    dataset_cfg = EasyDict(yaml.safe_load(open(cfg_path.absolute())))

    cfg_path = Path(model_cfg).resolve()
    model_cfg = EasyDict(yaml.safe_load(open(cfg_path.absolute())))

    detector_class_names = ['Vehicle', 'Pedestrian', 'Cyclist', 'Truck']
    #parse config
    cfg_from_yaml_file(cfg_path, cfg)

    logger = common_utils.create_logger()

    #build dataset
    dataset = ZODDataset(dataset_cfg=dataset_cfg, class_names=detector_class_names, training=False, logger=logger)

    #find sample index for frame
    sample_id_list = list(dataset.sample_id_list.copy())
    list_index = sample_id_list.index(frame_idx)

    #get data from info files
    data_dict = dataset.__getitem__(list_index)
    if no_detection:
        import tools.visual_utils.open3d_vis_utils as vis
        points = data_dict["points"][:,:3]
        color = data_dict["points"][:,3]
        gt_boxes_lidar = data_dict["gt_boxes"]

        vis.draw_scenes(points, point_colors=color, gt_boxes=gt_boxes_lidar)

        return
    
    data_dict = dataset.collate_batch([data_dict])

    #get model config 
    ckpt_path = Path(checkpoint_path).resolve()

    #build model
    model = PVRCNNPlusPlus(model_cfg.MODEL, len(model_cfg.CLASS_NAMES), dataset)

    #load pretrained model
    model.load_params_from_file(filename=ckpt_path, logger=logger, to_cpu=False)
   
    model.cuda()
    model.eval()

    #get eval frame        
    with torch.no_grad():
        load_data_to_gpu(data_dict)
        pred_dicts, _ = model.forward(data_dict)
    
    #de-batchify predictions
    annos = dataset.generate_prediction_dicts(
            data_dict, pred_dicts, dataset.class_names, None)
    
    points4 = data_dict['points'].detach().cpu().numpy()[:,1:]  #get rid of batch dim

    #evaluate for AP values with waymo evaluation method
    eval_metric = "waymo"
    result_str, results_dict, eval_gt_annos, eval_det_annos= dataset.evaluation(det_annos=annos, class_names=dataset.class_names, eval_metric=eval_metric, return_annos=True)
    #print(result_str)

    points = points4[:,:3]
    color = points4[:,-1]

    det_boxes = annos[0]["boxes_lidar"]
    det_scores = annos[0]["score"]
    gt_boxes = data_dict["gt_boxes"][0,:,:7]

    det_boxes = []
    det_scores = []

    gt_boxes = eval_gt_annos[0]["gt_boxes_lidar"]
    det_boxes.append(eval_det_annos[0]["boxes_lidar"])
    det_scores.append(eval_det_annos[0]["score"])

    det_boxes = np.concatenate(det_boxes, axis=0)
    det_scores = np.concatenate(det_scores, axis=0)


    from tools.visual_utils import open3d_vis_utils as vis
    vis.draw_scenes(points, gt_boxes,det_boxes ,point_colors=color, det_scores=det_scores)

    return


if __name__ == '__main__':

    main()