import tqdm
import numpy as np
import torch
import pickle
from pathlib import Path
from copy import deepcopy
from easydict import EasyDict as edict
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import pairwise_distances
from torch.utils.data import Dataset

#from pcdet.datasets import build_dataloader 
#from pcdet.models import build_network, load_data_to_gpu
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
from pcdet.models.model_utils.model_nms_utils import class_agnostic_nms


#with elongation and without truck class
#INSTANCE_PROP = [
#    'class_veh', 'class_ped', 'class_cyc', 'base_det_score', 
#    'cx', 'cy', 'cz', 'dx', 'dy', 'dz', 'heading_cos', 'heading_sin', 
#    'dist', 'alpha_cos', 'alpha_sin', 'nr_pts',
#    'min_x', 'min_y', 'min_z', 'min_refl', 'min_elongation',
#    'max_x', 'max_y', 'max_z', 'max_refl', 'max_elongation',
#    'mean_x', 'mean_y', 'mean_z', 'mean_refl', 'mean_elongation',
#    'std_x', 'std_y', 'std_z', 'std_refl', 'std_elongation',
#]

INSTANCE_PROP = [
     'class_veh', 'class_ped', 'class_cyc', 'class_tru', 'base_det_score', 
     'cx', 'cy', 'cz', 'dx', 'dy', 'dz', 'heading_cos', 'heading_sin', 
     'dist', 'alpha_cos', 'alpha_sin', 'nr_pts',
     'min_x', 'min_y', 'min_z', 'min_refl',
     'max_x', 'max_y', 'max_z', 'max_refl', 
     'mean_x', 'mean_y', 'mean_z', 'mean_refl', 
     'std_x', 'std_y', 'std_z', 'std_refl',
 ]

#no nr points
#INSTANCE_PROP = [
#    'class_veh', 'class_ped', 'class_cyc', 'class_tru', 'base_det_score', 
#    'cx', 'cy', 'cz', 'dx', 'dy', 'dz', 'heading_cos', 'heading_sin', 
#    'dist', 'alpha_cos', 'alpha_sin',
#    'min_x', 'min_y', 'min_z', 'min_refl',
#    'max_x', 'max_y', 'max_z', 'max_refl', 
#    'mean_x', 'mean_y', 'mean_z', 'mean_refl', 
#    'std_x', 'std_y', 'std_z', 'std_refl',
#]

#only car
#INSTANCE_PROP = [
#    'class_veh', 'base_det_score', 
#    'cx', 'cy', 'cz', 'dx', 'dy', 'dz', 'heading_cos', 'heading_sin', 
#    'dist', 'alpha_cos', 'alpha_sin', 'nr_pts',
#    'min_x', 'min_y', 'min_z', 'min_refl',
#    'max_x', 'max_y', 'max_z', 'max_refl', 
#    'mean_x', 'mean_y', 'mean_z', 'mean_refl', 
#    'std_x', 'std_y', 'std_z', 'std_refl',
#]

#without detection score
#INSTANCE_PROP = [
#    'class_veh', 'class_ped', 'class_cyc', 'class_tru', 
#    'cx', 'cy', 'cz', 'dx', 'dy', 'dz', 'heading_cos', 'heading_sin', 
#    'dist', 'alpha_cos', 'alpha_sin', 'nr_pts',
#    'min_x', 'min_y', 'min_z', 'min_refl',
#    'max_x', 'max_y', 'max_z', 'max_refl', 
#    'mean_x', 'mean_y', 'mean_z', 'mean_refl', 
#    'std_x', 'std_y', 'std_z', 'std_refl',
#]
#
CONTEXT_PROP = [
    'dist', 'dir_to_nb_x', 'dir_to_nb_y', 'dir_to_nb_z',
    'diff_heading_cos', 'diff_heading_sin', 'nb_det_scores',
    'nb_class_veh', 'nb_class_ped', 'nb_class_cyc', 'nb_class_tru'
]

#only car
#CONTEXT_PROP = [
#    'dist', 'dir_to_nb_x', 'dir_to_nb_y', 'dir_to_nb_z',
#    'diff_heading_cos', 'diff_heading_sin', 'nb_det_scores',
#    'nb_class_veh'
#]

# category: 1 true positive detection, 0 false positive detection
TARGETS = ['category', 'iou_w_gt']


class GACEDataset(Dataset):

    def __init__(self, cfg, logger, train):
        
        self.cfg = cfg
        self.logger = logger
        self.train = train

        self.context_radius = cfg.SELF_TRAIN.GACE.CONTEXT_RADIUS
        self.max_nr_neighbors = cfg.SELF_TRAIN.GACE.MAX_NR_NEIGHBORS
        
        # create instance property index dictionary for data array
        self.ip_dict = edict({k: i for i, k in enumerate(INSTANCE_PROP)})
        self.cp_dict = edict({k: i for i, k in enumerate(CONTEXT_PROP)})
        self.target_dict = edict({k: i for i, k in enumerate(TARGETS)})
        
        return

    
    def get_ip_dim(self):
        return len(self.ip_dict)


    def get_cp_dim(self):
        return len(self.cp_dict)


    def get_target_dim(self):
        return len(self.target_dict)


    def get_det_annos(self):
        return self.det_annos


    def __len__(self):
        return self.ip_data.shape[0]


    def __getitem__(self, idx):
        ipd = self.ip_dict
        cpd = self.cp_dict

        ip_data = self.ip_data[idx, :]
        cp_nb_ids = self.cp_nb_ids[idx, :]
        target_data = self.target_data[idx, :]
        
        #add noise to ip_data base_det_score
        #if self.train:
        #   noise = np.random.uniform(-0.1, 0.1, size=(ip_data.shape[0]))
        #   score_mask = ip_data[:, ipd.base_det_score] > 0
        #   noise[np.logical_not(score_mask)] = 0
#
        #   ip_data[:, ipd.base_det_score] += noise
#
        #   ##clip scores to [0.1, 1]
        #   ip_data[:, ipd.base_det_score] = np.clip(ip_data[:, ipd.base_det_score], 0.1, 1)
            

        box_center = ip_data[:, [ipd.cx, ipd.cy, ipd.cz]]
        box_heading_dir = ip_data[:, [ipd.heading_cos, ipd.heading_sin]]
        box_heading_angle = np.arctan2(box_heading_dir[:, 1], box_heading_dir[:, 0])

        mask = cp_nb_ids == -1
        
        nb_ip_data = self.ip_data[cp_nb_ids, :]
        nb_ip_data[mask] = 0

        cp_data = np.zeros((len(idx), self.max_nr_neighbors, len(cpd)), dtype=np.float32)

        nb_box_center = nb_ip_data[:, :, [ipd.cx, ipd.cy, ipd.cz]]
        nb_box_heading_dir = nb_ip_data[:, :, [ipd.heading_cos, ipd.heading_sin]]
        nb_box_heading_angle = np.arctan2(nb_box_heading_dir[:, :, 1], nb_box_heading_dir[:, :, 0])
        
        dir_to_nb = nb_box_center - box_center[:, None, :]
        dist = np.linalg.norm(dir_to_nb, axis=2)

        temp = deepcopy(dist)
        temp[temp == 0] = 1
        
        dir_to_nb = dir_to_nb / temp[:, :, None]

        cp_data[:, :, cpd.dist] = dist
        cp_data[:, :, cpd.dir_to_nb_x] = dir_to_nb[:, :, 0]
        cp_data[:, :, cpd.dir_to_nb_y] = dir_to_nb[:, :, 1]
        cp_data[:, :, cpd.dir_to_nb_z] = dir_to_nb[:, :, 2]

        cp_data[:, :, cpd.diff_heading_cos] = np.cos(box_heading_angle[:, None] - nb_box_heading_angle)
        cp_data[:, :, cpd.diff_heading_sin] = np.sin(box_heading_angle[:, None] - nb_box_heading_angle)

        cp_data[:, :, cpd.nb_det_scores] = nb_ip_data[:, :, ipd.base_det_score]

        cp_data[:, :, cpd.nb_class_veh] = nb_ip_data[:, :, ipd.class_veh]
        cp_data[:, :, cpd.nb_class_ped] = nb_ip_data[:, :, ipd.class_ped]
        cp_data[:, :, cpd.nb_class_cyc] = nb_ip_data[:, :, ipd.class_cyc]
        cp_data[:, :, cpd.nb_class_tru] = nb_ip_data[:, :, ipd.class_tru]

        cp_data[mask, :] = 0

        ip_data_n = deepcopy(ip_data)
        nb_ip_data_n = deepcopy(nb_ip_data)
        cp_data_n = deepcopy(cp_data)
        
        for ip_name in ipd.keys():
            if ip_name in self.cfg.SELF_TRAIN.GACE.NORM_FACTORS:
                norm_factor = self.cfg.GACE.SELF_TRAIN.NORM_FACTORS[ip_name]
                
                if isinstance(norm_factor, list):
                    norm_factor_veh = norm_factor[0]
                    norm_factor_ped = norm_factor[1]
                    norm_factor_cyc = norm_factor[2]
                    norm_factor_tru = norm_factor[3]

                    veh_mask = ip_data[:, ipd.class_veh] == 1
                    ped_mask = ip_data[:, ipd.class_ped] == 1
                    cyc_mask = ip_data[:, ipd.class_cyc] == 1
                    tru_mask = ip_data[:, ipd.class_tru] == 1
                    ip_data_n[veh_mask, ipd[ip_name]] /= norm_factor_veh 
                    ip_data_n[ped_mask, ipd[ip_name]] /= norm_factor_ped
                    ip_data_n[cyc_mask, ipd[ip_name]] /= norm_factor_cyc
                    ip_data_n[tru_mask, ipd[ip_name]] /= norm_factor_tru

                    veh_mask_nb = nb_ip_data[:, :, ipd.class_veh] == 1
                    ped_mask_nb = nb_ip_data[:, :, ipd.class_ped] == 1
                    cyc_mask_nb = nb_ip_data[:, :, ipd.class_cyc] == 1
                    tru_mask_nb = nb_ip_data[:, :, ipd.class_tru] == 1
                    nb_ip_data_n[veh_mask_nb, ipd[ip_name]] /= norm_factor_veh
                    nb_ip_data_n[ped_mask_nb, ipd[ip_name]] /= norm_factor_ped
                    nb_ip_data_n[cyc_mask_nb, ipd[ip_name]] /= norm_factor_cyc
                    nb_ip_data_n[tru_mask_nb, ipd[ip_name]] /= norm_factor_tru

                else:
                    ip_data_n[:, ipd[ip_name]] /= norm_factor
                    nb_ip_data_n[:, :, ipd[ip_name]] /= norm_factor

        for cp_name in cpd.keys():
            if cp_name in self.cfg.SELF_TRAIN.GACE.NORM_FACTORS_CP:
                norm_factor = self.cfg.GACE.SELF_TRAIN.NORM_FACTORS_CP[cp_name]
                cp_data_n[:, :, cpd[cp_name]] /= norm_factor

        nb_ip_data_n = np.swapaxes(nb_ip_data_n, 1, 2)
        cp_data_n = np.swapaxes(cp_data_n, 1, 2)

        target_category = target_data[:, self.target_dict.category]
        target_iou = target_data[:, self.target_dict.iou_w_gt]

        ip_data_n = torch.tensor(ip_data_n[..., None, None], dtype=torch.float32)
        nb_ip_data_n = torch.tensor(nb_ip_data_n[..., None], dtype=torch.float32)
        cp_data_n = torch.tensor(cp_data_n[..., None], dtype=torch.float32)
        target_category = torch.tensor(target_category, dtype=torch.float32)
        target_iou = torch.tensor(target_iou, dtype=torch.float32)

        return ip_data_n, cp_data_n, nb_ip_data_n, target_category, target_iou


    def de_collate_batch(self, batch_dict, pred_dicts):

        # points in boxes (over entire batch) to reduce pcl later 
        max_nr_dets = max([pred_dict['pred_labels'].shape[0] for pred_dict in pred_dicts])
        max_nr_pts = torch.unique(batch_dict['points'][:, 0], return_counts=True)[1].max()
        batch_pts = torch.zeros((len(pred_dicts), max_nr_pts, batch_dict['points'].shape[1]-1), 
                                dtype=torch.float32).cuda()
        batch_boxes = torch.zeros((len(pred_dicts), max_nr_dets, 7), dtype=torch.float32).cuda()

        for j, pred_dict in enumerate(pred_dicts):
            pcl_mask = batch_dict['points'][:, 0] == j
            batch_pts[j, :pcl_mask.sum(), :] = batch_dict['points'][pcl_mask, 1:]
            batch_pts[j, pcl_mask.sum():, :] = torch.inf
            batch_boxes[j, :pred_dict['pred_labels'].shape[0], :] = pred_dict['pred_boxes']

        box_ids_of_pts = points_in_boxes_gpu(batch_pts[:, :, :3], batch_boxes)
        
        sample_data = []

        for j, pred_dict in enumerate(pred_dicts):
            sample_dict = edict()

            # extract detection data
            sample_dict.det_boxes = pred_dict['pred_boxes'].cpu().numpy()
            sample_dict.det_scores = pred_dict['pred_scores'].cpu().numpy()
            sample_dict.det_labels = pred_dict['pred_labels'].cpu().numpy()
            
            if sample_dict.det_boxes.shape[0] == 0:
                continue

            # extract pcl and GT data
            nr_boxes = sample_dict.det_labels.shape[0]

            mask = torch.logical_and(box_ids_of_pts[j, :] >= 0, 
                                     box_ids_of_pts[j, :] < nr_boxes).cpu().numpy()
            sample_dict.pcl = batch_pts.cpu().numpy()[j, mask, :]
            gt_boxes = batch_dict['gt_boxes'][j, :, :].cpu().numpy()
            gt_boxes_mask = gt_boxes[:, -1] > 0
            sample_dict.gt_labels = gt_boxes[gt_boxes_mask, -1]
            sample_dict.gt_boxes = gt_boxes[gt_boxes_mask, :-1]
        
            sample_data.append(sample_dict)

        return sample_data
    

    def extract_context_neighbors(self, sample_dict):
        det_boxes = sample_dict.det_boxes
        det_scores = sample_dict.det_scores

        data = np.zeros((det_boxes.shape[0], self.max_nr_neighbors), dtype=np.int32) - 1
        
        distance_matrix = pairwise_distances(det_boxes[:, :3])

        sorted_idx = np.argsort(distance_matrix, axis=1)[:, 1:]

        # Replace indices with -1 if the corresponding distance is greater than max_dist
        nearest_nb = np.where(
            distance_matrix[np.arange(distance_matrix.shape[0])[:, None], sorted_idx] > self.context_radius,
            -1, sorted_idx)
        
        nms_temp_cfg = deepcopy(self.cfg.MODEL.POST_PROCESSING.NMS_CONFIG)
        nms_temp_cfg.NMS_THRESH = 0.1
        relevant_ids = class_agnostic_nms(torch.from_numpy(det_scores).cuda(),
                                     torch.from_numpy(det_boxes).cuda(),
                                     nms_temp_cfg)[0].cpu().numpy()
        
        nearest_nb_nms = deepcopy(nearest_nb)
        nearest_nb_nms[np.isin(nearest_nb, relevant_ids, invert=True)] = -1

        if nearest_nb_nms.shape[1] > self.max_nr_neighbors:
            nearest_nb_nms = nearest_nb_nms[:, :self.max_nr_neighbors]

        data[:, :nearest_nb_nms.shape[1]] = nearest_nb_nms
        
        return data


    def extract_target_data(self, sample_dict):
        det_boxes = sample_dict.det_boxes
        det_labels = sample_dict.det_labels
        gt_boxes = sample_dict.gt_boxes
        gt_labels = sample_dict.gt_labels

        td = self.target_dict

        data = np.zeros((det_boxes.shape[0], len(td)), dtype=np.float32)
            
        if gt_boxes.shape[0] > 0:
            category = np.zeros_like(det_labels, dtype=np.float32)
            
            iou = boxes_iou3d_gpu(torch.from_numpy(det_boxes).cuda(), 
                                  torch.from_numpy(gt_boxes).cuda()).cpu().numpy()

            same_class = det_labels[:, None] == gt_labels[None, :]
            iou = iou * same_class

            category[det_labels == 1] = np.max((iou[det_labels==1, :] >= 0.7).astype(np.float32), axis=1)
            category[det_labels != 1] = np.max((iou[det_labels!=1, :] >= 0.5).astype(np.float32), axis=1)

            data[:, td.iou_w_gt] = np.max(iou, axis=1)
            data[:, td.category] = category
        
        return data


    def extract_instance_properties(self, sample_dict):
        det_boxes = sample_dict.det_boxes
        det_scores = sample_dict.det_scores
        det_labels = sample_dict.det_labels
        pcl = sample_dict.pcl
        
        ipd = self.ip_dict
        
        data = np.zeros((det_boxes.shape[0], len(ipd)), dtype=np.float32)

        det_boxes[:, 6] = np.mod(det_boxes[:, 6], 2*np.pi)

        data[:, ipd.class_veh] = (det_labels == 1).astype(np.float32)
        data[:, ipd.class_ped] = (det_labels == 2).astype(np.float32)
        data[:, ipd.class_cyc] = (det_labels == 3).astype(np.float32)
        data[:, ipd.class_tru] = (det_labels == 4).astype(np.float32)

        data[:, ipd.base_det_score] = det_scores

        data[:, ipd.cx] = det_boxes[:, 0]
        data[:, ipd.cy] = det_boxes[:, 1]
        data[:, ipd.cz] = det_boxes[:, 2]
        data[:, ipd.dx] = det_boxes[:, 3]
        data[:, ipd.dy] = det_boxes[:, 4]
        data[:, ipd.dz] = det_boxes[:, 5]

        data[:, ipd.heading_cos] = np.cos(det_boxes[:, 6])
        data[:, ipd.heading_sin] = np.sin(det_boxes[:, 6])
        
        alpha = det_boxes[:, 6] - np.arctan2(det_boxes[:, 1], det_boxes[:, 0])
        data[:, ipd.alpha_cos] = np.cos(alpha)
        data[:, ipd.alpha_sin] = np.sin(alpha)

        data[:, ipd.dist] = np.linalg.norm(det_boxes[:, :3], axis=1)

        pcl = torch.from_numpy(pcl).cuda()

        # create point statistics
        for i in range(det_boxes.shape[0]):
            box = deepcopy(det_boxes[i, :])
            mask = points_in_boxes_gpu(pcl[None, :, :3], torch.from_numpy(box[None, None, :]).cuda())
            mask = mask[0, :] == 0
            
            pcl_in_box = deepcopy(pcl[mask, :].cpu().numpy())
            
            if pcl_in_box.shape[0] == 0:
                continue

            data[i, ipd.nr_pts] = pcl_in_box.shape[0]
            
            # move to center
            pcl_in_box[:, :3] -= box[:3]
            box[:3] = 0

            # rotate to align with x-axis
            rotmat = R.from_euler('z', box[6], degrees=False).as_matrix()
            
            pcl_in_box[:, :3] = np.matmul(pcl_in_box[:, :3], rotmat)
            box[6] = 0

            # scale to unit box
            pcl_in_box[:, :3] /= box[3:6]
            box[3:6] = 1

            data[i, ipd.min_x] = np.min(pcl_in_box[:, 0])
            data[i, ipd.min_y] = np.min(pcl_in_box[:, 1])
            data[i, ipd.min_z] = np.min(pcl_in_box[:, 2])
            data[i, ipd.min_refl] = np.min(pcl_in_box[:, 3])
            #data[i, ipd.min_elongation] = np.min(pcl_in_box[:, 4])

            data[i, ipd.max_x] = np.max(pcl_in_box[:, 0])
            data[i, ipd.max_y] = np.max(pcl_in_box[:, 1])
            data[i, ipd.max_z] = np.max(pcl_in_box[:, 2])
            data[i, ipd.max_refl] = np.max(pcl_in_box[:, 3])
            #data[i, ipd.max_elongation] = np.max(pcl_in_box[:, 4])

            data[i, ipd.mean_x] = np.mean(pcl_in_box[:, 0])
            data[i, ipd.mean_y] = np.mean(pcl_in_box[:, 1])
            data[i, ipd.mean_z] = np.mean(pcl_in_box[:, 2])
            data[i, ipd.mean_refl] = np.mean(pcl_in_box[:, 3])
            #data[i, ipd.mean_elongation] = np.mean(pcl_in_box[:, 4])

            data[i, ipd.std_x] = np.std(pcl_in_box[:, 0])
            data[i, ipd.std_y] = np.std(pcl_in_box[:, 1])
            data[i, ipd.std_z] = np.std(pcl_in_box[:, 2])
            data[i, ipd.std_refl] = np.std(pcl_in_box[:, 3])
            #data[i, ipd.std_elongation] = np.std(pcl_in_box[:, 4])

        return data


    def process_data_batch(self, batch_dict, pred_dicts):
        
        #model_cfg = deepcopy(self.cfg.MODEL)
        #
        ## INIT DATASET AND DATALOADER
        #if self.train:
        #    data_cfg = deepcopy(self.cfg.DATA_CONFIG) 
        #    # inference on training set with base detector without any augmentation
        #    # therefore we use the test cfg for training set inference
        #    #data_cfg.DATA_SPLIT['test'] = data_cfg.DATA_SPLIT['train']
        #    #data_cfg.SAMPLED_INTERVAL['test'] = data_cfg.SAMPLED_INTERVAL['train']
        #    model_cfg.POST_PROCESSING.NMS_CONFIG.NMS_THRESH = 0.1
        #    tqdm_desc = '[GACE] train data generation'
        #    dataset, dataloader, sampler = build_dataloader(
        #        dataset_cfg=data_cfg, class_names=self.cfg.CLASS_NAMES, 
        #        batch_size=self.args.batch_size_dg, dist=False, 
        #        workers=self.args.workers, logger=self.logger, training=True)
        #    dataloader.dataset.eval()
        #else:
        #    if self.cfg.DATA_CONFIG_TAR is not None:
        #        data_cfg = deepcopy(self.cfg.DATA_CONFIG_TAR)
        #    else:
        #        data_cfg = deepcopy(self.cfg.DATA_CONFIG)
        #    tqdm_desc = '[GACE] val data generation'
        #    det_annos = []
        #    dataset, dataloader, sampler = build_dataloader(
        #        dataset_cfg=data_cfg, class_names=self.cfg.CLASS_NAMES, 
        #        batch_size=self.args.batch_size_dg, dist=False, 
        #        workers=self.args.workers, logger=self.logger, training=False)
        #    dataloader.dataset.eval()
#
#
        #
        ## INIT BASE DETECTOR
        #base_det_model = build_network(model_cfg=model_cfg, num_class=len(self.cfg.CLASS_NAMES), 
        #                               dataset=dataset)
        ##dataset.class_names = ['Vehicle', 'Pedestrian', 'Cyclist', 'Truck']
        #base_det_model = build_network(model_cfg=model_cfg, num_class=4, 
        #                               dataset=dataset)
        #base_det_model.load_params_from_file(filename=self.args.base_detector_ckpt, logger=self.logger, 
        #                                     to_cpu=False)
        #base_det_model.cuda()
        #base_det_model.eval()

        # GENERATE DATA ARRAYS
        nr_max_det_per_sample = self.cfg.SELF_TRAIN.GACE.NR_MAX_DET_PER_SAMPLE

        est_shape = (len(batch_dict["frame_id"]) * nr_max_det_per_sample, len(self.ip_dict))
        ip_data = np.zeros(est_shape, dtype=np.float32)
        
        est_shape = (len(batch_dict["frame_id"]) * nr_max_det_per_sample, len(self.target_dict))
        target_data = np.zeros(est_shape, dtype=np.float32)

        est_shape = (len(batch_dict["frame_id"]) * nr_max_det_per_sample, self.max_nr_neighbors)
        cp_nb_ids = np.zeros(est_shape, dtype=np.int32)

        # LOOP OVER DATASET AND EXTRACT GEOMETRIC PROPERTIES
        #progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, 
        #                         desc=tqdm_desc, dynamic_ncols=True, 
        #                         smoothing=0.1)
        sample_idx = 0
        #for batch_dict in dataloader:
        #    load_data_to_gpu(batch_dict)
        #    
        #    with torch.no_grad():
        #        pred_dicts, _ = base_det_model(batch_dict)

            ##loop over all detections and replace gt box 
            #for batch_idx, pred_dict in enumerate(pred_dicts):
            #    for cls in range(1, len(self.cfg.CLASS_NAMES)+1):
            #        det_mask = pred_dict['pred_labels'] == cls
            #        if det_mask.sum() == 0:
            #            continue
#
            #        gt_mask = batch_dict['gt_boxes'][batch_idx, :, -1] == cls
            #        if gt_mask.sum() == 0:
            #            continue
#
            #        det_boxes = pred_dict['pred_boxes'][det_mask, :].cpu().numpy()
            #        gt_boxes = batch_dict['gt_boxes'][batch_idx, gt_mask, :-1].cpu().numpy()
            #        iou = boxes_iou3d_gpu(torch.from_numpy(det_boxes).cuda(),
            #                                torch.from_numpy(gt_boxes).cuda()).cpu().numpy()
            #        max_ious = np.max(iou, axis=1)
            #        max_iou_inds = np.argmax(iou, axis=1)
#
            #        for i in range(det_boxes.shape[0]):
            #            if max_ious[i] > 0.1:
            #                gt_box = gt_boxes[max_iou_inds[i], :]
            #                det_mask_indices = torch.nonzero(det_mask)
            #                pred_dict['pred_boxes'][det_mask_indices[i], :] = torch.from_numpy(gt_box).cuda()


        #if not self.train:
        #    # store annos for post evaluation
        #    #class_names = ['Vehicle', 'Pedestrian', 'Cyclist', 'Truck']
        #    class_names = self.cfg.CLASS_NAMES
        #    annos = dataset.generate_prediction_dicts(
        #        deepcopy(batch_dict), deepcopy(pred_dicts), class_names)
        #    det_annos += annos
        
        # de-collate batch
        sample_data = self.de_collate_batch(batch_dict, pred_dicts)

        #if len(sample_data) == 0:
        #    continue 
        
        for sample_dict in sample_data:
            start_idx = sample_idx
            stop_idx = sample_idx + sample_dict.det_labels.shape[0]

            sample_ip_data = self.extract_instance_properties(sample_dict) 
            ip_data[start_idx:stop_idx, :] = sample_ip_data

            sample_target_data = self.extract_target_data(sample_dict)
            target_data[start_idx:stop_idx, :] = sample_target_data

            sample_cp_nb_ids = self.extract_context_neighbors(sample_dict)
            sample_cp_nb_ids[sample_cp_nb_ids != -1] += start_idx
            cp_nb_ids[start_idx:stop_idx, :] = sample_cp_nb_ids
            
            sample_idx += sample_dict.det_labels.shape[0]

            #progress_bar.update()

        #progress_bar.close()

        # trim array
        ip_data = ip_data[:sample_idx, :]
        target_data = target_data[:sample_idx, :]
        cp_nb_ids = cp_nb_ids[:sample_idx, :]

        # save data to pickle file
        data = edict()
        data.ip_data = ip_data
        data.target_data = target_data
        data.cp_nb_ids = cp_nb_ids
        
        #with open(self.data_file, 'wb') as f:
        #    pickle.dump(data, f)
#
        #if not self.train:
        #    with open(self.det_annos_file, 'wb') as f:
        #        pickle.dump(det_annos, f)
#
        #    self.det_annos = det_annos
        ipd = self.ip_dict
        cpd = self.cp_dict

        self.ip_data = ip_data
        self.cp_nb_ids = cp_nb_ids
        self.target_data = target_data

        box_center = ip_data[:, [ipd.cx, ipd.cy, ipd.cz]]
        box_heading_dir = ip_data[:, [ipd.heading_cos, ipd.heading_sin]]
        box_heading_angle = np.arctan2(box_heading_dir[:, 1], box_heading_dir[:, 0])

        mask = cp_nb_ids == -1
        
        nb_ip_data = self.ip_data[cp_nb_ids, :]
        nb_ip_data[mask] = 0

        cp_data = np.zeros((ip_data.shape[0], self.max_nr_neighbors, len(cpd)), dtype=np.float32)

        nb_box_center = nb_ip_data[:, :, [ipd.cx, ipd.cy, ipd.cz]]
        nb_box_heading_dir = nb_ip_data[:, :, [ipd.heading_cos, ipd.heading_sin]]
        nb_box_heading_angle = np.arctan2(nb_box_heading_dir[:, :, 1], nb_box_heading_dir[:, :, 0])
        
        dir_to_nb = nb_box_center - box_center[:, None, :]
        dist = np.linalg.norm(dir_to_nb, axis=2)

        temp = deepcopy(dist)
        temp[temp == 0] = 1
        
        dir_to_nb = dir_to_nb / temp[:, :, None]

        cp_data[:, :, cpd.dist] = dist
        cp_data[:, :, cpd.dir_to_nb_x] = dir_to_nb[:, :, 0]
        cp_data[:, :, cpd.dir_to_nb_y] = dir_to_nb[:, :, 1]
        cp_data[:, :, cpd.dir_to_nb_z] = dir_to_nb[:, :, 2]

        cp_data[:, :, cpd.diff_heading_cos] = np.cos(box_heading_angle[:, None] - nb_box_heading_angle)
        cp_data[:, :, cpd.diff_heading_sin] = np.sin(box_heading_angle[:, None] - nb_box_heading_angle)

        cp_data[:, :, cpd.nb_det_scores] = nb_ip_data[:, :, ipd.base_det_score]

        cp_data[:, :, cpd.nb_class_veh] = nb_ip_data[:, :, ipd.class_veh]
        cp_data[:, :, cpd.nb_class_ped] = nb_ip_data[:, :, ipd.class_ped]
        cp_data[:, :, cpd.nb_class_cyc] = nb_ip_data[:, :, ipd.class_cyc]
        cp_data[:, :, cpd.nb_class_tru] = nb_ip_data[:, :, ipd.class_tru]

        cp_data[mask, :] = 0

        ip_data_n = deepcopy(ip_data)
        nb_ip_data_n = deepcopy(nb_ip_data)
        cp_data_n = deepcopy(cp_data)
        
        for ip_name in ipd.keys():
            if ip_name in self.cfg.SELF_TRAIN.GACE.NORM_FACTORS:
                norm_factor = self.cfg.SELF_TRAIN.GACE.NORM_FACTORS[ip_name]
                
                if isinstance(norm_factor, list):
                    norm_factor_veh = norm_factor[0]
                    norm_factor_ped = norm_factor[1]
                    norm_factor_cyc = norm_factor[2]
                    norm_factor_tru = norm_factor[3]

                    veh_mask = ip_data[:, ipd.class_veh] == 1
                    ped_mask = ip_data[:, ipd.class_ped] == 1
                    cyc_mask = ip_data[:, ipd.class_cyc] == 1
                    tru_mask = ip_data[:, ipd.class_tru] == 1
                    ip_data_n[veh_mask, ipd[ip_name]] /= norm_factor_veh 
                    ip_data_n[ped_mask, ipd[ip_name]] /= norm_factor_ped
                    ip_data_n[cyc_mask, ipd[ip_name]] /= norm_factor_cyc
                    ip_data_n[tru_mask, ipd[ip_name]] /= norm_factor_tru

                    veh_mask_nb = nb_ip_data[:, :, ipd.class_veh] == 1
                    ped_mask_nb = nb_ip_data[:, :, ipd.class_ped] == 1
                    cyc_mask_nb = nb_ip_data[:, :, ipd.class_cyc] == 1
                    tru_mask_nb = nb_ip_data[:, :, ipd.class_tru] == 1
                    nb_ip_data_n[veh_mask_nb, ipd[ip_name]] /= norm_factor_veh
                    nb_ip_data_n[ped_mask_nb, ipd[ip_name]] /= norm_factor_ped
                    nb_ip_data_n[cyc_mask_nb, ipd[ip_name]] /= norm_factor_cyc
                    nb_ip_data_n[tru_mask_nb, ipd[ip_name]] /= norm_factor_tru

                else:
                    ip_data_n[:, ipd[ip_name]] /= norm_factor
                    nb_ip_data_n[:, :, ipd[ip_name]] /= norm_factor

        for cp_name in cpd.keys():
            if cp_name in self.cfg.SELF_TRAIN.GACE.NORM_FACTORS_CP:
                norm_factor = self.cfg.SELF_TRAIN.GACE.NORM_FACTORS_CP[cp_name]
                cp_data_n[:, :, cpd[cp_name]] /= norm_factor

        nb_ip_data_n = np.swapaxes(nb_ip_data_n, 1, 2)
        cp_data_n = np.swapaxes(cp_data_n, 1, 2)

        target_category = target_data[:, self.target_dict.category]
        target_iou = target_data[:, self.target_dict.iou_w_gt]

        ip_data_n = torch.tensor(ip_data_n[..., None, None], dtype=torch.float32).cuda()
        nb_ip_data_n = torch.tensor(nb_ip_data_n[..., None], dtype=torch.float32).cuda()
        cp_data_n = torch.tensor(cp_data_n[..., None], dtype=torch.float32).cuda()
        target_category = torch.tensor(target_category, dtype=torch.float32)
        target_iou = torch.tensor(target_iou, dtype=torch.float32)

        return ip_data_n, cp_data_n, nb_ip_data_n, target_category, target_iou
 
