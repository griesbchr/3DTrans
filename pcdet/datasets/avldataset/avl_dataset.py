import pickle
import orjson
import copy
from dataclasses import dataclass
from pathlib import Path

from scipy.spatial.transform import Rotation
import numpy as np

from ..dataset import DatasetTemplate
from ...utils import box_utils, common_utils 
from ...ops.roiaware_pool3d import roiaware_pool3d_utils


class AVLDataset(DatasetTemplate):

    def __init__(self,
                 dataset_cfg,
                 class_names,
                 training=True,
                 root_path=None,
                 logger=None,
                 creating_infos=False):
        super().__init__(dataset_cfg=dataset_cfg,
                         class_names=class_names,
                         training=training,
                         root_path=root_path,
                         logger=logger)
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        split_dir = self.root_path / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()
                               ] if split_dir.exists() else None
        self.creating_infos = creating_infos

        self.avl_infos = []
        self.include_avl_data(self.split)
        self.avl_infos = [ai for ai in self.avl_infos
                          if 'annos' in ai]  # filter out frames wituoht labels
        
        self.train_fov_only = self.dataset_cfg.get('TRAIN_FOV_ONLY', False)
        self.fov_angle_deg = self.dataset_cfg.get('LIDAR_FOV', 120)
        self.lidar_heading_angle_deg = self.dataset_cfg.get('LIDAR_HEADING', 0)
        self.disregard_truncated = self.dataset_cfg.get('DISREGARD_TRUNCATED', True)

        self.eval_fov_only = self.dataset_cfg.get('EVAL_FOV_ONLY', False)

        self.lidar_z_shift = self.dataset_cfg.get('LIDAR_Z_SHIFT', 0.0)
        
        self.map_class_to_kitti = self.dataset_cfg.get('MAP_CLASS_TO_KITTI',None)

        subsamplefactor = self.dataset_cfg.get('SUBSAMPLEFACTOR', None)
        if subsamplefactor is not None and subsamplefactor > 1:
            self.sample_id_list = self.sample_id_list[::subsamplefactor]

            #filter infors for subsampled samples
            self.avl_infos = [info for info in self.avl_infos if info['point_cloud']['lidar_idx'] in self.sample_id_list]

    def map_merge_classes(self):
        if self.dataset_cfg.get('MAP_MERGE_CLASS', None) is None:
            return

        #update class names in avl_infos
        map_merge_class = self.dataset_cfg.MAP_MERGE_CLASS
        for info in self.avl_infos:
            if 'annos' not in info:
                continue
            info['annos']['name'] = np.vectorize(lambda name: map_merge_class[name], otypes=[str])(info['annos']['name'])
        
        if not(hasattr(self, 'data_augmentor')) or self.data_augmentor is None:
            return
        if self.dataset_cfg.get('DATA_AUGMENTOR', None) is None:
            return
        #get aug name list
        aug_name_list = [aug["NAME"] for aug in self.data_augmentor.augmentor_configs.AUG_CONFIG_LIST]
        data_augmentor = self.dataset_cfg.get('DATA_AUGMENTOR', {})
        disable_aug_list = data_augmentor.get('DISABLE_AUG_LIST', [])
        if "gt_sampling" in aug_name_list and "gt_sampling" not in disable_aug_list:
            aug_idx = aug_name_list.index("gt_sampling")
            db_infos = self.data_augmentor.data_augmentor_queue[aug_idx].db_infos
            for info_key in db_infos:
                infos = db_infos[info_key]
                for info in infos:
                    info["name"] = map_merge_class[info["name"]]


    def include_avl_data(self, split):
        if self.logger is not None:
            self.logger.info('Loading AVL dataset')
        avl_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[split]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                avl_infos.extend(infos)

        self.avl_infos.extend(avl_infos)

        if not self.creating_infos:
            self.map_merge_classes()

        if self.logger is not None:
            self.logger.info('Total samples for AVL dataset: %d' %
                             (len(avl_infos)))

    def set_split(self, split):
        
        super().__init__(dataset_cfg=self.dataset_cfg,
                         class_names=self.class_names,
                         training=self.training,
                         root_path=self.root_path,
                         logger=self.logger)
        self.split = split

        split_dir = self.root_path / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()
                               ] if split_dir.exists() else None
        self.include_avl_data(self.split)

    def fit_plane(self, sample_idx):
        points = self.get_lidar(sample_idx)
        import open3d as o3d
        pcl = o3d.geometry.PointCloud()
        pts = copy.deepcopy(points)[:, :3]
        pts = pts[np.abs(pts[..., 0]) < 20]
        pts = pts[np.abs(pts[..., 1]) < 3]
        pcl.points = o3d.utility.Vector3dVector(pts)
        plane_coefficients, inliers = pcl.segment_plane(distance_threshold=.1, ransac_n=3, num_iterations=1000)

        plane_mask = np.zeros((len(pts), 1)).astype(bool)
        plane_mask[inliers] = True

        # ones = np.ones((len(pts), 1))
        # pcl_wo_h = np.hstack((pts[:, :3], ones))
        # above_plane_mask = (np.einsum('ij,kj->ki', plane_coefficients[None], pcl_wo_h) > 0).flatten()
        # above_plane_mask = plane_mask.flatten()
        
        return plane_coefficients, plane_mask
    
    def estimate_lidar_z_shift(self, n_samples=100):
        from tqdm import tqdm
        z_shift = 0.0

        n_samples = min(n_samples, len(self.sample_id_list))
        sample_id_list = np.random.choice(self.sample_id_list, n_samples, replace=False)

        z_shifts = []
        for sample_idx in tqdm(sample_id_list, desc="Fitting ground plane..."):
            plane_coefficients, plane_mask = self.fit_plane(sample_idx)
            z_shifts.append(common_utils.plane_coefficients_to_z_offset(plane_coefficients))
        
        #find most occuring z_shift with a bin size of 1 cm
        bin_size = 0.01     #1 cm

        # Calculate the histogram and bin edges
        hist, bin_edges = np.histogram(z_shifts, bins=np.arange(min(z_shifts), max(z_shifts) + bin_size, bin_size))

        # Find the bin(s) with the highest frequency (the mode)
        modes = bin_edges[:-1][hist == hist.max()]

        # If there are multiple modes (multiple bins with the same highest frequency), you can get them all
        print("most occuring z-shift in", n_samples, "random samples: ", modes)
        z_shift = modes[0]

        return z_shift
    
    def get_label(self, idx):
        raise NotImplementedError

    def get_lidar(self, idx):
        raise NotImplementedError

    def get_infos(self,
                  num_workers=4,
                  has_label=True,
                  count_inside_pts=True,
                  sample_id_list=None):
        raise NotImplementedError

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.avl_infos) * self.total_epochs

        return len(self.avl_infos)

    '''
    from openpcdet 
    '''
    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7 or 9), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:
        """
        
        def get_template_prediction(num_samples):
            box_dim = 9 if self.dataset_cfg.get('TRAIN_WITH_SPEED', False) else 7
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, box_dim]), 'pred_labels': np.zeros(num_samples)
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            if 'metadata' in batch_dict:
                single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        from ...ops.iou3d_nms import iou3d_nms_utils
        if 'annos' not in self.avl_infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        if (len(det_annos) != len(self.avl_infos)):
            print("Number of frames in det_annos and avl_infos do not match")
            partial = True
        else:
            partial = False
                                                                                        
        def kitti_eval(eval_det_annos, eval_gt_annos, map_class_to_kitti, class_names):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils
            
            if self.map_class_to_kitti is not None:
                class_names = list(set([self.map_class_to_kitti[x] for x in class_names]))
            
            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_class_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_class_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get(
                    'INFO_WITH_FAKELIDAR', False))
            ap_result_str, ap_dict = kitti_eval.get_custom_eval_result(
                gt_annos=eval_gt_annos,
                dt_annos=eval_det_annos,
                current_classes=class_names)
            return ap_result_str, ap_dict

        def waymo_eval(eval_det_annos, eval_gt_annos):
            from ..waymo.waymo_eval import OpenPCDetWaymoDetectionMetricsEstimator
            eval = OpenPCDetWaymoDetectionMetricsEstimator()
            
            for anno in eval_gt_annos:
                anno['difficulty'] = np.zeros([anno['name'].shape[0]], dtype=np.int32)
            #waymo supports     WAYMO_CLASSES = ['unknown', 'Vehicle', 'Pedestrian', 'Truck', 'Cyclist']
            max_eval_dist = 1000
            ap_dict = eval.waymo_evaluation(eval_det_annos,
                                            eval_gt_annos,
                                            class_name= self.class_names,
                                            distance_thresh=max_eval_dist,
                                            fake_gt_infos=self.dataset_cfg.get(
                                                'INFO_WITH_FAKELIDAR', False))
            #filter out dict entries where the key contains SIGN
            ap_dict = {k: v for k, v in ap_dict.items() if 'SIGN' not in k}

            #filter out dict entries where the key contains APH
            ap_dict = {k: v for k, v in ap_dict.items() if 'APH' not in k}

            #filter out dict entries where the key contains APL
            ap_dict = {k: v for k, v in ap_dict.items() if 'APL' not in k}
            
            #reduce key OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1 TO VEHICLE_1
            ap_dict = {k.replace('OBJECT_TYPE_TYPE_', ''): v for k, v in ap_dict.items()} 
            ap_dict = {k.replace('LEVEL_', ''): v for k, v in ap_dict.items()} 

            ap_result_str = '\n'
            for key in ap_dict:
                ap_dict[key] = ap_dict[key][0]
                
                if 'AP' in key:    
                    ap_result_str += '%s: %.4f \n' % (key, ap_dict[key]*100)

            return ap_result_str, ap_dict
            
        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = []
        #drop_info = self.dataset_cfg.get('EVAL_REMOVE_CLASSES', None)
        drop_infos = ["DontCare", "Dont_Care", "Other"]
        for info in self.avl_infos:
            if partial:
                if info['point_cloud']['lidar_idx'] not in [det_anno['frame_id'] for det_anno in eval_det_annos]:
                    continue
            eval_gt_annos.append(copy.deepcopy(info['annos']))
            for drop_info in drop_infos:
                eval_gt_annos[-1] = common_utils.drop_info_with_name(
                    eval_gt_annos[-1], name=drop_info)

        #print number of objects in gt and det for each class in class_names
        for class_name in class_names:
            gt_count = 0
            det_count = 0
            for anno in eval_gt_annos:
                if len(anno['name']) == 0:    
                    continue
                gt_count += sum(anno['name'] == class_name)
            for anno in eval_det_annos:
                if len(anno['name']) == 0:
                    continue
                det_count += sum(anno['name'] == class_name)
            print("Pre Drop: Class:", class_name, "avg. gt_count/frame:", round(gt_count/len(eval_gt_annos),2), "avg. det_count/frame:", round(det_count/len(eval_det_annos),2))
        #remove_le_points = self.dataset_cfg.get('EVAL_REMOVE_LESS_OR_EQ_POINTS', None)
        remove_le_points = 0
        #ignore_classes = self.dataset_cfg.get('EVAL_IGNORE_CLASSES', None)
        ignore_classes = ["Cyclist", "Pedestrian", "Truck"]
        #ignore_classes = []
        #remove_overlapping = self.dataset_cfg.get('EVAL_REMOVE_OVERLAPPING_BEV_IOU', None)
        min_remove_overlap_bev_iou = 0.5
        sum_gt = 0
        sum_det = 0
        # remove gt objects and overlapping det objects  
        for i, gt_anno in enumerate(eval_gt_annos):

            remove_mask = np.zeros(len(gt_anno["name"]), dtype=bool)
            remove_mask_det = np.zeros(len(eval_det_annos[i]["name"]), dtype=bool)

            # remove with less then n points
            remove_mask[gt_anno['num_points_in_gt'] <= remove_le_points] = True

            # add ignore classes to mask
            for ignore_class in ignore_classes:
                remove_mask[ gt_anno['name'] == ignore_class] = True
                remove_mask_det[ eval_det_annos[i]['name'] == ignore_class] = True


            #add gt boxes that are not in fov to remove mask
            if self.eval_fov_only:
                remove_mask[~self.extract_fov_gt_nontruncated(gt_anno["gt_boxes_lidar"], 120, 0)] = True
                remove_mask_det[~self.extract_fov_gt_nontruncated(eval_det_annos[i]["boxes_lidar"], 120, 0)] = True	

            iou_matrix = iou3d_nms_utils.boxes_bev_iou_cpu(
                gt_anno["gt_boxes_lidar"][remove_mask], eval_det_annos[i]["boxes_lidar"])
            remove_mask_det[np.any(iou_matrix > min_remove_overlap_bev_iou, axis=0)] = True
            
            #print("dropping", np.sum(remove_mask), "gt objects and", np.sum(remove_mask_det), "det objects")
            sum_gt += np.sum(remove_mask)
            sum_det += np.sum(remove_mask_det)

            eval_gt_annos[i] = common_utils.drop_info_with_mask(gt_anno, remove_mask)
            eval_det_annos[i] = common_utils.drop_info_with_mask(eval_det_annos[i], remove_mask_det)

        #print number of objects in gt and det for each class in class_names
        print("\n")
        for class_name in class_names:
            gt_count = 0
            det_count = 0
            for anno in eval_gt_annos:
                if len(anno['name']) == 0:    
                    continue
                gt_count += sum(anno['name'] == class_name)
            for anno in eval_det_annos:
                if len(anno['name']) == 0:
                    continue
                det_count += sum(anno['name'] == class_name)
            print("Post Drop: Class:", class_name, "avg. gt_count/frame:", round(gt_count/len(eval_gt_annos),2), "avg. det_count/frame:", round(det_count/len(eval_det_annos),2))

        print("\ndropped", round(sum_gt/len(eval_gt_annos),2), "gt objects/frame and", round(sum_det/len(eval_det_annos),2), "det objects/frame")
        
        if self.eval_fov_only:
            print("\ndoing fov only evaluation\n")
                
                


        # z_shift = self.dataset_cfg.get('TRAINING_Z_SHIFT', None)
        # if z_shift is not None:
        #     for anno in eval_det_annos:
        #         anno['boxes_lidar'][:, 2] += z_shift
    

        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos, self.map_class_to_kitti, class_names)
        elif kwargs['eval_metric'] == 'waymo':
            ap_result_str, ap_dict = waymo_eval(eval_det_annos, eval_gt_annos)
        else:
            raise NotImplementedError
        
        if 'return_annos' in kwargs and kwargs['return_annos']:
            return ap_result_str, ap_dict, eval_gt_annos, eval_det_annos
        
        return ap_result_str, ap_dict
        
        
    def create_groundtruth_database(self,
                                    info_path=None,
                                    used_classes=None,
                                    split='train'):
        import torch
        from tqdm import tqdm

        database_save_path = Path(
            self.root_path) / ('gt_database' if split == 'train' else
                               ('gt_database_%s' % split))
        db_info_save_path = Path(
            self.root_path) / ('avl_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in tqdm(range(len(infos)), desc='gt_database'):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info.get('annos', None)
            if annos is None:
                continue
            names = annos['name']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]),
                torch.from_numpy(gt_boxes)).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx.replace(
                    '/', '_'), names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(
                        self.root_path))  # gt_database/xxxxx.bin
                    db_info = {
                        'name': names[i],
                        'path': db_path,
                        'image_idx': sample_idx,
                        'gt_idx': i,
                        'box3d_lidar': gt_boxes[i],
                        'num_points_in_gt': gt_points.shape[0]
                    }
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    def __getitem__(self, index):
        #index = 4850
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.avl_infos)

        frame_id = self.avl_infos[index]
        
        info = copy.deepcopy(frame_id)

        sample_idx = info['point_cloud']['lidar_idx']
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id': sample_idx,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='Other')
            annos = common_utils.drop_info_with_name(annos, name='Dont_Care')

            gt_names = annos['name']
            gt_boxes_lidar = annos["gt_boxes_lidar"].astype(np.float32)


            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)

            input_dict['points'] = points

        if self.train_fov_only:
            #remove gt boxes that are completely outside of fov
            outside_fov_mask = self.extract_bbox_outside_fov(gt_boxes_lidar, self.fov_angle_deg, self.lidar_heading_angle_deg)
            input_dict['gt_boxes'] =  gt_boxes_lidar[~outside_fov_mask]
            input_dict['gt_names'] = gt_names[~outside_fov_mask]

            if self.disregard_truncated:
                truncated = np.ones(len(input_dict['gt_names']), dtype=bool)
                non_truncated_mask = self.extract_fov_gt_nontruncated(input_dict['gt_boxes'], self.fov_angle_deg, self.lidar_heading_angle_deg)
                truncated[non_truncated_mask] = False

                input_dict['truncated'] = truncated

         # load saved pseudo label for unlabeled data
        if self.dataset_cfg.get('USE_PSEUDO_LABEL', None) and self.training:
            self.fill_pseudo_labels(input_dict)

        data_dict = self.prepare_data(data_dict=input_dict)



        if "truncated" in data_dict:            
            #set truncated gt boxes to label (last entry) to -1
            gt_boxes = data_dict['gt_boxes']
            truncated = data_dict['truncated'].astype(bool)
            gt_boxes[truncated, -1] = -1
            data_dict['gt_boxes'] = gt_boxes

            #debug outputs
            num_truncated = truncated.sum()
            #print("%d/%d gt boxes truncated" % (num_truncated, len(annos['name']))) 
            #print("%d/%d gt+sampled boxes truncated" % (num_truncated, len(truncated)))

            data_dict.pop('truncated')

        #do z shift after data preparation to be able to use pc range and anchor sizes for unified coordinate system 
        #z_shift = self.dataset_cfg.get('TRAINING_Z_SHIFT', None)
        #if z_shift is not None:
        #    if 'annos' in info:
        #        gt_boxes_lidar[:, 2] -= z_shift
        #    if "points" in get_item_list:
        #        points[:, 2] -= np.array(z_shift, dtype=np.float64)

        return data_dict


    def merge_labels(self, labels):
        '''
        The AVL datasets are seperately labels bikes (bicycles and motorbiles) and their riders. To achieve a unified
        label format, we merge the bike and rider labels into the bike label.
        '''

        #get label indices of Vehicle_Ridable_Bicycle and Vehicle_Ridable_Motorcycle
        #for each label, find the corresponding human label index. 
        #assert check if it exists
        #overwrite bike label dimenstions with merges dimenstions
        #delete human label

        #get indices of bike and human labels
        label_categories = np.array([label['label_category_id'] for label in labels])
        bike_indices = np.where(np.logical_or(label_categories == 'Vehicle_Ridable_Bicycle', label_categories == 'Vehicle_Ridable_Motorcycle'))[0]
        
        if len(bike_indices) == 0:
            return labels
        
        delete_indices = []

        for bike_label_idx in bike_indices:
            
            bike_label = labels[bike_label_idx]
            
            human_label_exists = True
            
            #check if "attributes" attribute exists
            #if not, keep bike label and continue
            if 'attributes' not in bike_label.keys():
                print("No 'attributes' attribute found in sequence", bike_label["dataset_name"])
                human_label_exists = False
            
            #check if "with-rider" attribute exists
            #if not, keep bike label and continue
            elif 'with-rider' not in bike_label['attributes'].keys():
                print("No 'with-rider' attribute found in sequence", bike_label["dataset_name"])
                human_label_exists = False


            #check if "Connected_to" attribute exists
            #if not, keep bike label and continue
            elif 'Connected_to' not in bike_label['attributes'].keys():
                print("No 'Connected_to' attribute found in sequence", bike_label["dataset_name"])
                human_label_exists = False

            #check if "with-rider" attribute is true, of not delete bike label and continue
            if human_label_exists:
                if bike_label['attributes']['with-rider'] != "yes":
                    delete_indices.append(bike_label_idx)
                    continue
                else:                 
                    human_id = bike_label['attributes']["Connected_to"]
                    
                    #find human label index
                    label_ids = np.array([label['label_id'] for label in labels])
                    human_index = np.where(label_ids == human_id)[0]
                    
                    #assert that human label exists
                    if (len(human_index) == 0 or len(human_index) > 1):
                        print("No or multiple human label found for bike label in sequence", bike_label["dataset_name"])
                        continue
                    
                    human_index = human_index[0]
                    human_label = labels[human_index]

            else:
                human_labels = [label for label in labels if label['label_category_id'] == 'Human'] 
                try:  
                    human_label, dist = self.get_closest_bbox(bike_label, human_labels)
                except:
                    continue
                human_index = labels.index(human_label)
                #check if human belongs to bike 
                #center of bboxes must be within 0.7 meters of each other
                if dist > 0.7:
                    delete_indices.append(bike_label_idx)
                    continue
            
            #merge bike and human bbox dimenstions
            bbox = self.merge_boxes(bike_label["three_d_bbox"], human_label["three_d_bbox"])
            labels[bike_label_idx]["three_d_bbox"] = bbox

            #remove human label
            delete_indices.append(human_index)

        #delete labels from label list
        labels = [label for i, label in enumerate(labels) if i not in delete_indices]

        return labels
    @staticmethod
    def get_closest_bbox(target_label, labels):
        '''
        Returns the index of the closest label to the target label
        '''
        #get target label center
        target_center = np.array((target_label['three_d_bbox']['cx'], target_label['three_d_bbox']['cy'], target_label['three_d_bbox']['cz']))
        #get label centers
        label_centers = np.array([(label['three_d_bbox']['cx'], label['three_d_bbox']['cy'], label['three_d_bbox']['cz']) for label in labels])
        #calculate distances
        distances = np.linalg.norm(label_centers - target_center, axis=1)
        #get index of closest label
        closest_index = np.argmin(distances)

        return labels[closest_index], distances[closest_index]
    
    @staticmethod
    def merge_boxes(box_bike, box_human):
        # Compute corners of the first and second box
        bbox_bike_np = np.array((box_bike['cx'], box_bike['cy'], box_bike['cz'], 
                                box_bike['l'], box_bike['w'], box_bike['h'], 
                                box_bike['rot_z']))[np.newaxis, :]
        bbox_human_np = np.array((box_human['cx'], box_human['cy'], box_human['cz'],
                                box_human['l'], box_human['w'], box_human['h'],
                                box_human['rot_z']))[np.newaxis, :]
        
        corners_bike = box_utils.boxes_to_corners_3d(bbox_bike_np)[0]
        corners_human = box_utils.boxes_to_corners_3d(bbox_human_np)[0]

        # Find the min and max coordinates for the new box
        #min_x = min(min(corners_bike[:, 0]), min(corners_human[:, 0]))
        #max_x = max(max(corners_bike[:, 0]), max(corners_human[:, 0]))
        #min_y = min(min(corners_bike[:, 1]), min(corners_human[:, 1]))
        #max_y = max(max(corners_bike[:, 1]), max(corners_human[:, 1]))
        min_z = min(min(corners_bike[:, 2]), min(corners_human[:, 2]))
        max_z = max(max(corners_bike[:, 2]), max(corners_human[:, 2]))

        # Calculate new box center and dimensions
        cx = box_bike['cx']  
        cy = box_bike['cy']
        cz = (min_z + max_z) / 2
        l = box_bike['l']  
        w =  box_bike['w'] 
        h = max_z - min_z

        # keep rotation of bike
        rot_z = box_bike['rot_z']

        # if quaternion is given, use it
        if 'quaternion' in box_bike.keys():
            quaternion = box_bike['quaternion']
            new_box = {'cx': cx, 'cy': cy, 'cz': cz, 'h': h, 'l': l, 'w': w, 'rot_z': rot_z, 'quaternion': quaternion}
            return new_box
        
        # Create and return new box
        new_box = {'cx': cx, 'cy': cy, 'cz': cz, 'h': h, 'l': l, 'w': w, 'rot_z': rot_z}

        return new_box