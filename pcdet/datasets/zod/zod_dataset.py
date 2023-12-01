import copy
import pickle
import os

import numpy as np

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from ..dataset import DatasetTemplate

from zod import ZodFrames
from zod.constants import AnnotationProject, TRAIN, VAL

class ZODDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, creating_infos=False):
        """
        Args:
            root_path: data(set) root path
            dataset_cfg:
            class_names: class names used by the detector, 
                         missmatches between dataset and detector are filtered out
            training: sets mode to either 'train' or 'test'
            logger:

        mode is either 'train' or 'test'   (in general)
        data_split is either 'train' or 'val'  (in zod)
        """

        super().__init__(dataset_cfg, class_names, training, root_path, logger)
        
        self.data_root = self.dataset_cfg.DATA_PATH

        self.data_split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.version = self.dataset_cfg.VERSION
        self.creating_infos = creating_infos
        self.set_split(self.data_split, self.version)
        
        self.zod_infos = []
        self.include_zod_infos(self.mode)

        self.num_point_features = len(self.dataset_cfg.POINT_FEATURE_ENCODING["used_feature_list"])
        self.fov_points_only = self.dataset_cfg.FOV_POINTS_ONLY
        
        self.load_version = 'mini' if self.version == 'mini' else 'full'
        self.zod_frames = ZodFrames(dataset_root=self.data_root, version=self.load_version)
 
        # Transformation from zod lidar frame to waymo lidar frame
        self.T_zod_lidar_to_waymo_lidar = np.array([[0, -1, 0],
                                                  [1,  0, 0],
                                                  [0,  0, 1]])
        self.lidar_z_shift = self.dataset_cfg.get('LIDAR_Z_SHIFT', 0.0)

        self.map_class_to_kitti = self.dataset_cfg.get('MAP_CLASS_TO_KITTI',None)

        self.disregard_truncated = self.dataset_cfg.get('DISREGARD_TRUNCATED',True)

        subsamplefactor = self.dataset_cfg.get('SUBSAMPLEFACTOR', None)
        if subsamplefactor is not None and subsamplefactor > 1:
            self.sample_id_list = self.sample_id_list[::subsamplefactor]

    def map_merge_classes(self):
        if self.dataset_cfg.get('MAP_MERGE_CLASS', None) is None:
            return

        #update class names in zod_infos
        map_merge_class = self.dataset_cfg.MAP_MERGE_CLASS
        for info in self.zod_infos:
            assert 'annos' in info
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



        ##filter classes that are not in class_names
        #if self.detector_class_names is not None:
        #    filtered_labels = 0
        #    for info in self.zod_infos:
        #        if 'annos' not in info:
        #            continue
        #        num_labels_pre = info['annos']['name'].__len__()
        #        
        #        filtered_annos = {}
        #        keep_indices = [i for i, name in enumerate(info['annos']['name']) if name in self.detector_class_names]
        #        for key in info["annos"].keys():
        #            filtered_annos[key] = info["annos"][key][keep_indices]
        #        info["annos"] = filtered_annos
#
        #        num_labels_post = info['annos']['name'].__len__()
        #        filtered_labels += num_labels_pre - num_labels_post
        #    
        #    print("Filtered out %d labels that are not in detector_class_names" % filtered_labels)
          
    def include_zod_infos(self, mode):
        
        if self.logger is not None:
            self.logger.info('Loading ZOD dataset.')
        zod_infos = []

        info_path = self.root_path / self.dataset_cfg.INFO_PATH[mode]
        
        if not info_path.exists():
            print("Infos do not exist yet")
            return
        
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
            zod_infos.extend(infos)
        
        self.zod_infos.extend(zod_infos)
        if self.logger is not None:
            self.logger.info('Total samples for ZOD dataset: %d' % (len(zod_infos)))

        if not self.creating_infos:
            self.map_merge_classes()

    def get_fov_points_only(self, points, calib):
        from zod.constants import Camera, Lidar
        from zod.utils.geometry import get_points_in_camera_fov, transform_points
        from zod.data_classes.geometry import Pose

        camera = Camera.FRONT
        lidar = Lidar.VELODYNE

        t_refframe_to_frame = calib.lidars[lidar].extrinsics
        t_refframe_from_frame = calib.cameras[camera].extrinsics

        t_from_frame_refframe = t_refframe_from_frame.inverse
        t_from_frame_to_frame = Pose(t_from_frame_refframe.transform @ t_refframe_to_frame.transform)


        camera_data = transform_points(points, t_from_frame_to_frame.transform)
        positive_depth = camera_data[:, 2] > 0
        camera_data = camera_data[positive_depth]
        if not camera_data.any():
            return positive_depth

        camera_data, mask = get_points_in_camera_fov(calib.cameras[camera].field_of_view, camera_data)

        final_mask = np.zeros_like(positive_depth)
        final_mask[positive_depth] = mask

        return final_mask
    
    def get_object_truncation(self, corners, calib): 
        corners_flat = corners.reshape(-1,3)
        
        #check if all corners are in fov
        #points in camera mask
        mask = self.get_fov_points_only(corners_flat, calib)

        #respahe mask to per box shape
        mask = mask.reshape(-1,8)

        #if all corners are in fov, object is not truncated
        truncated = np.zeros(mask.shape[0], dtype=bool)
        truncated[mask.sum(axis=1) < 8] = 1

        return truncated
    
    def get_lidar(self, idx, num_features=4):

        #need to import zod frames for this to work
        zod_frame = self.zod_frames[idx]
        lidar_core_frame = zod_frame.info.get_key_lidar_frame()
        pc = lidar_core_frame.read()

        if self.fov_points_only:
            mask = self.get_fov_points_only(pc.points, zod_frame.calibration)
            pc.points = pc.points[mask]
            pc.intensity = pc.intensity[mask]
        
        if num_features == 4:
            #scale intensity to [0,1] from [0,255]
            pc.intensity = pc.intensity / 255
            points = np.concatenate((pc.points, pc.intensity.reshape(-1,1)), axis=1)
        elif num_features == 3:
            points = pc.points
        else:
            raise NotImplementedError
        


        points[:,:3] = points[:,:3] @ self.T_zod_lidar_to_waymo_lidar
        points[:,2] -= self.lidar_z_shift


        return points
    
    def get_label(self, sample_idx):
        """
        returns label data loaded from data folder, 
        only loads classes that are in class_names
        """
        zod_frame = self.zod_frames[sample_idx]
        obj_annos = zod_frame.get_annotation(AnnotationProject.OBJECT_DETECTION)
        #filter out objects without 3d anno
        obj_annos = [obj for obj in obj_annos if obj.box3d is not None]     
        #print("filtered out %d objects without 3d anno" % (len(zod_frame.get_annotation(AnnotationProject.OBJECT_DETECTION)) - len(obj_annos)))
        if self.class_names is not None:
            #filter out objects that are not in class_names
            obj_annos = [obj for obj in obj_annos if obj.subclass in self.class_names]
            
        annotations = {}
        annotations['name'] = np.array([obj.subclass for obj in obj_annos])
        annotations['dimensions'] = np.array([obj.box3d.size for obj in obj_annos])
        annotations['location'] = np.array([obj.box3d.center for obj in obj_annos])
        annotations['yaw'] = np.array([obj.box3d.orientation.yaw_pitch_roll[0] for obj in obj_annos])
        annotations['corners'] = np.array([obj.box3d.corners for obj in obj_annos])

        if len(obj_annos) == 0:
            annotations['gt_boxes_lidar'] = np.zeros((0,7))
            annotations['truncated'] = np.zeros((0))
            annotations['corners'] = np.zeros((0,8,3))
            return annotations

        #rotate and shift coordinate system to match waymo (90 deg around z axis and shift to ground plane)
        annotations['location'] = annotations['location'] @ self.T_zod_lidar_to_waymo_lidar
        annotations['location'][:,2] -= self.lidar_z_shift

        annotations['yaw'] = annotations['yaw'] + np.pi/2

        loc = annotations['location']
        dims = annotations['dimensions']
        rots = annotations['yaw']
        l, w, h = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
        gt_boxes_lidar = np.concatenate(
            [loc, l, w, h, rots[..., np.newaxis]], axis=1)

        annotations['gt_boxes_lidar'] = gt_boxes_lidar       

        #calculate truncation
        annotations['truncated'] = self.get_object_truncation(annotations['corners'], zod_frame.calibration)
        
        return annotations

    #re-initialize dataset with different split
    def set_split(self, split, version):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )

        self.split = split
        self.version = version

        split_dir = self.root_path  / (self.split + '_' + version+ '.txt')
        self.sample_id_list = [x.strip() for x in open
                               (split_dir).readlines()] if split_dir.exists() else None
    
    
    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.sample_id_list) * self.total_epochs

        return len(self.zod_infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.zod_infos)

        info = copy.deepcopy(self.zod_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']

        #get points
        points = self.get_lidar(sample_idx, self.num_point_features)

        input_dict = {
            'frame_id': self.sample_id_list[index],
            'points': points
        }

        #get annos
        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': annos['gt_boxes_lidar'],
            })

        if self.disregard_truncated:
            if "truncated" in annos:
                input_dict['truncated'] = annos['truncated']
            else:
                print("\nWARNING: truncated not in annos for sample id", sample_idx)


        # load saved pseudo label for unlabeled data
        if self.dataset_cfg.get('USE_PSEUDO_LABEL', None) and self.training:
            self.fill_pseudo_labels(input_dict)
            
        data_dict = self.prepare_data(data_dict=input_dict)

        #do z shift after data preparation to be able to use pc range and anchor sizes for unified coordinate system
        # z_shift = self.dataset_cfg.get('TRAINING_Z_SHIFT', None)
        # if z_shift is not None:
        #         data_dict["gt_boxes"][:,2] -= z_shift
        #         data_dict['points'][:,2] -= np.array(z_shift, dtype=np.float64)

        if "truncated" in data_dict:            
            #set truncated gt boxes to label (last entry) to -1
            gt_boxes = data_dict['gt_boxes']
            truncated = data_dict['truncated'].astype(bool)
            gt_boxes[truncated, -1] = -1
            data_dict['gt_boxes'] = gt_boxes

            #debug outputs
            #num_truncated = truncated.sum()
            #print("%d/%d gt boxes truncated" % (num_truncated, len(annos['name']))) 
            #print("%d/%d gt+sampled boxes truncated" % (num_truncated, len(truncated)))

            data_dict.pop('truncated')

        return data_dict

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
        #class_names are dataset.class_names
        if 'annos' not in self.zod_infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}
        
        if (len(det_annos) != len(self.zod_infos)):
            print("Number of frames in det_annos and zod_infos do not match")
            partial = True
        else:
            partial = False

        def kitti_eval(eval_det_annos, eval_gt_annos, map_name_to_kitti):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            #if  annos has "truncated" annotations then add, else add empty np array
            truncation_annos = []
            for anno in eval_gt_annos:
                if "truncated" not in anno:
                    truncation_annos.append(np.array([]))
                else: 
                    truncation_annos.append(copy.deepcopy(anno["truncated"].astype(np.float64)))
            
            kitti_utils.transform_annotations_to_kitti_format(
                eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )

            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_custom_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, 
                current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict
      
        def waymo_eval(eval_det_annos, eval_gt_annos):
            from ..waymo.waymo_eval import OpenPCDetWaymoDetectionMetricsEstimator
            eval = OpenPCDetWaymoDetectionMetricsEstimator()
            
            for anno in eval_gt_annos:
                anno['difficulty'] = np.zeros([anno['name'].shape[0]], dtype=np.int32)
            #waymo supports     WAYMO_CLASSES = ['unknown', 'Vehicle', 'Pedestrian', 'Truck', 'Cyclist']
            ap_dict = eval.waymo_evaluation(eval_det_annos,
                                            eval_gt_annos,
                                            class_name= self.class_names,
                                            distance_thresh=1000,
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
        drop_infos = []
        for info in self.zod_infos:
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
            print("Class:", class_name, "gt_count:", gt_count, "det_count:", det_count)


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
            
            # remove with less then n points
            remove_mask[gt_anno['num_points_in_gt'] <= remove_le_points] = True

            # add ignore classes to mask
            name = gt_anno['name']
            for ignore_class in ignore_classes:
                remove_mask[name == ignore_class] = True

            remove_mask_det = np.zeros(len(eval_det_annos[i]["name"]), dtype=bool)
            iou_matrix = iou3d_nms_utils.boxes_bev_iou_cpu(
                gt_anno["gt_boxes_lidar"][remove_mask], eval_det_annos[i]["boxes_lidar"])
            remove_mask_det[np.any(iou_matrix > min_remove_overlap_bev_iou, axis=0)] = True

            #ignore truncated gt boxes
            if self.disregard_truncated:
                remove_mask[gt_anno["truncated"] == 1] = True
                #XXX next line can also be left out, now quit sure if it makes a difference
                remove_mask_det[~self.extract_fov_gt_nontruncated(eval_det_annos[i]["boxes_lidar"], 120, 0)] = True	


            sum_gt += np.sum(remove_mask)
            sum_det += np.sum(remove_mask_det)

            eval_gt_annos[i] = common_utils.drop_info_with_mask(gt_anno, remove_mask)
            eval_det_annos[i] = common_utils.drop_info_with_mask(eval_det_annos[i], remove_mask_det)              

        print("dropped", sum_gt/len(eval_gt_annos), "gt objects/frame and", sum_det/len(eval_det_annos), "det objects/frame")

        # z_shift = self.dataset_cfg.get('TRAINING_Z_SHIFT', None)
        # if z_shift is not None:
        #     for anno in eval_det_annos:
        #         anno['boxes_lidar'][:, 2] += z_shift

        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos, 
                                                self.map_class_to_kitti)
        if kwargs['eval_metric'] == 'waymo':
            ap_result_str, ap_dict = waymo_eval(eval_det_annos, eval_gt_annos)
        else:
            raise NotImplementedError
        
        if "return_annos" in kwargs and kwargs["return_annos"]:
            return ap_result_str, ap_dict, eval_gt_annos, eval_det_annos

        return ap_result_str, ap_dict

    def get_infos(self, class_names, num_workers=4, has_label=True, sample_id_list=None, num_features=4,  count_inside_pts=True):
        '''
        class_names: list of classes (for zod they are called subclasses) that are contained in the info files
        '''
        import concurrent.futures as futures
        import time

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': num_features, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            if not has_label:
                return info

            annotations = self.get_label(sample_idx)    
            
            #filter out annos where name is not in class_names
            annotations = common_utils.drop_info_with_name(annotations, name=class_names)

            if count_inside_pts:   
                
                # use truncated pc to get points in gt boxes
                points = self.get_lidar(sample_idx, num_features=num_features)

                num_gt = len(annotations['name'])
                num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                for k in range(num_gt):
                    corners_lidar = box_utils.boxes_to_corners_3d(
                        annotations['gt_boxes_lidar'])
                    flag = box_utils.in_hull(points[:, 0:3],
                                                corners_lidar[k])
                    num_points_in_gt[k] = flag.sum()
                annotations['num_points_in_gt'] = num_points_in_gt
            
            info['annos'] = annotations
            
            #check if "truncated" is in annotations
            if "truncated" not in annotations:
                print("\nWARNING: truncated not in annos for sample id", sample_idx)
            
            return info
        
        load_version = "mini" if self.version == 'mini' else 'full'
        self.zod_frames = ZodFrames(dataset_root=self.data_root, version=load_version)
        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list

        # create a thread pool to improve the velocity
        start_time = time.time()
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        end_time = time.time()
        print("Total time for loading infos: ", end_time - start_time, "s")
        print("Loading speed for infos: ", len(sample_id_list) / (end_time - start_time), "sample/s")

        return list(infos)

    def create_groundtruth_database(self, info_path=None, version="full", used_classes=None, split='train', num_point_features=4):
        import torch
        from tqdm import tqdm
        database_save_path = Path(self.root_path) / ('gt_database_%s' % version if split == 'train' else ('gt_database_%s_%s' % (split, version)))
        db_info_save_path = Path(self.root_path) / ('zod_dbinfos_%s_%s.pkl' % (split, version))

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        #make sure we dont use truncated gt boxes for gt sampling
        self.fov_points_only = False

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in tqdm(range(len(infos)), total=len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx, num_point_features)
            annos = info['annos']
            names = annos['name']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'gt_idx': i,
                            'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]

        # Output the num of all classes in database
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    #@staticmethod
    #def create_label_file_with_name_and_box(class_names, gt_names, gt_boxes, save_label_path):
    #    with open(save_label_path, 'w') as f:
    #        for idx in range(gt_boxes.shape[0]):
    #            boxes = gt_boxes[idx]
    #            name = gt_names[idx]
    #            if name not in class_names:
    #                continue
    #            line = "{x} {y} {z} {l} {w} {h} {angle} {name}\n".format(
    #                x=boxes[0], y=boxes[1], z=(boxes[2]), l=boxes[3],
    #                w=boxes[4], h=boxes[5], angle=boxes[6], name=name
    #            )
    #            f.write(line)

def split_zod_data(data_path, versions):
    faulty_train_frames = ['097451', '062073', '008896', '046291', '020452', '026378', '000410', '001554', '057300', '004782', '058043', '002639', '061077', '059396', '062628', '063518', '090283', '069293', '044369', '056545', '030924', '052151', '057144', '052749', '028087', '024391', '027256', '016020', '024304', '056158', '012439', '056269', '003027', '072719', '005912', '053347', '054192', '057435', '070476', '014942', '028927', '052528', '049713', '006494', '009277', '009608']
    faulty_val_frames = ['050949', '081704', '058346', '063667']
    for version in versions:
        
        load_version = "mini" if version == 'mini' else 'full'
        
        zod_frames = ZodFrames(dataset_root=data_path, 
                                            version=load_version)
        
        train_id_list = zod_frames.get_split(TRAIN)
        train_id_list = {s for s in train_id_list if not any(exclude_frames in s for exclude_frames in faulty_train_frames)}
        valid_id_list = zod_frames.get_split(VAL)
        valid_id_list = {s for s in valid_id_list if not any(exclude_frames in s for exclude_frames in faulty_val_frames)}
        
        if version == 'small':
            train_id_list = set(list(train_id_list)[::100])
            valid_id_list = set(list(valid_id_list)[::100])
        with open(str(data_path) + '/train'+"_"+version+'.txt', 'w') as f:
            for item in train_id_list:
                f.write(item + '\n')
        print('saved train split to %s' % str(data_path) + '/train'+"_"+version+'.txt')
        with open(str(data_path) + '/val'+"_"+version+'.txt', 'w') as f:
            for item in valid_id_list:
                f.write(item + '\n')
        print('saved val split to %s' % str(data_path) + '/val'+"_"+version+'.txt')
        

    #add custom split here

def create_zod_infos(dataset_cfg, class_names, data_path, save_path, workers=4):

    splits = ['train', 'val']
    versions = ['full', 'mini', 'small']
    #versions = ['small']

    #split_zod_data(data_path, versions)

    dataset = ZODDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger(), creating_infos=True
    )

    num_features = len(dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)


    for split in splits:
        for version in versions:
            print('------------------------generating infos for version %s ,split %s------------------------' % (version, split))
            dataset.set_split(split, version)
            zod_infos = dataset.get_infos(
                class_names, num_workers=workers, has_label=True, 
                num_features=num_features, count_inside_pts=True
            )
            filename = save_path / ('zod_infos_%s_%s.pkl' % (split ,version))
            with open(filename, 'wb') as f:
                pickle.dump(zod_infos, f)
            print('ZOD info train file is saved to %s' % filename)

            if split == 'train':
                print('------------------------generating gt db for version %s------------------------' % version)
                dataset.create_groundtruth_database(filename, version, split=split, num_point_features=num_features)
    
    print('------------------------Data preparation done------------------------')

#  ~/thesis/3DTrans
# python -m pcdet.datasets.zod.zod_dataset debug tools/cfgs/dataset_configs/zod/OD/zod_dataset.yaml
# python -m pcdet.datasets.zod.zod_dataset create_zod_infos tools/cfgs/dataset_configs/zod/OD/zod_dataset.yaml
if __name__ == '__main__':
    import sys

    if sys.argv.__len__() > 1 and sys.argv[1] == 'debug':
        from pathlib import Path
        import yaml
        from easydict import EasyDict

        data_dir = Path("/") / 'data' / 'zod'
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        dataset = ZODDataset(
            dataset_cfg=dataset_cfg, class_names=['Vehicle', 'Pedestrian', 'Cyclist'],
            root_path=data_dir, training=True, logger=common_utils.create_logger()
        )
        labels = dataset.get_label("000000")

    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_zod_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict

        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        
        #class names included in the dataset info files and gt database
        #in zod subclass names are composed of the class name and the subclass name 
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
        
        create_zod_infos(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            data_path=ROOT_DIR / 'data' / 'zod',
            save_path=ROOT_DIR / 'data' / 'zod',
        )