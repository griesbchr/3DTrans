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

        self.map_class_to_kitti = self.dataset_cfg.get('MAP_CLASS_TO_KITTI',None)

    def map_merge_classes(self):
        if self.dataset_cfg.get('MAP_MERGE_CLASS', None) is None:
            return

        #update class names in zod_infos
        map_merge_class = self.dataset_cfg.MAP_MERGE_CLASS
        for info in self.zod_infos:
            if 'annos' not in info:
                continue
            info['annos']['name'] = np.vectorize(lambda name: map_merge_class[name], otypes=[np.str])(info['annos']['name'])
        
        if self.data_augmentor is None:
            return
        if self.dataset_cfg.get('DATA_AUGMENTOR', None) is None:
            return
        #get aug name list
        aug_name_list = [aug["NAME"] for aug in self.data_augmentor.augmentor_configs.AUG_CONFIG_LIST]
        disable_aug_list = self.dataset_cfg.get('DISABLE_AUG_LIST', [])
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

    def get_fov_points_only(self, lidar_data, calib):
        from zod.constants import Camera, Lidar
        from zod.utils.geometry import get_points_in_camera_fov, transform_points
        from zod.data_classes.geometry import Pose

        camera = Camera.FRONT
        lidar = Lidar.VELODYNE

        t_refframe_to_frame = calib.lidars[lidar].extrinsics
        t_refframe_from_frame = calib.cameras[camera].extrinsics

        t_from_frame_refframe = t_refframe_from_frame.inverse
        t_from_frame_to_frame = Pose(t_from_frame_refframe.transform @ t_refframe_to_frame.transform)


        camera_data = transform_points(lidar_data.points, t_from_frame_to_frame.transform)
        positive_depth = camera_data[:, 2] > 0
        camera_data = camera_data[positive_depth]
        if not camera_data.any():
            return camera_data, positive_depth

        camera_data, mask = get_points_in_camera_fov(calib.cameras[camera].field_of_view, camera_data)

        final_mask = np.zeros_like(positive_depth)
        final_mask[positive_depth] = mask

        return final_mask
    
    def get_lidar(self, idx, num_features=4):

        #need to import zod frames for this to work
        zod_frame = self.zod_frames[idx]
        lidar_core_frame = zod_frame.info.get_key_lidar_frame()
        pc = lidar_core_frame.read()

        if self.fov_points_only:
            mask = self.get_fov_points_only(pc, zod_frame.calibration)
            pc.points = pc.points[mask]
            pc.intensity = pc.intensity[mask]
        
        if num_features == 4:
            points = np.concatenate((pc.points, pc.intensity.reshape(-1,1)), axis=1)
        elif num_features == 3:
            points = pc.points
        else:
            raise NotImplementedError
        

        #from zod.visualization.lidar_on_image import project_lidar_to_image
        ##get points in camera fov
        #_, mask = project_lidar_to_image(pc, zod_frame.calibration)
        #points = points[mask]

        points[:,:3] = points[:,:3] @ self.T_zod_lidar_to_waymo_lidar


        return points
    
    def get_label(self, sample_idx, class_names=None):
        zod_frame = self.zod_frames[sample_idx]
        obj_annos = zod_frame.get_annotation(AnnotationProject.OBJECT_DETECTION)
        #filter out objects without 3d anno
        obj_annos = [obj for obj in obj_annos if obj.box3d is not None]     
        #print("filtered out %d objects without 3d anno" % (len(zod_frame.get_annotation(AnnotationProject.OBJECT_DETECTION)) - len(obj_annos)))
        if class_names is not None:
            #filter out objects that are not in class_names
            obj_annos = [obj for obj in obj_annos if obj.subclass in class_names]
            
        annotations = {}
        annotations['name'] = np.array([obj.subclass for obj in obj_annos])
        annotations['dimensions'] = np.array([obj.box3d.size for obj in obj_annos])
        annotations['location'] = np.array([obj.box3d.center for obj in obj_annos])
        annotations['yaw'] = np.array([obj.box3d.orientation.yaw_pitch_roll[0] for obj in obj_annos])

        if len(obj_annos) == 0:
            annotations['gt_boxes_lidar'] = np.zeros((0,7))
            annotations['obj_annos'] = obj_annos
            return annotations

        #rotate coordinate system to match waymo (90 deg around z axis)
        annotations['location'] = annotations['location'] @ self.T_zod_lidar_to_waymo_lidar
        annotations['yaw'] = annotations['yaw'] + np.pi/2

        loc = annotations['location']
        dims = annotations['dimensions']
        rots = annotations['yaw']
        l, w, h = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
        gt_boxes_lidar = np.concatenate(
            [loc, l, w, h, rots[..., np.newaxis]], axis=1)

        annotations['gt_boxes_lidar'] = gt_boxes_lidar       
        annotations['obj_annos'] = obj_annos
        #print("Found %d objects" % annotations['name'].shape[0])
        #print("shape of gt_boxes_lidar: ", gt_boxes_lidar.shape)
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
            gt_names = annos['name']
            gt_boxes_lidar = annos['gt_boxes_lidar']
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

        data_dict = self.prepare_data(data_dict=input_dict)

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
        if 'annos' not in self.zod_infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos, map_name_to_kitti):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

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

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.zod_infos]

        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos, 
                                                self.map_class_to_kitti)
        else:
            raise NotImplementedError

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

            annotations = self.get_label(sample_idx, class_names)            

            if count_inside_pts:   
                obj_annos = annotations.pop("obj_annos")            
                points = self.get_lidar(sample_idx, num_features=num_features)

                corners_lidar = np.array([obj.box3d.corners for obj in obj_annos])
                num_gt = len(annotations['name'])
                num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                for k in range(num_gt):
                    flag = box_utils.in_hull(points[:, 0:3],
                                                corners_lidar[k])
                    num_points_in_gt[k] = flag.sum()
                annotations['num_points_in_gt'] = num_points_in_gt
            
            info['annos'] = annotations


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
        
        database_save_path = Path(self.root_path) / ('gt_database_%s' % version if split == 'train' else ('gt_database_%s_%s' % (split, version)))
        db_info_save_path = Path(self.root_path) / ('zod_dbinfos_%s_%s.pkl' % (split, version))

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
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

    for version in versions:
        
        load_version = "mini" if version == 'mini' else 'full'
        
        zod_frames = ZodFrames(dataset_root=data_path, 
                                            version=load_version)
        
        train_id_list = zod_frames.get_split(TRAIN)
        valid_list = zod_frames.get_split(VAL)
        
        if version == 'small':
            train_id_list = set(list(train_id_list)[::100])
            valid_list = set(list(valid_list)[::100])
        with open(str(data_path) + '/train'+"_"+version+'.txt', 'w') as f:
            for item in train_id_list:
                f.write(item + '\n')
        print('saved train split to %s' % str(data_path) + '/train'+"_"+version+'.txt')
        with open(str(data_path) + '/val'+"_"+version+'.txt', 'w') as f:
            for item in valid_list:
                f.write(item + '\n')
        print('saved val split to %s' % str(data_path) + '/val'+"_"+version+'.txt')
        

    #add custom split here

def create_zod_infos(dataset_cfg, class_names, data_path, save_path, workers=4):

    splits = ['train', 'val']
    #versions = ['full', 'mini', 'small']
    versions = ['small']

    split_zod_data(data_path, versions)

    dataset = ZODDataset(
        dataset_cfg=dataset_cfg, class_names=None, root_path=data_path,
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

    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_zod_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict

        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = Path("/")
        
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