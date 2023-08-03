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


@dataclass
class AvlObject:
    name: str
    x: float
    y: float
    z: float
    l: float
    w: float
    h: float
    yaw: float


class AVLDataset(DatasetTemplate):

    def __init__(self,
                 dataset_cfg,
                 class_names,
                 training=True,
                 root_path=None,
                 logger=None):
        super().__init__(dataset_cfg=dataset_cfg,
                         class_names=class_names,
                         training=training,
                         root_path=root_path,
                         logger=logger)
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split
                                                 != 'test' else 'testing')
        split_dir = self.root_path / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()
                               ] if split_dir.exists() else None
        self.avl_infos = []
        self.include_avl_data(self.mode)

        self.avl_infos = [ai for ai in self.avl_infos
                          if 'annos' in ai]  # filter out frames wituoht labels
        
        if self.dataset_cfg.get('MAP_MERGE_CLASS', None) is not None:
            for infos_idx in range(self.avl_infos.__len__()):
                if 'annos' not in self.avl_infos[infos_idx]:
                    continue
                for name_idx in range(self.avl_infos[infos_idx]['annos']['name'].shape[0]):
                    self.avl_infos[infos_idx]['annos']['name'][
                        name_idx] = self.dataset_cfg.MAP_MERGE_CLASS[
                            self.avl_infos[infos_idx]['annos']['name']
                            [name_idx]]

        self.map_class_to_kitti = self.dataset_cfg.get('MAP_CLASS_TO_KITTI',None)

    def include_avl_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading AVL dataset')
        avl_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                avl_infos.extend(infos)

        self.avl_infos.extend(avl_infos)

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
        self.root_split_path = self.root_path / ('training' if self.split
                                                 != 'test' else 'testing')

        split_dir = self.root_path / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()
                               ] if split_dir.exists() else None

    def get_label(self, idx):
        label_file = (self.root_path / Path(idx)).parent.parent / 'label'
        label_file = [lf for lf in label_file.glob('*.json')]
        assert label_file.__len__(
        ) == 1, 'Multiple label files are not expected'
        label_file = label_file[0]

        with open(str(self.root_path / idx), 'r') as f:
            data = orjson.loads(f.read())
        dev_loc, dev_rot = data['device_position'], data['device_heading']
        dev_loc = np.array([dev_loc['x'], dev_loc['y'], dev_loc['z']])
        dev_q = Rotation.from_quat(
            [dev_rot['x'], dev_rot['y'], dev_rot['z'], dev_rot['w']])

        with open(str(label_file), 'r') as f:
            labels = orjson.loads(f.read())

        avl_labels = []
        for l in labels['labels']:
            if Path(idx).stem in l['file_id']:
                bb = l.get('three_d_bbox', None)
                name = l.get('label_category_id', None)
                q = l.get('three_d_bbox', {}).get('quaternion', None)
                if bb is None or name is None or q is None:
                    continue
                yaw_q = Rotation.from_quat(
                    np.array([q['x'], q['y'], q['z'], q['w']]))
                loc = (np.array([bb['cx'], bb['cy'], bb['cz']]) -
                       dev_loc) @ dev_q.as_matrix()
                yaw = (dev_q.inv() * yaw_q).as_euler('xyz', degrees=False)[-1]
                avl_labels.append(
                    AvlObject(name, *loc, bb['l'], bb['w'], bb['h'], yaw))

        return avl_labels

    def get_lidar(self, idx):
        lidar_file = self.root_path / idx

        np_lidar_file = lidar_file.with_suffix('.npy')
        if np_lidar_file.is_file(
        ):  # all preprocessing done in `convert_json_to_numpy`
            points = np.load(np_lidar_file, allow_pickle=False)
        else:  # fallback to json (very slow)
            assert lidar_file.exists()
            with open(str(lidar_file), 'r') as f:
                data = orjson.loads(f.read())
            points = np.array([[p['x'], p['y'], p['z'], p['i']]
                               for p in data['points']])
            dev_loc, dev_heading = data['device_position'], data[
                'device_heading']
            dev_loc = np.array([dev_loc['x'], dev_loc['y'], dev_loc['z']])
            points[:, 0:3] -= dev_loc[None]
            rot_matrix = Rotation.from_quat([
                dev_heading['x'], dev_heading['y'], dev_heading['z'],
                dev_heading['w']
            ]).as_matrix()
            points[:, 0:3] = points[:, 0:3] @ rot_matrix

        points[:, -1] = np.clip(points[:, -1], a_min=0, a_max=1.)
        return points

    def get_infos(self,
                  num_workers=4,
                  has_label=True,
                  count_inside_pts=True,
                  sample_id_list=None):
        from joblib import Parallel, delayed

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            if has_label:
                obj_list = self.get_label(sample_idx)
                if obj_list.__len__() > 0:
                    annotations = {}
                    annotations['name'] = np.array(
                        [obj.name for obj in obj_list])
                    annotations['dimensions'] = np.array(
                        [[obj.l, obj.w, obj.h] for obj in obj_list])
                    annotations['location'] = np.array([[obj.x, obj.y, obj.z]
                                                        for obj in obj_list])
                    annotations['yaw'] = np.array(
                        [obj.yaw for obj in obj_list])

                    loc = annotations['location']
                    dims = annotations['dimensions']
                    rots = annotations['yaw']
                    l, w, h = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                    gt_boxes_lidar = np.concatenate(
                        [loc, l, w, h, rots[..., np.newaxis]], axis=1)
                    annotations['gt_boxes_lidar'] = gt_boxes_lidar

                    if count_inside_pts:
                        points = self.get_lidar(sample_idx)

                        corners_lidar = box_utils.boxes_to_corners_3d(
                            gt_boxes_lidar)
                        num_gt = len(annotations['name'])
                        num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                        for k in range(num_gt):
                            flag = box_utils.in_hull(points[:, 0:3],
                                                     corners_lidar[k])
                            num_points_in_gt[k] = flag.sum()
                        annotations['num_points_in_gt'] = num_points_in_gt

                    info['annos'] = annotations

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        infos = Parallel(n_jobs=num_workers)(delayed(process_single_scene)(sid)
                                             for sid in sample_id_list)
        return infos

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
    '''
    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.avl_infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def accomodate_eval(annos, name_map=None, shift_coord=None):
            for anno in annos:
                if name_map is not None:
                    for k in range(anno['name'].shape[0]):
                        anno['name'][k] = name_map[anno['name'][k]]

                if shift_coord is not None:
                    if 'boxes_lidar' in anno:
                        anno['boxes_lidar'][:, 0:3] -= shift_coord
                    else:
                        anno['gt_boxes_lidar'][:, 0:3] -= shift_coord

                anno['difficulty'] = np.ones(anno['name'].shape[0])

            return annos

        def kitti_eval(eval_det_annos, eval_gt_annos, map_class_to_kitti):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_class_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_class_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get(
                    'INFO_WITH_FAKELIDAR', False))
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos,
                dt_annos=eval_det_annos,
                current_classes=class_names)
            return ap_result_str, ap_dict

        def waymo_eval(eval_det_annos, eval_gt_annos):
            from ..waymo.waymo_eval import OpenPCDetWaymoDetectionMetricsEstimator
            eval = OpenPCDetWaymoDetectionMetricsEstimator()

            #get name map from config
            #get shift coord from config
            #eval_det_annos = accomodate_eval(eval_det_annos, name_map, shift_coord)
            #eval_gt_annos = accomodate_eval(eval_gt_annos, name_map, shift_coord)
            #TODO map avl labels to waymo labels and add shift coord
            raise NotImplementedError
            ap_dict = eval.waymo_evaluation(eval_det_annos,
                                            eval_gt_annos,
                                            class_name=class_names,
                                            distance_thresh=1000,
                                            fake_gt_infos=self.dataset_cfg.get(
                                                'INFO_WITH_FAKELIDAR', False))
            ap_result_str = '\n'
            for key in ap_dict:
                ap_dict[key] = ap_dict[key][0]
                ap_result_str += '%s: %.4f \n' % (key, ap_dict[key])

            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = []
        for info in self.avl_infos:
            eval_gt_annos.append(copy.deepcopy(info['annos']))
            eval_gt_annos[-1] = common_utils.drop_info_with_name(
                eval_gt_annos[-1], name='Other')
            eval_gt_annos[-1] = common_utils.drop_info_with_name(
                eval_gt_annos[-1], name='Dont_Care')

        #eval_det_annos = accomodate_eval(eval_det_annos, self.map_class_to_kitti,self.dataset_cfg.get('SHIFT_COOR', None))
        #eval_gt_annos = accomodate_eval(eval_gt_annos, self.map_class_to_kitti)
        if self.map_class_to_kitti is not None:
            class_names = [self.map_class_to_kitti[x] for x in class_names]

        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos, self.map_class_to_kitti)
        elif kwargs['eval_metric'] == 'waymo':
            #TODO map classes to waymo
            ap_result_str, ap_dict = waymo_eval(eval_det_annos, eval_gt_annos, self.map_class_to_kitti)
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict
    '''
    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.avl_infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def accomodate_eval(annos, name_map=None, shift_coord=None):
            for anno in annos:
                if name_map is not None:
                    for k in range(anno['name'].shape[0]):
                        anno['name'][k] = name_map[anno['name'][k]]

                if shift_coord is not None:
                    if 'boxes_lidar' in anno:
                        anno['boxes_lidar'][:, 0:3] -= shift_coord
                    else:
                        anno['gt_boxes_lidar'][:, 0:3] -= shift_coord

                anno['difficulty'] = np.ones(anno['name'].shape[0])

            return annos

        def kitti_eval(eval_det_annos, eval_gt_annos, map_class_to_kitti):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

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

            ap_dict = eval.waymo_evaluation(eval_det_annos,
                                            eval_gt_annos,
                                            class_name=class_names,
                                            distance_thresh=1000,
                                            fake_gt_infos=self.dataset_cfg.get(
                                                'INFO_WITH_FAKELIDAR', False))
            ap_result_str = '\n'
            for key in ap_dict:
                ap_dict[key] = ap_dict[key][0]
                ap_result_str += '%s: %.4f \n' % (key, ap_dict[key])

            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = []
        for info in self.avl_infos:
            eval_gt_annos.append(copy.deepcopy(info['annos']))
            eval_gt_annos[-1] = common_utils.drop_info_with_name(
                eval_gt_annos[-1], name='Other')
            eval_gt_annos[-1] = common_utils.drop_info_with_name(
                eval_gt_annos[-1], name='Dont_Care')

        #eval_det_annos = accomodate_eval(
        #    eval_det_annos, self.map_class_to_kitti,
        #    self.dataset_cfg.get('SHIFT_COOR', None))
        #eval_gt_annos = accomodate_eval(eval_gt_annos, self.map_class_to_kitti)
        if self.map_class_to_kitti is not None:
            class_names = [self.map_class_to_kitti[x] for x in class_names]

        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos, self.map_class_to_kitti)
        elif kwargs['eval_metric'] == 'waymo':
            ap_result_str, ap_dict = waymo_eval(eval_det_annos, eval_gt_annos)
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict

    def create_groundtruth_database(self,
                                    info_path=None,
                                    used_classes=None,
                                    split='train'):
        import torch

        database_save_path = Path(
            self.root_path) / ('gt_database' if split == 'train' else
                               ('gt_database_%s' % split))
        db_info_save_path = Path(
            self.root_path) / ('avl_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
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

        info = copy.deepcopy(self.avl_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id': sample_idx,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='Other')
            annos = common_utils.drop_info_with_name(annos, name='Dont_Care')
            loc, dims, rots = annos['location'], annos['dimensions'], annos[
                'yaw']
            gt_names = annos['name']
            gt_boxes_lidar = np.concatenate(
                [loc, dims, rots[..., np.newaxis]
                 ],  # watch out on order of dims!
                axis=1).astype(np.float32)
            if self.dataset_cfg.get('SHIFT_COOR', None):
                gt_boxes_lidar[:, 0:3] += self.dataset_cfg.SHIFT_COOR

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)

            if self.dataset_cfg.get('SHIFT_COOR', None):
                points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR,
                                           dtype=np.float32)

            input_dict['points'] = points

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict


def split_avl_data(data_path):
    import random
    dirs = [str(x) for x in data_path.iterdir() if x.is_dir()]
    # CityStreet
    # CityThoroughfare
    # MinorRoad
    # Motorway
    # PrimaryHighway
    # SecondaryHighway
    t1 = [x for x in dirs if 'CityStreet' in x]
    t2 = [x for x in dirs if 'CityThoroughfare' in x]
    t3 = [x for x in dirs if 'MinorRoad' in x]
    t4 = [x for x in dirs if 'Motorway' in x]
    t5 = [x for x in dirs if 'PrimaryHighway' in x]
    t6 = [x for x in dirs if 'SecondaryHighway' in x]

    random.shuffle(t1)
    random.shuffle(t2)
    random.shuffle(t3)
    random.shuffle(t4)
    random.shuffle(t5)
    random.shuffle(t6)

    t1train, t1val = t1[:len(t1) // 2], t1[len(t1) // 2:]
    t2train, t2val = t2[:len(t2) // 2], t2[len(t2) // 2:]
    t3train, t3val = t3[:len(t3) // 2], t3[len(t3) // 2:]
    t4train, t4val = t4[:len(t4) // 2], t4[len(t4) // 2:]
    t5train, t5val = t5[:len(t5) // 2], t5[len(t5) // 2:]
    t6train, t6val = t6[:len(t6) // 2], t6[len(t6) // 2:]

    with open(str(data_path / 'train.txt'), 'w') as f:
        train = [t1train, t2train, t3train, t4train, t5train, t6train]
        for tt in train:
            for line in tt:
                all_frames = [
                    p for p in (Path(line) / 'dataset').glob('*.json')
                ]
                for frame in all_frames:
                    f.write(frame.parent.parent.parent.name + '/' +
                            frame.parent.parent.name + '/' +
                            frame.parent.name + '/' + frame.name)
                    f.write('\n')

    with open(str(data_path / 'val.txt'), 'w') as f:
        val = [t1val, t2val, t3val, t4val, t5val, t6val]
        for tt in val:
            for line in tt:
                all_frames = [
                    p for p in (Path(line) / 'dataset').glob('*.json')
                ]
                for frame in all_frames:
                    f.write(frame.parent.parent.parent.name + '/' +
                            frame.parent.parent.name + '/' +
                            frame.parent.name + '/' + frame.name)
                    f.write('\n')


def create_avl_infos(dataset_cfg,
                     class_names,
                     data_path,
                     save_path,
                     workers=4):
    dataset = AVLDataset(dataset_cfg=dataset_cfg,
                         class_names=class_names,
                         root_path=data_path,
                         training=False)

    train_split, val_split = 'train', 'val'
    train_filename = save_path / f'avl_infos_{train_split}.pkl'
    val_filename = save_path / f'avl_infos_{val_split}.pkl'

    print('---------------Start to generate data infos---------------')
    dataset.set_split(train_split)
    avl_infos_train = dataset.get_infos(num_workers=workers,
                                        has_label=True,
                                        count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(avl_infos_train, f)
    print(f'AVL info train file is saved to {train_filename}')

    dataset.set_split(val_split)
    avl_infos_val = dataset.get_infos(num_workers=workers,
                                      has_label=True,
                                      count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(avl_infos_val, f)
    print(f'AVL info val file is saved to {val_filename}')

    print(
        '---------------Start create groundtruth database for data augmentation---------------'
    )
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


def convert_json_to_numpy(data_path, num_workers=4):
    from tqdm import tqdm
    import concurrent.futures as futures

    def process_single_sequence(seq):
        frames = [f for f in (seq / 'dataset').glob('*.json')]
        for fr in tqdm(frames):
            with open(str(fr), 'r') as f:
                data = orjson.loads(f.read())
            points = np.array([[p['x'], p['y'], p['z'], p['i']]
                               for p in data['points']])
            dev_loc, dev_heading = data['device_position'], data[
                'device_heading']
            dev_loc = np.array([dev_loc['x'], dev_loc['y'], dev_loc['z']])
            points[:, 0:3] -= dev_loc[None]
            rot_matrix = Rotation.from_quat([
                dev_heading['x'], dev_heading['y'], dev_heading['z'],
                dev_heading['w']
            ]).as_matrix()
            points[:, 0:3] = points[:, 0:3] @ rot_matrix
            np.save(str(fr).replace('.json', '.npy'),
                    points,
                    allow_pickle=False)

    sequences = [p for p in (data_path / 'sequences').iterdir() if p.is_dir()]
    with futures.ThreadPoolExecutor(num_workers) as executor:
        executor.map(process_single_sequence, sequences)


if __name__ == '__main__':
    import sys
    from pathlib import Path
    #ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
    ROOT_DIR = Path("/")
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_avl_infos':
        import yaml
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        create_avl_infos(
            dataset_cfg=dataset_cfg,
            class_names=[
                'Vehicle_Drivable_Car', 'Vehicle_Drivable_Van', 'Human',
                'Vehicle_Ridable_Motorcycle', 'Vehicle_Ridable_Bicycle',
                'LargeVehicle_Bus', 'LargeVehicle_TruckCab',
                'LargeVehicle_Truck', 'Trailer'
            ],
            data_path=ROOT_DIR / 'data' / 'AVLTruck',
            save_path=ROOT_DIR / 'data' / 'AVLTruck',
            workers=int(sys.argv[3]) if sys.argv.__len__() > 3 else 4)
    elif sys.argv.__len__() > 1 and sys.argv[1] == 'json2np':
        convert_json_to_numpy(
            ROOT_DIR / 'data' / 'AVLTruck',
            num_workers=int(sys.argv[2]) if sys.argv.__len__() > 2 else 4)
    elif sys.argv.__len__() > 1 and sys.argv[1] == 'debug_avl_eval':
        eval_gt_annos = pickle.load(open("eval_gt_annos.pkl", 'rb'))
        eval_det_annos = pickle.load(open("eval_det_annos.pkl", 'rb'))
        class_names = ['Vehicle_Drivable_Car', 'Human', 'Vehicle_Ridable_Bicycle']
        map_classes_to_kitti =   {
            'Vehicle_Drivable_Car': 'Car',
            'Vehicle_Drivable_Van': 'Car',
            'Vehicle_Ridable_Motorcycle': 'Cyclist',
            'Vehicle_Ridable_Bicycle': 'Cyclist',
            'Human': 'Pedestrian',
            'LargeVehicle_Bus': 'DontCare',
            'LargeVehicle_TruckCab': 'DontCare',
            'LargeVehicle_Truck': 'DontCare',
            'Trailer': 'DontCare'
        }
        def kitti_eval(eval_det_annos, eval_gt_annos):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_classes_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(eval_gt_annos, map_classes_to_kitti)
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos,
                dt_annos=eval_det_annos,
                current_classes=class_names)
            return ap_result_str, ap_dict
        if map_classes_to_kitti is not None:
            class_names = [map_classes_to_kitti[x] for x in class_names]
        ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos)
        print(ap_result_str)