import copy
import pickle
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import common_utils
from pcdet.datasets.avl import calibration
from pcdet.datasets.dataset import DatasetTemplate

from pcdet.datasets.avl.avl_utils import extract_frame, apply_transform

import open3d as o3d
from open3d.visualization import gui, O3DVisualizer


def fit_plane(points):
    pcl = o3d.geometry.PointCloud()
    pts = copy.deepcopy(points)[:, :3]
    pts = pts[np.abs(pts[..., 0]) < 20]
    pts = pts[np.abs(pts[..., 1]) < 3]
    pcl.points = o3d.utility.Vector3dVector(pts)
    plane_model, inliers = pcl.segment_plane(distance_threshold=.1, ransac_n=3, num_iterations=1000)

    plane_mask = np.zeros((len(pts), 1)).astype(bool)
    plane_mask[inliers] = True

    # ones = np.ones((len(pts), 1))
    # pcl_wo_h = np.hstack((pts[:, :3], ones))
    # above_plane_mask = (np.einsum('ij,kj->ki', plane_model[None], pcl_wo_h) > 0).flatten()
    # above_plane_mask = plane_mask.flatten()

    return plane_model, pts[plane_mask.flatten()].copy()


class AVLDataset(DatasetTemplate):
    def __init__(self, 
                 dataset_cfg, 
                 class_names, 
                 training=True, 
                 root_path=None, 
                 logger=None, 
                 is_generic=True):
        # version is either rooftopbox or truck (two different setups)
        root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)) / dataset_cfg.VERSION
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.infos = []
        self.index_mapping = []

        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]
        self.include_avl_data()

        self.is_fov = dataset_cfg.FOV_POINTS_ONLY
        self.rot = R.from_euler('z', -np.pi / 2) if is_generic else R.from_euler('z', 0)
        self.fov = np.deg2rad(60.0) if self.is_fov else np.deg2rad(360.0)

        # TODO: not needed in the first place --> implement after more sophisticated data handling
        self.generate_record_mappings(self.sample_sequence_list)
        # TODO: handle ground truth boxes
        # self.preprocess_infos()

    def include_avl_data(self):
        self.logger.info('Loading AVL dataset')

        for info_path in self.dataset_cfg.INFO_PATH[self.split]:
            info_path = self.root_path / info_path
            # check if dataset info file exists
            if not info_path.exists():
                self.logger.info('Info path [%s] does not exist!' % str(info_path))
                continue
            # add all infos to list
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                self.infos.extend(infos)

            # remove sequence infos not in list (means folder not found)
            for i in reversed(range(len(self.infos))):
                if self.infos[i]['name'] not in self.sample_sequence_list:
                    del self.infos[i]

        self.logger.info('Total samples for AVL dataset: %d' % (len(self.infos)))

    def generate_record_mappings(self, records):
        self.index_mapping = []
        # TODO: add pseudo label replacement
        for i, info in enumerate(self.infos):
            if info['name'] in records:
                self.index_mapping.append(i)

    # def preprocess_infos(self):
    #     for idx, info in enumerate(tqdm(self.infos, desc='Prepare FOV / Generic')):
    #         info['sample_id'] = info['name'] + '_' + str(int(float(info['timestamp']) * 10))
    #         if 'gt_boxes' in info:
    #             info['gt_boxes'][:, 6] *= -1
    #             if self.is_generic:
    #                 gt_boxes = info['gt_boxes'].copy()
    #                 gt_boxes[:, :3] = self.get_calib(idx).world_to_lidar(gt_boxes[:, :3])
    #                 yaw = R.from_matrix(np.linalg.inv(self.get_calib(idx).vehicle_to_world)[:3, :3]).as_euler('xyz')[-1]
    #                 gt_boxes[:, 6] += yaw
    #                 info['gt_boxes'] = gt_boxes.copy()
    #                 info['gt_names'] = np.array([avl_n2n_kitti(gtn) for gtn in info['gt_names']])
    #             if self.is_fov:
    #                 gt_boxes = info['gt_boxes'].copy()
    #                 angle = np.arctan2(gt_boxes[:, 1], gt_boxes[:, 0])
    #                 mask = (angle > -self.fov / 2.) & (angle < self.fov / 2.)
    #                 info['gt_boxes'] = gt_boxes[mask].copy()
    #                 info['gt_names'] = info['gt_names'][mask]
    #                 info['gt_boxes_token'] = info['gt_boxes_token'][mask]
    #                 info['num_lidar_pts'] = info['num_lidar_pts'][mask]
    #
    #             # hack for batch -> adding class to gt box
    #             if len(info['gt_boxes']) == 0:
    #                 info['gt_boxes'] = np.empty((0, 8))
    #         else:
    #             # TODO: should be fixed in utils
    #             info['gt_boxes'] = np.empty((0, 8))
    #             info['gt_names'] = np.array([])
    #             info['gt_boxes_token'] = np.array([])
    #             info['num_lidar_pts'] = np.array([])

    @staticmethod
    def prepare_info_input(info, class_names, fov_only=False):
        if 'gt_boxes' in info:
            info_prep = {'gt_bbs3d_lidar': info['gt_boxes'][:, :7], 'gt_names': info['gt_names'],
                         'tid': info['gt_boxes_token']}
        else:
            info_prep = {'gt_bbs3d_lidar': [], 'gt_names': [], 'tid': []}
        return info_prep

    def get_pcl(self, idx, ret_list=False, is_fov=False, n_past_frames=0, corrected=False):
        def remove_ego_points(pts, center_radius=2.0):
            mask = np.linalg.norm(pts[:, :2], axis=1) > center_radius
            return pts[mask]

        info = self.get_info(idx)
        points_all = []
        for lidar_id, lidar_data in info['lidar_sensors'].items():
            transform = lidar_data['lidar_to_ref']
            lidar_path = self.root_path / self.split / Path(lidar_data['path'])
            if not lidar_path.exists():
                continue
            if lidar_path.suffix == '.bin':
                points = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 4)
            elif lidar_path.suffix == '.npy':
                points = np.load(lidar_path)
            elif lidar_path.suffix == '.pkl':
                with lidar_path.open(mode='rb') as fp:
                    points = pickle.load(fp)
            else:
                raise NotImplementedError

            points = apply_transform(points, transform)
            points[:, 3] /= 255
            points[:, 3] = np.tanh(points[:, 3])
            points_all.append(points)

        points = np.vstack(points_all)[..., :4]
        points = remove_ego_points(points, 2.5)
        plane_model, plane_points = fit_plane(points)

        if self.is_fov:
            angle = np.arctan2(points[:, 1], points[:, 0])
            mask = (angle > -self.fov / 2.) & (angle < self.fov / 2.)
            points = points[mask]

        if np.shape(points)[-1] == 3:
            points = np.hstack((points, np.zeros((len(points), 1))))

        if corrected:
            # correct the scene
            # angle between plane and z axis
            dot_product = np.dot([0, 0, 1], plane_model[:3])
            scale = np.linalg.norm(plane_model[:3]) * np.linalg.norm([0, 0, 1])
            angle = np.arccos(dot_product / scale)
            # rotate scene to match with x-y plane
            rot_mat = np.eye(4)
            rot_y = R.from_euler('y', -angle, degrees=False).as_matrix()
            rot_mat[:3, :3] = rot_y
            points[:, :3] = apply_transform(copy.deepcopy(points[:, :3]), rot_mat)
            # correct height
            points[:, 2] += plane_model[-1]

        if ret_list:
            points = [points]


        # gui.Application.instance.initialize()
        # w = O3DVisualizer("Open3D", 1024, 768)
        # w.set_background((1.0, 1.0, 1.0, 1.0), None)
        # pcl = o3d.geometry.PointCloud()
        # pclp = o3d.geometry.PointCloud()
        # pclr = o3d.geometry.PointCloud()
        # pcl.points = o3d.utility.Vector3dVector(points[:, :3])
        # pclp.points = o3d.utility.Vector3dVector(plane_points[:, :3])
        # pclr.points = o3d.utility.Vector3dVector(points_rot[:, :3])
        # pcl.colors = o3d.utility.Vector3dVector(np.ones((len(points), 3)) * [1, 0, 0])
        # pclp.colors = o3d.utility.Vector3dVector(np.ones((len(plane_points), 3)) * [0, 0, 1])
        # pclr.colors = o3d.utility.Vector3dVector(np.ones((len(points_rot), 3)) * [0, 1, 0])
        #
        # w.add_geometry("Pointcloud", pcl)
        # w.add_geometry("Pointcloud plane", pclp)
        # w.add_geometry("Pointcloud rot", pclr)
        #
        # gui.Application.instance.add_window(w)
        # gui.Application.instance.run()
        if corrected:
            return points, [rot_mat, plane_model[-1]]
        else:
            return points, None

    def get_road_plane(self, index):
        return None

    def get_image(self, index):
        # TODO: adjust to handle all available cameras
        raise NotImplementedError
        # info = self.infos[index]
        # img_path = self.root_path / self.dataset_cfg.DATA_SPLIT[self.split] / info['cam_front_path']
        # return np.array(io.imread(img_path))

    def get_image_shape(self, index):
        # TODO: adjust to handle all available cameras
        raise NotImplementedError
        # if self.image_shape is None:
        #     self.image_shape = np.shape(self.get_image(index))[:2]
        # return self.image_shape

    def get_calib(self, index):
        # TODO: implement such that calibration handles all available camera lidar projections
        raise NotImplementedError
        # info = self.get_info(index)
        # calib = calibration.Calibration(intrinsic=info['cam_intrinsic'],
        #                                 lidar_to_world=info['lidar_to_world'],
        #                                     lidar_to_cam=lidar_to_cam,
        #                                     shape_image=self.get_image_shape(index))
        # return calib

    def get_velocity(self, index):
        info = self.get_info(index)
        ego_info = info['ego']
        return [ego_info['vx'], ego_info['vy'], ego_info['vz'], ego_info['vyaw']]

    def get_timestamp(self, index, separate=False):
        info = self.get_info(index)
        if separate:
            return [info['timestamp_sec'], info['timestamp_nano']]
        else:
            return info['timestamp_sec'] * 1e9 + info['timestamp_nano']

    def get_info(self, index):
        mapped_index = self.index_mapping[index]
        return self.infos[mapped_index].copy()

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return min(len(self.index_mapping), len(self.infos)) * self.total_epochs

        return min(len(self.index_mapping), len(self.infos))

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self)

        info = self.get_info(index)
        points, point_corrections = self.get_pcl(index, corrected=True)

        input_dict = {
            'points': points,
            'frame_id': info['name'] + '_' + info['file_id'],
            'metadata': {'token': info['name']},
            'corrections': point_corrections
        }

        if 'gt_boxes' in info:
            # TODO: implement correctly
            raise NotImplementedError
            if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                mask = (info['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
            else:
                mask = None

            input_dict.update({
                'gt_names': info['gt_names'] if mask is None else info['gt_names'][mask],
                'gt_boxes': info['gt_boxes'] if mask is None else info['gt_boxes'][mask]
            })

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:
        Returns:
        """

        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 7]), 'pred_labels': np.zeros(num_samples)
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
            single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos

    # def evaluation(self, det_annos, class_names, **kwargs):
    #     import json
    #     from nuscenes.nuscenes import NuScenes
    #     from . import nuscenes_utils
    #     nusc = NuScenes(version=self.dataset_cfg.VERSION, dataroot=str(self.root_path), verbose=True)
    #
    #     for anno in det_annos:
    #         for idx, name in enumerate(anno['name']):
    #             anno['name'][idx] = kitti_n2n_nuscenes(name)
    #
    #     print(det_annos)
    #     nusc_annos = nuscenes_utils.transform_det_annos_to_nusc_annos(det_annos, nusc)
    #     nusc_annos['meta'] = {
    #         'use_camera': False,
    #         'use_lidar': True,
    #         'use_radar': False,
    #         'use_map': False,
    #         'use_external': False,
    #     }
    #
    #     # output_path = Path(kwargs['output_path'])
    #     # output_path.mkdir(exist_ok=True, parents=True)
    #     # res_path = str(output_path / 'results_nusc.json')
    #     # with open(res_path, 'w') as f:
    #     #     json.dump(nusc_annos, f)
    #
    #     # self.logger.info(f'The predictions of NuScenes have been saved to {res_path}')
    #
    #     if self.dataset_cfg.VERSION == 'v1.0-test':
    #         return 'No ground-truth annotations for evaluation', {}
    #
    #     from nuscenes.eval.detection.config import config_factory
    #     from nuscenes.eval.detection.evaluate import NuScenesEval
    #
    #     eval_set_map = {
    #         'v1.0-mini': 'mini_val',
    #         'v1.0-trainval': 'val',
    #         'v1.0-test': 'test'
    #     }
    #     try:
    #         eval_version = 'detection_cvpr_2019'
    #         eval_config = config_factory(eval_version)
    #     except:
    #         eval_version = 'cvpr_2019'
    #         eval_config = config_factory(eval_version)
    #
    #     nusc_eval = NuScenesEval(
    #         nusc,
    #         config=eval_config,
    #         result_path=res_path,
    #         eval_set=eval_set_map[self.dataset_cfg.VERSION],
    #         output_dir=str(output_path),
    #         verbose=True,
    #     )
    #     metrics_summary = nusc_eval.main(plot_examples=0, render_curves=False)
    #
    #     with open(output_path / 'metrics_summary.json', 'r') as f:
    #         metrics = json.load(f)
    #
    #     result_str, result_dict = nuscenes_utils.format_nuscene_results(metrics, self.class_names, version=eval_version)
    #     return result_str, result_dict

    # def create_groundtruth_database(self, used_classes=None, max_sweeps=10):
    #     import torch
    #
    #     fov_string = '_fov' if self.is_fov else ''
    #     generic_string = '_gen' if self.is_generic else ''
    #     database_save_path = self.root_path / f'gt_database{generic_string}{fov_string}_{max_sweeps}_sweeps_withvelo'
    #     db_info_save_path = self.root_path / f'nuscenes_dbinfos{generic_string}{fov_string}_{max_sweeps}_sweeps_withvelo.pkl'
    #
    #     database_save_path.mkdir(parents=True, exist_ok=True)
    #     all_db_infos = {}
    #
    #     for idx in tqdm(range(len(self.infos))):
    #         sample_idx = idx
    #         info = self.infos[idx]
    #         points = self.get_lidar_with_sweeps(idx, max_sweeps=max_sweeps)
    #         gt_boxes = info['gt_boxes']
    #         gt_names = info['gt_names']
    #
    #         box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
    #             torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
    #             torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
    #         ).long().squeeze(dim=0).cpu().numpy()
    #
    #         for i in range(gt_boxes.shape[0]):
    #             filename = '%s_%s_%d.bin' % (sample_idx, gt_names[i], i)
    #             filepath = database_save_path / filename
    #             gt_points = points[box_idxs_of_pts == i]
    #
    #             gt_points[:, :3] -= gt_boxes[i, :3]
    #             with open(filepath, 'w') as f:
    #                 gt_points.tofile(f)
    #
    #             if (used_classes is None) or gt_names[i] in used_classes:
    #                 db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
    #                 db_info = {'name': gt_names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
    #                            'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
    #                 if gt_names[i] in all_db_infos:
    #                     all_db_infos[gt_names[i]].append(db_info)
    #                 else:
    #                     all_db_infos[gt_names[i]] = [db_info]
    #     for k, v in all_db_infos.items():
    #         print('Database %s: %d' % (k, len(v)))
    #
    #     with open(db_info_save_path, 'wb') as f:
    #         pickle.dump(all_db_infos, f)


def create_avl_info(version, splits, data_path, save_path):
    from pcdet.datasets.avl import avl_utils
    data_path = data_path / version
    save_path = save_path / version

    for k_split, split in splits.items():
        split_path = data_path / split
        # TODO: remove second condition
        if not split_path.exists() or split == 'training':
            continue
        sequence_list_path = data_path / 'ImageSets' / '{}.txt'.format(split)
        # load sequence folders -> sequences with leading '_' are ignored
        # get sequence list from ImageSets file
        # sequence_list = [p for p in sorted(list(split_path.glob('*'))) if (p.is_dir() and p.stem[0]!='_')]
        with sequence_list_path.open('r') as fp:
            sequence_list = [split_path / line.strip() for line in fp.readlines()]
        # check if unpacked folder exists
        # unpacked when: deepen AI downloaded and extracted (downloader) or extracted from ros bag (rename export to unpacked)
        # not unpacked when deepen data directly from avl (not downloaded)
        # count number of sequences containing unpacked folder
        n_unpacked = sum([(seq / 'unpacked').exists() for seq in sequence_list])
        # check if raw folder exists --> downloaded from deepen AI but not extracted
        if n_unpacked != len(sequence_list):
            assert n_unpacked == 0, 'Should not be unequal zero, not tested!'
            if n_unpacked != 0:
                # --> remove already unpacked
                print('Not all sequences contain unpacked folder --> inconsistent state')
                print('Deleting all unpacked folders...')
                [(seq / 'unpacked').unlink() for seq in sequence_list]
            else:
                # deepen data (not downloaded) --> delete unpacked folders
                pbar = tqdm(sequence_list)
                pbar.set_description('Unpack')
                for seq in pbar:
                    pbar.set_postfix_str(seq.stem)
                    seq_raw_path = seq / 'raw'
                    file_list = sorted(list(seq_raw_path.glob('*.json')))
                    file_list = [p for p in file_list if (not 'metadata' in p.stem and not 'labels' in p.stem)]

                    unpacked_path = seq / 'unpacked'
                    unpacked_path.mkdir(exist_ok=True, parents=True)
                    info_path = unpacked_path / 'info.pkl'

                    info_dict = {}
                    for fid, p in enumerate(file_list): # FIXME: use correct file id, not index! 
                        info = extract_frame(split_path, p, fid, seq.stem, unpacked_path, remove_source=False)
                        info_dict[info['file_id']] = info

                    with info_path.open(mode='wb') as fp:
                        pickle.dump(info_dict, fp)
        else:
            # already unpacked (downloaded deepen ai data, unpacked rosbag)
            pass

        # gather dataset infos
        save_path_tmp = save_path / ('avl_infos_%s.pkl' % k_split)
        avl_infos = avl_utils.fill_avl_infos(split_path, sequence_list)

        print('%s sample: %d' % (split, len(avl_infos)))
        with open(save_path_tmp, 'wb') as f:
            pickle.dump(avl_infos, f)


if __name__ == '__main__':
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_avl_infos', help='')
    args = parser.parse_args()

    if args.func == 'create_avl_infos':
        dataset_cfg = EasyDict(yaml.load(open(args.cfg_file), Loader=yaml.FullLoader))
        #ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        ROOT_DIR = Path("/")
        create_avl_info(
            version=dataset_cfg.VERSION,
            splits=dataset_cfg.DATA_SPLIT,
            data_path=ROOT_DIR / 'data' / 'AVLRooftop',
            save_path=ROOT_DIR / 'data' / 'AVLRooftop'
        )
        class_names = ['Vehicle', 'Pedestrian', 'Cyclist']
        #TODO: classnames
        avl_dataset = AVLDataset(
            dataset_cfg=dataset_cfg, class_names=class_names,
            root_path=ROOT_DIR / 'data' / 'AVLRooftop',
            logger=common_utils.create_logger(), training=False
        )

        test = avl_dataset[0]
        print('Done!')
