from functools import partial

import numpy as np

from ...utils import common_utils, downsample_utils
from . import augmentor_utils, database_sampler
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
import torch

class DataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None, oss_flag=False):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger
        self.oss_flag = oss_flag
        self.augmentor_configs = augmentor_configs

        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST
        
        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def gt_sampling(self, config=None):
        db_sampler = database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger,
            oss_flag=self.oss_flag
        )
        return db_sampler
    
    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d
    
    def __setstate__(self, d):
        self.__dict__.update(d)
    
    def random_object_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_object_rotation, config=config)

        gt_boxes, points = augmentor_utils.rotate_objects(
            data_dict['gt_boxes'],
            data_dict['points'],
            data_dict['gt_boxes_mask'],
            rotation_perturb=config['ROT_UNIFORM_NOISE'],
            prob=config['ROT_PROB'],
            num_try=50
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_object_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_object_scaling, config=config)
        points, gt_boxes = augmentor_utils.scale_pre_object(
            data_dict['gt_boxes'], data_dict['points'],
            # gt_boxes_mask=data_dict['gt_boxes_mask'],
            scale_perturb=config['SCALE_UNIFORM_NOISE']
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_sampling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_sampling, config=config)
        gt_boxes, points, gt_boxes_mask = augmentor_utils.global_sampling(
            data_dict['gt_boxes'], data_dict['points'],
            gt_boxes_mask=data_dict['gt_boxes_mask'],
            sample_ratio_range=config['WORLD_SAMPLE_RATIO'],
            prob=config['PROB']
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_boxes_mask'] = gt_boxes_mask
        data_dict['points'] = points
        return data_dict

    def label_point_cloud_beam(self, polar_image, points, beam=32):
        if polar_image.shape[0] <= beam:
            print("too small point cloud!")
            return np.arange(polar_image.shape[0])
        beam_label, centroids = downsample_utils.beam_label(polar_image[:,1], beam)
        idx = np.argsort(centroids)
        rev_idx = np.zeros_like(idx)
        for i, t in enumerate(idx):
            rev_idx[t] = i
        beam_label = rev_idx[beam_label]
        return beam_label

    def get_polar_image(self, points):
        theta, phi = downsample_utils.compute_angles(points[:,:3])
        r = np.sqrt(np.sum(points[:,:3]**2, axis=1))
        polar_image = points.copy()
        polar_image[:,0] = phi 
        polar_image[:,1] = theta
        polar_image[:,2] = r 
        return polar_image
    
    #def beam_mask(self, data_dict=None, config=None):
    #    if data_dict is None:
    #        return partial(self.beam_mask, config=config)
    #    assert 'beam_labels' in data_dict
    #    beam_label = data_dict['beam_labels']
    #    beam_mask = np.ones(64, dtype=np.int)
    #    if config['BEAM'] == 48:
    #        beam_mask[::4] = 0
    #    elif config['BEAM'] == 32:
    #        beam_mask[::2] = 0
    #    elif config['BEAM'] == 16:
    #        beam_mask[::4] = 0
    #        beam_mask[::2] = 0
    #    else:
    #        raise NotImplementedError
    #    points_mask = beam_mask[beam_label]
    #    data_dict['points'] = points[points_mask]
    #    data_dict['beam_labels'] = beam_label[points_mask]
    #    return data_dict
        
    def random_points_downsample(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_points_downsample, config=config)
        points = data_dict['points']
        points_mask = np.random.rand(points.shape[0]) < config['POINTS_PROB']
        data_dict['points'] = points[points_mask]
        return data_dict

    def random_beam_upsample(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_beam_upsample, config=config)
        points = data_dict['points']
        polar_image = self.get_polar_image(points)
        beam_label = self.label_point_cloud_beam(polar_image, points, config['BEAM'])
        new_pcs = [points]
        phi = polar_image[:,0]
        for i in range(config['BEAM'] - 1):
            if np.random.rand() < config['BEAM_PROB'][i]:
                cur_beam_mask = (beam_label == i)
                next_beam_mask = (beam_label == i + 1)
                delta_phi = np.abs(phi[cur_beam_mask, np.newaxis] - phi[np.newaxis, next_beam_mask])
                corr_idx = np.argmin(delta_phi,1)
                min_delta = np.min(delta_phi,1)
                mask = min_delta < config['PHI_THRESHOLD']
                cur_beam = polar_image[cur_beam_mask][mask]
                next_beam = polar_image[next_beam_mask][corr_idx[mask]]
                new_beam = (cur_beam + next_beam)/2
                new_pc = new_beam.copy()
                new_pc[:,0] = np.cos(new_beam[:,1]) * np.cos(new_beam[:,0]) * new_beam[:,2]
                new_pc[:,1] = np.cos(new_beam[:,1]) * np.sin(new_beam[:,0]) * new_beam[:,2]
                new_pc[:,2] = np.sin(new_beam[:,1]) * new_beam[:,2]
                new_pcs.append(new_pc)
        data_dict['points'] = np.concatenate(new_pcs,0)
        return data_dict

    def random_beam_downsample(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_beam_downsample, config=config)
        
        assert "num_aug_beams" in data_dict, "num_aug_beams not in data_dict. Make sure to enable INCLUDE_DIODE_IDS in cfg"

        points_with_beam_labels = data_dict['points']
        #beam labels are last column of points
        beam_label = points_with_beam_labels[:,-1].astype(np.int)
        points = points_with_beam_labels[:,:-1] 

        #points = data_dict['points']
        #polar_image = self.get_polar_image(points)
        #beam_label = self.label_point_cloud_beam(polar_image, points,data_dict['num_aug_beams'])
        
        if isinstance(config['BEAM_PROB'], list):
            #assert that list len is 2
            assert len(config['BEAM_PROB']) == 2, "BEAM_PROB must be a list containing the upper and lower bounds of the probability of keeping a beam"
            #randomly sample a probability between the upper and lower bounds
            beam_prob = np.random.uniform(config['BEAM_PROB'][0], config['BEAM_PROB'][1])
        else:
            beam_prob = config['BEAM_PROB']
        beam_mask = np.random.rand(data_dict['num_aug_beams']) < beam_prob
        beam_mask = np.append(beam_mask, True) #always keep points with beam_label == -1
        points_mask = beam_mask[beam_label]
        data_dict['points'] = points[points_mask]
        if config.get('FILTER_GT_BOXES', None):
            num_points_in_gt = roiaware_pool3d_utils.points_in_boxes_cpu(
                        torch.from_numpy(data_dict['points'][:, :3]),
                        torch.from_numpy(data_dict['gt_boxes'][:, :7])).numpy().sum(axis=1)

            mask = (num_points_in_gt >= config.get('MIN_POINTS_OF_GT', 1))
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
            data_dict['gt_names'] = data_dict['gt_names'][mask]
            if 'gt_classes' in data_dict:
                data_dict['gt_classes'] = data_dict['gt_classes'][mask]
                data_dict['gt_scores'] = data_dict['gt_scores'][mask]
            if 'gt_boxes_mask' in data_dict:
                data_dict['gt_boxes_mask'] = data_dict['gt_boxes_mask'][mask]

        return data_dict
    
    def normalize_object_size_multiclass(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.normalize_object_size_multiclass, config=config)
        
        assert len(self.class_names) == len(config['SIZE_RES'])
        
        for i in range(len(self.class_names)):

            #mask for class
            mask = data_dict['gt_names'] == self.class_names[i]

            points, gt_boxes = augmentor_utils.normalize_object_size(
                data_dict['gt_boxes'], data_dict['points'], mask, config['SIZE_RES'][i]
            )
            data_dict['gt_boxes'] = gt_boxes
            data_dict['points'] = points
        
        return data_dict    
    
    def normalize_object_size(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.normalize_object_size, config=config)
        points, gt_boxes = augmentor_utils.normalize_object_size(
            data_dict['gt_boxes'], data_dict['points'], data_dict['gt_boxes_mask'], config['SIZE_RES']
        )
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            gt_boxes, points = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                gt_boxes, points,
            )
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict
    
    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points = augmentor_utils.global_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
        )
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict
    
    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        gt_boxes, points = augmentor_utils.global_scaling(
            data_dict['gt_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE']
        )
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict
    
    def random_image_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_image_flip, config=config)
        images = data_dict["images"]
        depth_maps = data_dict["depth_maps"]
        gt_boxes = data_dict['gt_boxes']
        gt_boxes2d = data_dict["gt_boxes2d"]
        calib = data_dict["calib"]
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['horizontal']
            images, depth_maps, gt_boxes = getattr(augmentor_utils, 'random_image_flip_%s' % cur_axis)(
                images, depth_maps, gt_boxes, calib,
            )
        
        data_dict['images'] = images
        data_dict['depth_maps'] = depth_maps
        data_dict['gt_boxes'] = gt_boxes
        return data_dict
    
    def random_world_translation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_translation, config=config)
        noise_translate_std = config['NOISE_TRANSLATE_STD']
        if noise_translate_std == 0:
            return data_dict
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y', 'z']
            gt_boxes, points = getattr(augmentor_utils, 'random_translation_along_%s' % cur_axis)(
                gt_boxes, points, noise_translate_std,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_local_translation(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_translation, config=config)
        offset_range = config['LOCAL_TRANSLATION_RANGE']
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y', 'z']
            gt_boxes, points = getattr(augmentor_utils, 'random_local_translation_along_%s' % cur_axis)(
                gt_boxes, points, offset_range,
            )
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict
    
    def random_local_rotation(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_rotation, config=config)
        rot_range = config['LOCAL_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points = augmentor_utils.local_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
        )
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict
    
    def random_local_scaling(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_scaling, config=config)
        gt_boxes, points = augmentor_utils.local_scaling(
            data_dict['gt_boxes'], data_dict['points'], config['LOCAL_SCALE_RANGE']
        )
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict
    
    def random_world_frustum_dropout(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_world_frustum_dropout, config=config)
        
        intensity_range = config['INTENSITY_RANGE']
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for direction in config['DIRECTION']:
            assert direction in ['top', 'bottom', 'left', 'right']
            gt_boxes, points = getattr(augmentor_utils, 'global_frustum_dropout_%s' % direction)(
                gt_boxes, points, intensity_range,
            )
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict
    
    def random_local_frustum_dropout(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_frustum_dropout, config=config)
        
        intensity_range = config['INTENSITY_RANGE']
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for direction in config['DIRECTION']:
            assert direction in ['top', 'bottom', 'left', 'right']
            gt_boxes, points = getattr(augmentor_utils, 'local_frustum_dropout_%s' % direction)(
                gt_boxes, points, intensity_range,
            )
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict
    
    def random_local_pyramid_aug(self, data_dict=None, config=None):
        """
        Refer to the paper: 
            SE-SSD: Self-Ensembling Single-Stage Object Detector From Point Cloud
        """
        if data_dict is None:
            return partial(self.random_local_pyramid_aug, config=config)
        
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        
        gt_boxes, points, pyramids = augmentor_utils.local_pyramid_dropout(gt_boxes, points, config['DROP_PROB'])
        gt_boxes, points, pyramids = augmentor_utils.local_pyramid_sparsify(gt_boxes, points,
                                                                            config['SPARSIFY_PROB'],
                                                                            config['SPARSIFY_MAX_NUM'],
                                                                            pyramids)
        gt_boxes, points = augmentor_utils.local_pyramid_swap(gt_boxes, points,
                                                                 config['SWAP_PROB'],
                                                                 config['SWAP_MAX_NUM'],
                                                                 pyramids)
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict
    
    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)
        
        data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
            data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
        )
        if 'calib' in data_dict:
            data_dict.pop('calib')
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')
        if 'gt_boxes_mask' in data_dict:
            gt_boxes_mask = data_dict['gt_boxes_mask']
            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
            if 'gt_boxes2d' in data_dict:
                data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][gt_boxes_mask]
            
            data_dict.pop('gt_boxes_mask')
        return data_dict
    
    def re_prepare(self, augmentor_configs=None, intensity=None, aug_times=1):
        self.data_augmentor_queue = []

        if augmentor_configs is None:
            augmentor_configs = self.augmentor_configs

        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            
            # scale data augmentation intensity
            if intensity is not None:
                if cur_cfg.NAME == 'normalize_object_size':
                    #rate = np.power(0.5, aug_times) 
                    cur_cfg = self.adjust_augment_intensity_SN(cur_cfg, 0.25)
                    print ("***********cur_cfg:", aug_times)
                    print ("***********cur_cfg:", cur_cfg)
                else:
                    cur_cfg = self.adjust_augment_intensity(cur_cfg, intensity)
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)


    def adjust_augment_intensity_SN(self, config, rate):
        adjust_map = {
            'normalize_object_size': 'SIZE_RES',
        }

        def cal_new_intensity(config):
            origin_intensity_list = config.get(adjust_map[config.NAME])
            assert len(origin_intensity_list) == 3
            
            new_intensity_list = [x*rate for x in origin_intensity_list]
            return new_intensity_list

        if config.NAME not in adjust_map:
            return config
        
        # for data augmentations that init with 1
        #print ("***********config.NAME**************", config.get(adjust_map[config.NAME]))
        if config.NAME in ['normalize_object_size']:
            new_intensity_list = cal_new_intensity(config)
            setattr(config, adjust_map[config.NAME], new_intensity_list)
            return config
        else:
            raise NotImplementedError


    def adjust_augment_intensity(self, config, intensity):
        adjust_map = {
            #'normalize_object_size': 'SIZE_RES',
            'random_object_scaling': 'SCALE_UNIFORM_NOISE',
            'random_object_rotation': 'ROT_UNIFORM_NOISE',
            'random_world_rotation': 'WORLD_ROT_ANGLE',
            'random_world_scaling': 'WORLD_SCALE_RANGE',
        }

        def cal_new_intensity(config, flag):
            origin_intensity_list = config.get(adjust_map[config.NAME])
            assert len(origin_intensity_list) == 2
            assert np.isclose(flag - origin_intensity_list[0], origin_intensity_list[1] - flag)
            
            noise = origin_intensity_list[1] - flag
            new_noise = noise * intensity
            new_intensity_list = [flag - new_noise, new_noise + flag]
            return new_intensity_list

        if config.NAME not in adjust_map:
            return config
        
        # for data augmentations that init with 1
        if config.NAME in ['random_object_scaling', 'random_world_scaling']:
            new_intensity_list = cal_new_intensity(config, flag=1)
            setattr(config, adjust_map[config.NAME], new_intensity_list)
            return config
        elif config.NAME in ['random_object_rotation', 'random_world_rotation']:
            new_intensity_list = cal_new_intensity(config, flag=0)
            setattr(config, adjust_map[config.NAME], new_intensity_list)
            return config
        # modified
        # elif config.NAME in ['normalize_object_size']:
        #     new_intensity_list = cal_new_intensity(config, flag=0)
        #     setattr(config, adjust_map[config.NAME], new_intensity_list)
        #     return config
        else:
            raise NotImplementedError
