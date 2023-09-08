import pickle
import orjson
from pathlib import Path

from scipy.spatial.transform import Rotation
import numpy as np

from ..avldataset.avl_dataset import AVLDataset
from .avl_utils import apply_transform, load_file, transform_matrix
from ...utils import box_utils



class AVLRooftopDataset(AVLDataset):

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
                        logger=logger,
                        creating_infos=creating_infos)
        self.is_fov = self.dataset_cfg.get('FOV_POINTS_ONLY',False)
        
    def get_label(self, idx):
        sequence_name = Path(idx).parent.parent.parent.stem
        #load metadata
        metadata_path = self.root_path / 'sequences' / sequence_name / "unpacked" / "info.pkl"
        metadata = load_file(metadata_path, file_type='pickle')
    
        sequence_labels = self.get_sequence_labels(sequence_name)
        frame_nr = idx.split("/")[-1].split(".")[0]   #eg. 0001
        labels = [label for label in sequence_labels if 
                  label['file_id'].split("_")[-1].split(".")[0] == frame_nr and
                  label['sensor_id'] == 'lidar' and
                  label['label_type'] == '3d_bbox']
        if len(labels) == 0:
            return None
        
        #transform labels to lidar frame
        frame_metadata = metadata[labels[0]["file_id"].split(".")[0]]
        R_ref_to_waymo_lidar = frame_metadata["lidar_sensors"]["lidar_0"]["R_ref_to_waymo_lidar"]
        device_heading_dict = frame_metadata["lidar_sensors"]["lidar_0"]["device_heading"]
        device_position = frame_metadata["lidar_sensors"]["lidar_0"]["device_position"]
        dev_q = Rotation.from_quat(
            [device_heading_dict['x'], 
             device_heading_dict['y'], 
             device_heading_dict['z'], 
             device_heading_dict['w']])
        
        gt_boxes_lidar = []
        names = []
        for label in labels:
            bbox = label['three_d_bbox']
            bbox_position = np.array([bbox['cx'], bbox['cy'], bbox['cz']])
            name = label["label_category_id"]
            gt_boxes_lidar.append(np.array([bbox_position[0], bbox_position[1], bbox_position[2], bbox['l'], bbox['w'], bbox['h'], -bbox["rot_z"]]))
            names.append(name)

        gt_boxes_lidar = np.array(gt_boxes_lidar)
        names = np.array(names)

        gt_boxes_lidar[:,:3] = (gt_boxes_lidar[:,:3] - device_position) @ R_ref_to_waymo_lidar
        gt_boxes_lidar[:,6] =  gt_boxes_lidar[:,6] - dev_q.as_rotvec()[-1]
        
        gt_boxes_lidar[:,2] -= self.lidar_z_shift

        return {"gt_boxes_lidar": gt_boxes_lidar, "name": names}
    
    def get_sequence_labels(self, sequence_name):
        unpacked_path = self.root_path / 'sequences' / sequence_name / "unpacked"
        label_path = unpacked_path / 'labels.json'
        lidar_labels = load_file(label_path, file_type='json')
        return lidar_labels["labels"]

    def get_lidar(self, idx):
        # def remove_ego_points(pts, center_radius=2.0):
        #     mask = np.linalg.norm(pts[:, :2], axis=1) > center_radius
        #     return pts[mask]

        lidar_path = Path(self.root_path) / idx
        if not lidar_path.exists():
            raise FileNotFoundError(str(lidar_path)+' not found')
        if lidar_path.suffix == '.bin':
            points = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 4)
        elif lidar_path.suffix == '.npy':
            points = np.load(lidar_path)
        elif lidar_path.suffix == '.pkl':
            with lidar_path.open(mode='rb') as fp:
                points = pickle.load(fp)
        else:
            raise NotImplementedError

        if self.is_fov:
            angle = np.arctan2(points[:, 1], points[:, 0])
            mask = (angle > -self.fov / 2.) & (angle < self.fov / 2.)
            points = points[mask]
        ##load info file
        #sequence_name = Path(idx).parent.parent.parent.stem
        #metadata_path = self.root_path / 'sequences' / sequence_name / "unpacked" / "info.pkl"
        #metadata = load_file(metadata_path, file_type='pickle')
        #frame_nr = idx.split("/")[-1].split(".")[0]   #eg. 0001
        #frame_metadata = metadata["camera_front__"+frame_nr]
        ##transform points to lidar frame
        #device_position_dict = frame_metadata["lidar_sensors"]["lidar_0"]["device_position"]
        #device_heading_dict = frame_metadata["lidar_sensors"]["lidar_0"]["device_heading"]
        #
        #dev_position = np.array([device_position_dict['x'], 
        #                         device_position_dict['y'], 
        #                         device_position_dict['z']])
        #dev_q = Rotation.from_quat(
        #    [device_heading_dict['x'], 
        #     device_heading_dict['y'], 
        #     device_heading_dict['z'], 
        #     device_heading_dict['w']])
    #
        #
        #R_avl_lidar_to_waymo_lidar = np.array([[-1, 0, 0],
        #                                       [0,  -1, 0],
        #                                       [0,  0, 1]])
        #R_ref_to_avl_lidar = dev_q.as_matrix()
        #R_ref_to_waymo_liar = R_avl_lidar_to_waymo_lidar @ R_ref_to_avl_lidar
        #points[:,:3] = (points[:,:3] - dev_position) @ R_ref_to_waymo_liar 

        points[:, -1] = np.clip(points[:, -1], a_min=0, a_max=1.)
        points[:,2] -= self.lidar_z_shift

        return points
    
    def get_sequence_list(self):
        '''
        extract sequences from sample_id_list
        '''
        sequence_list_nonunique = [Path(sample_id).parent.parent.parent.stem 
                                    for sample_id in self.sample_id_list]
        sequence_list = list(set(sequence_list_nonunique))
        return sequence_list
    

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        '''
        generate info for each sequence in self.sample_id_list
        '''
        from tqdm import tqdm
        # gather dataset infos
        sequence_list =  self.get_sequence_list()

        from joblib import Parallel, delayed

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            annotations = self.get_label(sample_idx)

            if count_inside_pts:
                points = self.get_lidar(sample_idx)

                corners_lidar = box_utils.boxes_to_corners_3d(
                    annotations["gt_boxes_lidar"])
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
        # avl_info_list = []
        # process_bar = tqdm(sequence_list, desc='Fill infos')
        # for seq_name in process_bar:
        #     process_bar.set_postfix_str(seq_name)
        #     unpacked_path = self.root_path / 'sequences' / seq_name / "unpacked"
        #     label_path = unpacked_path / 'labels.json'
        #     info_path = unpacked_path / 'info.pkl'

            # # load info/event file
            # seq_infos_dict = load_file(info_path, file_type='pickle')
            # # load and organize lidar labels
            # seq_lidar_labels = load_file(label_path, file_type='json')

            # #create label dict from lidar label files
            # label_dict = {}
            # if lidar_labels is not None:
            #     for label in lidar_labels['labels']:
            #         file_id = Path(label['file_id']).stem
            #         if file_id not in label_dict:
            #             label_dict[file_id] = {'gt_boxes_lidar': [], 'name': [],
            #                                 'num_points_in_gt': [], 'gt_boxes_token': []}
            #         if label['sensor_id'] == 'lidar' and label['label_type'] == '3d_bbox':
            #             box = label['three_d_bbox']
            #             bb = np.array([box['cx'], box['cy'], box['cz'], box['l'], box['w'], box['h'], box['rot_z']])
                        

            #             label_dict[file_id]['gt_boxes_lidar'].append(bb)
            #             label_dict[file_id]['name'].append(label['label_category_id'])
            #             label_dict[file_id]['gt_boxes_token'].append(label['label_id'])
            #             if count_inside_pts:
            #                 frame_id = file_id.split("_")[-1]   #eg. 0001
            #                 lidar_path = unpacked_path / 'lidar' / ('%s.pkl' % frame_id)
            #                 points = load_file(lidar_path, file_type='pickle')
                            
            #                 corners_lidar = box_utils.boxes_to_corners_3d(np.expand_dims(bb, axis=0))[0]   #box_utils needs (N, 7) arraym [0] to flatten again
                            
            #                 flag = box_utils.in_hull(points[:, 0:3],
            #                                         corners_lidar)
            #                 num_points_in_gt = flag.sum()
            #                 label_dict[file_id]['num_points_in_gt'].append(num_points_in_gt)
            
            # # add labels to preprocessed infos (labelled data) and append to global info list
            # if infos_dict is not None:
            #     for info_id, info in infos_dict.items():
            #         #save point cloud data
            #         save_info = {}
            #         pc_info = {'num_features': 4, 
            #                 'lidar_idx': 'sequences/'+info["lidar_sensors"]["lidar_0"]["path"],
            #                 'timestamp': info["timestamp"],
            #                 'sequence_name': info["name"],
            #                 'file_id': info["file_id"],
            #                 'lidar_to_ref': info["lidar_sensors"]["lidar_0"]["lidar_to_ref"],
            #                 'world_to_lidar': info["lidar_sensors"]["lidar_0"]["world_to_lidar"]
            #                     }
            #         save_info['point_cloud'] = pc_info
                    
            #         #save label data
            #         if info_id in label_dict:
            #             labels = label_dict[info_id]
            #             for k, v in labels.items():
            #                 labels[k] = np.array(v)
            #             save_info["annos"] = labels
            #         else:
            #             save_info["annos"] = {'gt_boxes_lidar': np.array([]), 
            #                         'name': np.array([]),
            #                         'num_points_in_gt': np.array([]), 
            #                         'gt_boxes_token': np.array([])}
            #         avl_info_list.append(save_info)
        #return avl_info_list
    
def split_avl_data(data_path, sequence_file_path, train_test_split=0.8):
    '''
    loads sequence_file_path and splits the contained sequences into train and val, and 
    stores the resulting frame ids in train.txt and val.txt
    '''
    import random

    #folders = [f.stem for f in (data_path/"sequences").iterdir() if f.is_dir()]
    #read folders from sequence_file_path
    with open(sequence_file_path, 'r') as f:
        seqs = f.readlines()
    seqs = [f.strip() for f in seqs]
    sequence_categories_list = ["_".join(folder.split("_")[:3]) for folder in seqs]
    
    #dict with number of occurences of each sequence category
    categories_count = {i:sequence_categories_list.count(i) for i in sequence_categories_list}
    sequence_categories = list(set(sequence_categories_list))

    print("Number of sequences: ", len(sequence_categories_list))
    print("Number of sequence categories: ", len(sequence_categories))


    dirs = [str(f) for f in (data_path/"sequences").iterdir() if f.is_dir()]

    #filter out all sequences that are not in sequence list (seqs)
    
    dirs = [d for d in dirs if d.split("/")[-1] in seqs]

    assert(len(dirs) == len(seqs), "Sequences in sequence list not found in sequence folder.")

    category_sequences = []
    for category in sequence_categories:
        category_sequences.append([x for x in dirs if category in x])

    #shuffle sequences within each category
    for sequence in category_sequences:
        random.shuffle(sequence)

    train_sequences = []
    val_sequences = []

    #split sequences from each category into train and val
    #We train on sequences of each category
    for i, category in enumerate(category_sequences):
        print("Number of sequences in category",sequence_categories[i],":", len(category))
        #we can grab the sequences in order because they are already shuffled
        if (len(category) == 1):
            train_sequences.extend(category)
            continue
        train_sequences.extend(category[:int(len(category)*train_test_split)])
        val_sequences.extend(category[int(len(category)*train_test_split):])

    print("Number of train sequences:", len(train_sequences))
    print("Number of val sequences:", len(val_sequences))

    for split in ['train', 'val']:
        sequences = train_sequences if split == 'train' else val_sequences
        with open(str(data_path / (split+'.txt')), 'w') as f:
            for seq in sequences:
                frames = [frame for frame in (Path(seq)/'unpacked'/'lidar').glob('*.pkl')]
                for frame in frames:
                    seq_id = "/".join(str(frame).split('/')[-5:])
                    f.write(seq_id)
                    f.write('\n')
    #total number of frames in train and val
    num_train_frames = sum([len([frame for frame in (Path(seq)/'unpacked'/'lidar').glob('*.pkl')]) for seq in train_sequences])
    num_val_frames = sum([len([frame for frame in (Path(seq)/'unpacked'/'lidar').glob('*.pkl')]) for seq in val_sequences])
    print("Number of train frames:", num_train_frames)
    print("Number of val frames:", num_val_frames)
    print("Total number of frames:", num_train_frames + num_val_frames)
    print("Train/Val split:", num_train_frames/(num_train_frames + num_val_frames))
# def _create_avl_infos(splits, 
#                      dataset_cfg,
#                      class_names,
#                      data_path,
#                      save_path,
#                      workers=4):
#     data_path = data_path 
#     save_path = save_path

#     assert all([split in ['train', 'val', 'test'] for split in splits])
#     train_split = 'train'
#     dataset = AVLRooftopDataset(dataset_cfg=dataset_cfg,
#                          class_names=class_names,
#                          root_path=data_path,
#                          training=False)
#     for split in splits:
#         print('---------------Start to generate data infos for split %s---------------'%split)
#         frame_list_path = data_path / '{}.txt'.format(split)
#         with frame_list_path.open('r') as fp:
#             frame_list = [line.strip() for line in fp.readlines()]

#         # gather dataset infos
#         save_path_tmp = save_path / ('avl_infos_%s.pkl' % split)
#         all_sequence_list = [Path(p).parent.parent.parent.stem for p in frame_list]
#         sequence_list = list(set(all_sequence_list))

#         avl_infos = fill_avl_infos(data_path, sequence_list)

#         print('%s sample: %d' % (split, len(avl_infos)))
#         with open(save_path_tmp, 'wb') as f:
#             pickle.dump(avl_infos, f)

#     print('---------------Start to generate gt database for split %s---------------'%split)
#     dataset.set_split("train")
#     train_filename = save_path / f'avl_infos_{train_split}.pkl'
#     dataset.create_groundtruth_database(train_filename, split=train_split)


def create_avl_infos(dataset_cfg,
                     class_names,
                     data_path,
                     save_path,
                     workers=4):
    dataset = AVLRooftopDataset(dataset_cfg=dataset_cfg,
                         class_names=class_names,
                         root_path=data_path,
                         training=False,
                         creating_infos=True)

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



# def create_avl_infos(dataset_cfg,
#                      class_names,
#                      data_path,
#                      save_path,
#                      workers=4):
#     dataset = AVLRooftopDataset(dataset_cfg=dataset_cfg,
#                          class_names=class_names,
#                          root_path=data_path,
#                          training=False)

#     train_split, val_split = 'train', 'val'
#     train_filename = save_path / f'avl_infos_{train_split}.pkl'
#     val_filename = save_path / f'avl_infos_{val_split}.pkl'

#     print('---------------Start to generate data infos---------------')
#     dataset.set_split(train_split)
#     avl_infos_train = dataset.get_infos(num_workers=workers,
#                                         has_label=True,
#                                         count_inside_pts=True)
#     with open(train_filename, 'wb') as f:
#         pickle.dump(avl_infos_train, f)
#     print(f'AVL info train file is saved to {train_filename}')

#     dataset.set_split(val_split)
#     avl_infos_val = dataset.get_infos(num_workers=workers,
#                                       has_label=True,
#                                       count_inside_pts=True)
#     with open(val_filename, 'wb') as f:
#         pickle.dump(avl_infos_val, f)
#     print(f'AVL info val file is saved to {val_filename}')

#     print(
#         '---------------Start create groundtruth database for data augmentation---------------'
#     )
#     dataset.set_split(train_split)
#     dataset.create_groundtruth_database(train_filename, split=train_split)

#     print('---------------Data preparation Done---------------')



#['Vehicle_Ridable_Bicycle', 'Vehicle_Ridable_Motorcycle', 
# 'LargeVehicle_Truck', 'LargeVehicle_TruckCab', 
# 'Trailer', 'LargeVehicle_Bus', 'LargeVehicle_Bus_Bendy', 
# 'Vehicle_Drivable_Van', 'Vehicle_Drivable_Car', 
# 'Human', 'PPObject', 
# 'PPObject_Stroller', 'PPObject_BikeTrailer', 
# 'Vehicle_PMD']


# python -m pcdet.datasets.avlrooftop.downloader --root_path /data/AVLRooftop/ --sequence_file /data/AVLRooftop/training_sequences_small.txt --split sequences
# python -m pcdet.datasets.avlrooftop.avlrooftop_dataset split_avl_data
#
# cd 3DTrans/
# python -m pcdet.datasets.avlrooftop.avlrooftop_dataset create_avl_infos tools/cfgs/dataset_configs/avlrooftop/OD/avlrooftop_dataset.yaml
if __name__ == '__main__':
    import sys
    from pathlib import Path
    #ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
    ROOT_DIR = Path("/")
    if sys.argv[1] == 'split_avl_data':
        data_path = ROOT_DIR / 'data' / 'AVLRooftop'
        sequence_file_path = sys.argv[2]
        split_avl_data(data_path, sequence_file_path=sequence_file_path)
    elif sys.argv.__len__() > 1 and sys.argv[1] == 'create_avl_infos':
        import yaml
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        class_names = ['Vehicle_Ridable_Bicycle', 'Vehicle_Ridable_Motorcycle', 
            'LargeVehicle_Truck', 'LargeVehicle_TruckCab', 
            'Trailer', 'LargeVehicle_Bus', 'LargeVehicle_Bus_Bendy', 
            'Vehicle_Drivable_Van', 'Vehicle_Drivable_Car', 
            'Human', 'PPObject', 
            'PPObject_Stroller', 'PPObject_BikeTrailer', 
            'Vehicle_PMD']
        create_avl_infos(
            dataset_cfg = dataset_cfg,
            class_names = class_names,
            data_path=ROOT_DIR / 'data' / 'AVLRooftop',
            save_path=ROOT_DIR / 'data' / 'AVLRooftop',)
        

