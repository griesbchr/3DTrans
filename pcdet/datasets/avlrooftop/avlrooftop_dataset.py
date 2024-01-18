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
        
        #merge bike and rider labels
        labels = self.merge_labels(labels)

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

    def get_lidar(self, idx, with_beam_label=False):
        # def remove_ego_points(pts, center_radius=2.0):
        #     mask = np.linalg.norm(pts[:, :2], axis=1) > center_radius
        #     return pts[mask]

        lidar_path = Path(self.root_path) / idx

        if with_beam_label:
            lidar_path += "_beamlabels.npy"
            #check if file exists
            if not lidar_path.exists():
                raise FileNotFoundError(str(lidar_path)+' not found')
            points = np.load(lidar_path)

        else:   
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

        if self.train_fov_only:
            points = self.extract_fov_data(points, self.fov_angle_deg, self.lidar_heading_angle_deg)

        #points[:, -1] = np.clip(points[:, -1], a_min=0, a_max=1.)
        points[:, 3] *= 255
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
    

    def get_infos(self, num_workers=8, has_label=True, count_inside_pts=True, sample_id_list=None):
        '''
        generate info for each sequence in self.sample_id_list
        '''
        from tqdm import tqdm
        # gather dataset infos

        from joblib import Parallel, delayed

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            annotations = self.get_label(sample_idx)

            if count_inside_pts and len(annotations['name']) > 0:
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
        
        #debug
        infos = []
        for sample_idx in tqdm(sample_id_list, desc='generate infos'):
            infos.append(process_single_scene(sample_idx))
        
        #infos = Parallel(n_jobs=num_workers)(delayed(process_single_scene)(sid)
        #                                     for sid in sample_id_list)
        return infos
    
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
    sequence_categories_list = ["_".join(folder.split("_")[:1]) for folder in seqs]
    
    #dict with number of occurences of each sequence category
    categories_count = {i:sequence_categories_list.count(i) for i in sequence_categories_list}
    sequence_categories = list(set(sequence_categories_list))

    print("Number of sequences: ", len(sequence_categories_list))
    print("Number of sequence categories: ", len(sequence_categories))


    dirs = [str(f) for f in (data_path/"sequences").iterdir() if f.is_dir()]

    #filter out all sequences that are not in sequence list (seqs)
    
    dirs = [d for d in dirs if d.split("/")[-1] in seqs]

    #assert(len(dirs) == len(seqs), "Sequences in sequence list not found in sequence folder.")

    #if second to last char is an underline, then remove last two chars
    dirs_ext_sequences = list(set([d[:-2] if d[-2] == "_" else d for d in dirs]))

    category_sequences = []
    for category in sequence_categories:
        category_sequences.append([x for x in dirs_ext_sequences if category in x])

    #shuffle sequences within each category with fixed seed
    random.seed(42)
    for sequence in category_sequences:
        random.shuffle(sequence)

    ext_train_sequences = []
    ext_val_sequences = []

    #split sequences from each category into train and val
    #We train on sequences of each category
    for i, category in enumerate(category_sequences):
        print("Number of sequences in category",sequence_categories[i],":", len(category))
        #we can grab the sequences in order because they are already shuffled
        if (len(category) == 1):
            ext_train_seq.extend(category)
            continue

        ext_train_sequences.extend(category[:int(len(category)*train_test_split)])
        ext_val_sequences.extend(category[int(len(category)*train_test_split):])

    # get all sequences that are in ext_sequences
    train_sequences = []
    for ext_train_seq in ext_train_sequences:
        train_sequences += [dir for dir in dirs if ext_train_seq in dir]
    val_sequences = []
    for ext_val_seq in ext_val_sequences:
        val_sequences += [dir for dir in dirs if ext_val_seq in dir]

    print("Number of train sequences:", len(train_sequences), " which is about ", len(train_sequences)/len(dirs)*100, "percent of all sequences")
    print("Number of val sequences:", len(val_sequences), " which is about ", len(val_sequences)/len(dirs)*100, "percent of all sequences")


    #the frame names have the following structure: '/home/cgriesbacher/thesis/3DTrans/data/avlrooftop/sequences/CATEGORY_SUBCATEGORY_SUBSUBCATEGORY_20200528101524'
    #list all frames in subcategory for train and val

    #for each frame in train, filter the subcategory
    train_subcategory_list = []
    for train_seq in train_sequences:
        train_subcategory_list.append(train_seq.split("/")[-1].split("_")[1])
    train_subcategories = list(set(train_subcategory_list))
    for train_subcategory in train_subcategories:
        print("Number of train frames in subcategory", train_subcategory, ":", len([seq for seq in train_sequences if train_subcategory in seq]))
    #for each frame in val, filter the subcategory
    val_subcategory_list = []
    for val_seq in val_sequences:
        val_subcategory_list.append(val_seq.split("/")[-1].split("_")[1])
    val_subcategories = list(set(val_subcategory_list))
    for val_subcategory in val_subcategories:
        print("Number of val frames in subcategory", val_subcategory, ":", len([seq for seq in val_sequences if val_subcategory in seq]))
    
    print("\n")
    #for each frame in train, filter the subsubcategory
    train_subsubcategory = []
    for train_seq in train_sequences:
        train_subsubcategory.append(train_seq.split("/")[-1].split("_")[2])
    train_subcategories = list(set(train_subsubcategory))
    for train_subcategory in train_subcategories:
        print("Number of train frames in subcategory", train_subcategory, ":", len([seq for seq in train_sequences if train_subcategory in seq]))

    #for each frame in val, filter the subcategory
    val_subsubcategory = []
    for val_seq in val_sequences:
        val_subsubcategory.append(val_seq.split("/")[-1].split("_")[2])
    val_subcategories = list(set(val_subsubcategory))
    for val_subcategory in val_subcategories:
        print("Number of val frames in subcategory", val_subcategory, ":", len([seq for seq in val_sequences if val_subcategory in seq]))
        

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



def create_avl_infos(dataset_cfg,
                     class_names,
                     data_path,
                     save_path,
                     workers=8):
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



#['Vehicle_Ridable_Bicycle', 'Vehicle_Ridable_Motorcycle', 
# 'LargeVehicle_Truck', 'LargeVehicle_TruckCab', 
# 'Trailer', 'LargeVehicle_Bus', 'LargeVehicle_Bus_Bendy', 
# 'Vehicle_Drivable_Van', 'Vehicle_Drivable_Car', 
# 'Human', 'PPObject', 
# 'PPObject_Stroller', 'PPObject_BikeTrailer', 
# 'Vehicle_PMD']


# python -m pcdet.datasets.avlrooftop.avlrooftop_dataset split_avl_data
#
# cd 3DTrans/
# python -m pcdet.datasets.avlrooftop.avlrooftop_dataset create_avl_infos tools/cfgs/dataset_configs/avlrooftop/OD/avlrooftop_dataset.yaml
if __name__ == '__main__':
    import sys
    from pathlib import Path
    ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
    if sys.argv[1] == 'split_avl_data':
        data_path = ROOT_DIR / 'data' / 'avlrooftop'
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
            data_path=ROOT_DIR / 'data' / 'avlrooftop',
            save_path=ROOT_DIR / 'data' / 'avlrooftop',)

    elif sys.argv.__len__() > 1 and sys.argv[1] == 'debug':
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
        
        data_path=ROOT_DIR / 'data' / 'avlrooftop'

        dataset = AVLRooftopDataset(dataset_cfg=dataset_cfg,
                            class_names=class_names,
                            root_path=data_path,
                            training=False,
                            creating_infos=True)
        import cProfile
        cProfile.run("for i in range(1000): dataset.__getitem__(1234)", sort="cumtime")
    elif sys.argv.__len__() > 1 and sys.argv[1] == 'create_avl_gtdatabase_with_beamlabels':
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
        
        data_path=ROOT_DIR / 'data' / 'avlrooftop'

        dataset = AVLRooftopDataset(dataset_cfg=dataset_cfg,
                            class_names=class_names,
                            root_path=data_path,
                            training=False,
                            creating_infos=True)
        dataset.create_groundtruth_database(data_path / f'avl_infos_train.pkl', class_names, split='train', with_beam_labels=True)
            