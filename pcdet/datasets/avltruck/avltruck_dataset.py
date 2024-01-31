import pickle
import orjson
from pathlib import Path
from dataclasses import dataclass

from scipy.spatial.transform import Rotation
import numpy as np

from ..avldataset.avl_dataset import AVLDataset
from ...utils import box_utils


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

class AVLTruckDataset(AVLDataset):

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
        label_files = list(((self.root_path / Path(idx)).parent.parent / "label").glob('*.json'))
        assert len(label_files) == 1, f"Found {len(label_files)} label files for {idx}"
        label_file = label_files[0]

        metadata_file = self.root_path / Path(idx)
        assert metadata_file.exists(), f"{metadata_file} does not exist"
        with open(str(metadata_file), 'r') as f:
            data = orjson.loads(f.read())

        dev_loc, dev_rot = data['device_position'], data['device_heading']
        dev_loc = np.array([dev_loc['x'], dev_loc['y'], dev_loc['z']])
        dev_q = Rotation.from_quat(
            [dev_rot['x'], dev_rot['y'], dev_rot['z'], dev_rot['w']])

        with open(str(label_file), 'r') as f:
            labels = orjson.loads(f.read())["labels"]

        #filter labels for labels in frame index
        labels = [l for l in labels if Path(idx).stem in l['file_id']]

        #merge bike and rider labels
        labels = self.merge_labels(labels)

        gt_boxes_lidar = []
        names = []
        for l in labels:
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
                loc[2] -= self.lidar_z_shift
                yaw = (dev_q.inv() * yaw_q).as_euler('xyz', degrees=False)[-1] + np.pi                
                names.append(name)
                gt_boxes_lidar.append(np.array([loc[0], loc[1], loc[2], bb['l'], bb['w'], bb['h'], yaw]))
        gt_boxes_lidar = np.array(gt_boxes_lidar)
        names = np.array(names)
        if len(gt_boxes_lidar) == 0:
            gt_boxes_lidar = np.zeros((0, 7))
            names = np.zeros((0))
        return {"gt_boxes_lidar": gt_boxes_lidar, "name": names}

    def get_lidar(self, idx, with_beam_label=False):
        lidar_file = self.root_path / idx

        if with_beam_label:
            lidar_path = Path(str(lidar_file.with_suffix("")) + "_beamlabels.npy")
            #check if file exists
            if not lidar_path.exists():
                raise FileNotFoundError(str(lidar_path)+' not found')
            points = np.load(lidar_path)
        else:
            np_lidar_file = lidar_file.with_suffix('.npy')
            assert np_lidar_file.exists(), f"{np_lidar_file} does not exist"
            points = np.load(np_lidar_file, allow_pickle=False)

        if self.train_fov_only:
            points = self.extract_fov_data(points, self.fov_angle_deg, self.lidar_heading_angle_deg)

        points[:, 3] = np.clip(points[:, 3], a_min=0, a_max=1.)
        points[:,2] -= self.lidar_z_shift

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
                annotations = self.get_label(sample_idx)

                if count_inside_pts:
                    if len(annotations['name']) == 0:
                        annotations['num_points_in_gt'] = np.zeros((0, ),
                                                                   dtype=np.int32)
                    else:
                        points = self.get_lidar(sample_idx)

                        corners_lidar = box_utils.boxes_to_corners_3d(
                            annotations['gt_boxes_lidar'])
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
        from tqdm import tqdm
        #debug
        infos = []
        for i, sid in tqdm(enumerate(sample_id_list),total=len(sample_id_list)):
            infos.append(process_single_scene(sid))
        #infos = Parallel(n_jobs=num_workers)(delayed(process_single_scene)(sid)
        #                                     for sid in tqdm(sample_id_list))
        return infos

def split_avl_data(data_path, train_test_split=0.8):
    import random
    dirs = [str(x) for x in (data_path/"sequences").iterdir() if x.is_dir()]
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

    random.seed(42)
    random.shuffle(t1)
    random.shuffle(t2)
    random.shuffle(t3)
    random.shuffle(t4)
    random.shuffle(t5)
    random.shuffle(t6)

    t1train, t1val = t1[:int(len(t1)*train_test_split)], t1[int(len(t1)*train_test_split):]
    t2train, t2val = t2[:int(len(t2)*train_test_split)], t2[int(len(t2)*train_test_split):]
    t3train, t3val = t3[:int(len(t3)*train_test_split)], t3[int(len(t3)*train_test_split):]
    t4train, t4val = t4[:int(len(t4)*train_test_split)], t4[int(len(t4)*train_test_split):]
    t5train, t5val = t5[:int(len(t5)*train_test_split)], t5[int(len(t5)*train_test_split):]
    t6train, t6val = t6[:int(len(t6)*train_test_split)], t6[int(len(t6)*train_test_split):]

    #t2train, t2val = t2[:len(t2) // 2], t2[len(t2) // 2:]
    #t3train, t3val = t3[:len(t3) // 2], t3[len(t3) // 2:]
    #t4train, t4val = t4[:len(t4) // 2], t4[len(t4) // 2:]
    #t5train, t5val = t5[:len(t5) // 2], t5[len(t5) // 2:]
    #t6train, t6val = t6[:len(t6) // 2], t6[len(t6) // 2:]

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
                     workers=4,
                     creating_infos=True):
    dataset = AVLTruckDataset(dataset_cfg=dataset_cfg,
                         class_names=class_names,
                         root_path=data_path,
                         training=False,
                         creating_infos=creating_infos)

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

# python -m pcdet.datasets.avltruck.avltruck_dataset create_avl_infos tools/cfgs/dataset_configs/avltruck/OD/avltruck_dataset.yaml
# python -m pcdet.datasets.avltruck.avltruck_dataset create_avl_gtdatabase_with_beamlabels cfgs/dataset_configs/avltruck/OD/avltruck_dataset.yaml

if __name__ == '__main__':
    import sys
    from pathlib import Path
    ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_avl_infos':
        import yaml
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        create_avl_infos(dataset_cfg=dataset_cfg,
                         class_names=['Vehicle_Drivable_Car', 'Vehicle_Drivable_Van', 
                                      'Human',
                                      'Vehicle_Ridable_Motorcycle', 'Vehicle_Ridable_Bicycle',
                                      'LargeVehicle_Bus', 'LargeVehicle_TruckCab','LargeVehicle_Truck', 
                                      'Trailer'],
                                      data_path=ROOT_DIR / 'data' / 'avltruck',
                                      save_path=ROOT_DIR / 'data' / 'avltruck',
                                      creating_infos=True)
        
    if sys.argv.__len__() > 1 and sys.argv[1] == 'split_avl_data':
        import yaml
        from easydict import EasyDict
        data_path=ROOT_DIR / 'data' / 'avltruck'
        split_avl_data(data_path)
    elif sys.argv.__len__() > 1 and sys.argv[1] == 'json2np':
        convert_json_to_numpy(
            ROOT_DIR / 'data' / 'avltruck',
            num_workers=int(sys.argv[2]) if sys.argv.__len__() > 2 else 4)
    elif sys.argv.__len__() > 1 and sys.argv[1] == 'debug':
        import yaml
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        class_names=[
                'Vehicle_Drivable_Car', 'Vehicle_Drivable_Van', 'Human',
                'Vehicle_Ridable_Motorcycle', 'Vehicle_Ridable_Bicycle',
                'LargeVehicle_Bus', 'LargeVehicle_TruckCab',
                'LargeVehicle_Truck', 'Trailer']
        
        data_path=ROOT_DIR / 'data' / 'avltruck'

        dataset = AVLTruckDataset(dataset_cfg=dataset_cfg,
                            class_names=class_names,
                            root_path=data_path,
                            training=False,
                            creating_infos=True)
        import cProfile
        cProfile.run("for i in range(1000): dataset.__getitem__(4850)", sort="cumtime")
    elif sys.argv.__len__() > 1 and sys.argv[1] == 'create_avl_gtdatabase_with_beamlabels':
        import yaml
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        class_names=[
                'Vehicle_Drivable_Car', 'Vehicle_Drivable_Van', 'Human',
                'Vehicle_Ridable_Motorcycle', 'Vehicle_Ridable_Bicycle',
                'LargeVehicle_Bus', 'LargeVehicle_TruckCab',
                'LargeVehicle_Truck', 'Trailer']
        
        data_path=ROOT_DIR / 'data' / 'avltruck'

        dataset = AVLTruckDataset(dataset_cfg=dataset_cfg,
                            class_names=class_names,
                            root_path=data_path,
                            training=False,
                            creating_infos=True)
        
        dataset.create_groundtruth_database(data_path / f'avl_infos_train.pkl', class_names, split='train', with_beam_labels=True)
