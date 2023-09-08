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
        label_file = (self.root_path / Path(idx)).parent.parent / "label.json"
        assert label_file.exists(), f"{label_file} does not exist"
        
        metadata_file = (self.root_path / Path(idx)).parent.parent / "metadata.json"
        assert metadata_file.exists(), f"{metadata_file} does not exist"
        with open(str(metadata_file), 'r') as f:
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
                loc[:,2] -= self.lidar_z_shift
                yaw = (dev_q.inv() * yaw_q).as_euler('xyz', degrees=False)[-1]
                avl_labels.append(
                    AvlObject(name, *loc, bb['l'], bb['w'], bb['h'], yaw))

        return avl_labels

    def get_lidar(self, idx):
        lidar_file = self.root_path / idx

        np_lidar_file = lidar_file.with_suffix('.npy')
        pkl_lidar_file = lidar_file.with_suffix('.pkl')
        if np_lidar_file.is_file(
        ):  # all preprocessing done in `convert_json_to_numpy`
            points = np.load(np_lidar_file, allow_pickle=False)
        elif pkl_lidar_file.is_file():
            with open(str(pkl_lidar_file), 'rb') as f:
                points = pickle.load(f)
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
    dataset = AVLTruckDataset(dataset_cfg=dataset_cfg,
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
            workers=int(sys.argv[3]) if sys.argv.__len__() > 3 else 4,
            creating_infos=True)
    elif sys.argv.__len__() > 1 and sys.argv[1] == 'json2np':
        convert_json_to_numpy(
            ROOT_DIR / 'data' / 'AVLTruck',
            num_workers=int(sys.argv[2]) if sys.argv.__len__() > 2 else 4)
