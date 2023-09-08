import pickle

import yaml
from tqdm import tqdm
import json
import shutil

import numpy as np

from pathlib import Path
from pyquaternion import Quaternion
from copy import deepcopy
from ...utils import box_utils
# from natsort import natsorted, ns
from scipy.spatial.transform import Rotation


def apply_transform(pts, transformation):
    pts_ = deepcopy(pts)
    ones = np.ones((len(pts_), 1))
    pts_wo_h = np.hstack((pts_[:, :3], ones))
    pts_[..., :3] = np.einsum('ij,kj->ki', transformation, pts_wo_h)[..., :3]
    return pts_


def load_file(file_path, file_type='pickle'):
    if file_path.exists():
        if file_type == 'pickle':
            with file_path.open('rb') as fp:
                data = pickle.load(fp)
        elif file_type == 'json':
            with file_path.open(mode='r') as lf:
                data = json.load(lf)
        elif file_type == 'dat':
            with file_path.open(mode='r') as f:
                data = f.read().splitlines()
        elif file_type == 'yaml':
            with file_path.open(mode='r') as fp:
                data = yaml.load(fp, Loader=yaml.FullLoader)
        else:
            raise NotImplementedError
        return data
    else:
        return None


def load_calibration_from_metadata(metadata):
    calibration = {}
    components = {}
    calibrations = {}
    key_mapping = {}
    for t in metadata['streams'][0]['signals']:
        comp_ref = t['component_ref']
        while comp_ref in components:
            comp_ref += 1
        components.update({comp_ref: t['name']})
        key_mapping[comp_ref] = t['component_ref']
    [calibrations.update({s['id']: s['parameters']['calibration']}) for s in metadata['systems'][0]['components']]
    for k, v in components.items():
        # topic is new key
        new_key = v.replace('/', '')
        calibration[new_key] = {}
        calibration[new_key].update(calibrations[key_mapping[k]])
        sensor_type = 'gnss'
        if int(k) // 100 == 1:
            sensor_type = 'camera'
        elif int(k) // 100 == 2:
            sensor_type = 'lidar'

        calibration[new_key].update({'id': k, 'type': sensor_type, 'index': k % 100})
        if 'extrinsic' in calibration[new_key]:
            extrinsic = calibration[new_key]['extrinsic']['transformation_to_reference_system']
            calibration[new_key]['extrinsic'] = np.array(extrinsic).reshape(4, 4)
    return calibration


def load_calibration_from_yaml(calibration_yaml):
    calibration = {}
    for topic, params in calibration_yaml.items():
        if len(params) > 0 and topic != 'oxts':
            sensor_type = 'camera' if 'model' in params else 'lidar'
            calibration[topic] = {'type': sensor_type,
                                  'extrinsic': np.array(params['CoordTrns']),
                                  'index': 0 if topic == 'Bsw_CloudCentrRaw' else 99}
            if 'intrinsic' in params:
                intrinsic = {}
                [intrinsic.update(entry) for entry in params['intrinsic']]
                calibration[topic]['intrinsic'] = intrinsic

    return calibration


def transform_matrix(position, heading, inverse: bool = False) -> np.ndarray:
    translation = np.array([position['x'], position['y'], position['z']])
    quaternion = np.array([heading['w'], heading['x'], heading['y'], heading['z']])
    rotation = Quaternion(quaternion)
    tm = np.eye(4)

    if inverse:
        rot_inv = rotation.rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation.rotation_matrix
        tm[:3, 3] = np.transpose(np.array(translation))

    return tm


def extract_frame(root_path, frame_path, frame_nr, dataset_name, out_path, remove_source=False):
    print("frame path: ", frame_path)
    pcl_ref_pts = []
    with frame_path.open(mode='r') as fp:
        f = json.load(fp)

    info = {
        'timestamp': f['timestamp'],
        'name': dataset_name,
        'file_id': frame_path.stem
    }

    label_path = root_path / dataset_name / 'raw' / 'labels.json'

    if label_path.exists():
        shutil.copy(label_path, out_path / 'labels.json')

    cameras = {}
    for cid, camera in enumerate(f['images']):
        # camera transforms
        camera_name = 'camera_%d' % cid
        camera_to_ref = transform_matrix(camera['position'], camera['heading'], inverse=False)
        intrinsics = [camera['fx'], camera['cx'], camera['fy'], camera['cy']]
        # prepare image movement
        source_path = frame_path.parent / camera['image_url']
        destination_path = out_path / camera_name / ('%s%s' % (frame_nr, Path(camera['image_url']).suffix))
        # move camera images
        if source_path.exists():
            destination_path.parent.mkdir(exist_ok=True, parents=True)
            if remove_source:
                shutil.move(source_path, destination_path)
            else:
                shutil.copy(source_path, destination_path)
            # update info dict
            cameras[camera_name] = {'intrinsics': intrinsics, 'cam_to_ref': camera_to_ref,
                                 'path': Path(destination_path).relative_to(root_path).__str__()}
        else:
            print('%s in %s [%s] not available.' % (camera_name, dataset_name, frame_nr))

    info['cameras'] = cameras

    # lidar transforms
    device_position_dict = f['device_position']
    device_heading_dict = f['device_heading']
    ##
    device_position = np.array([device_position_dict['x'], 
                                device_position_dict['y'], 
                                device_position_dict['z']])
    device_q = Rotation.from_quat(
        [device_heading_dict['x'], 
            device_heading_dict['y'], 
            device_heading_dict['z'], 
            device_heading_dict['w']])

    R_ref_to_waymo_lidar = device_q.as_matrix() 
    
    lidar_out_path = out_path / 'lidar' / ('%s.pkl' % frame_nr)
    world_to_lidar = np.eye(4)
    if 'gnns' in f:
        world_to_lidar = np.array(f['gnss']['world2front_lidar']).reshape((4, 4))

    lidar_sensors = {}
    # extract and dump lidar data if not available
    lidar_out_path.parent.mkdir(exist_ok=True, parents=True)
    if not lidar_out_path.exists():
        for point in f['points']:
            pcl_ref_pts.append(np.array([point['x'], point['y'], point['z'], point['i']]))
            if len(pcl_ref_pts) == 0:
                print('No lidar points in %s [%s]' % (dataset_name, frame_nr))
        points = np.array(pcl_ref_pts)
        points[:,:3] = (points[:,:3] - device_position) @ R_ref_to_waymo_lidar
        
        points[:, 3] /= 255
        points[:, 3] = np.tanh(points[:, 3])
        with lidar_out_path.open(mode='wb') as lop:
            pickle.dump(np.array(points), lop)

    lidar_sensors['lidar_0'] = {'device_heading': device_heading_dict,
                                'device_position': device_position,
                                'R_ref_to_waymo_lidar': R_ref_to_waymo_lidar,
                                'path': Path(lidar_out_path).relative_to(root_path).__str__()}
    info['lidar_sensors'] = lidar_sensors


    if remove_source:
        Path(frame_path).unlink()

    return info


def prepare_extracted_event_data(path, calibration):
    info_list = []
    topic_dict = {}
    root_path = path.parent.parent
    [topic_dict.update({k: []}) for k in calibration.keys()]
    topic_dict_lists = {}
    for topic, data in calibration.items():
        # filling camera info
        if data['type'] == 'camera':
            file_list = sorted(list((path / topic).glob('*.jpg')))
            topic_dict_lists[topic] = file_list
        # filling lidar info
        elif data['type'] == 'lidar':
            file_list = sorted(list((path / topic).glob('*.npy')))
            topic_dict_lists[topic] = file_list
        elif data['type'] == 'gnss':
            file_list = sorted(list((path / topic).glob('*.npy')))
            topic_dict_lists[topic] = file_list
        else:
            raise NotImplementedError

    timestamp_file_list = sorted(list((path / 'timestamps').glob('*.npy')))
    topic_dict_lists['timestamp'] = timestamp_file_list

    topic_path_list = np.stack((topic_dict_lists.values())).T

    for fnr, frame in tqdm(enumerate(topic_path_list), total=len(topic_path_list)):
        cameras = {}
        lidar_sensors = {}
        ego = {}
        file_names = [path.stem for path in frame]
        assert np.all([fn == file_names[0] for fn in file_names[1:]]), 'File names does not match'
        with frame[-1].open('rb') as fp:
            time_stamp_data = np.load(fp)

        info = {'timestamp_sec': time_stamp_data[0], 'timestamp_nano': time_stamp_data[1],
                'name': path.parent.stem, 'file_id': file_names[0]}
        for cid, (topic, data) in enumerate(calibration.items()):
            # TODO: add timestamp
            if data['type'] == 'camera':
                cameras[topic] = {'intrinsics': calibration[topic]['intrinsic'],
                                  'cam_to_ref': calibration[topic]['extrinsic'],
                                  'path': frame[cid].relative_to(root_path).__str__()}
            elif data['type'] == 'lidar':
                lidar_sensors[topic] = {'lidar_to_ref': calibration[topic]['extrinsic'],
                                        'path': frame[cid].relative_to(root_path).__str__()}
            else:
                with frame[cid].open('rb') as fp:
                    data = np.load(fp)
                if 'StGnss' in topic:
                    ego['enu_lat'], ego['enu_lon'], ego['enu_alt'] = data[0:3]
                    ego['lat'], ego['lon'], ego['alt'] = data[3:6]
                elif 'StImu' in topic:
                    ego['qx'], ego['qy'], ego['qz'], ego['qw'] = data[:4]
                    ego['aroll'], ego['apitch'], ego['ayaw'] = data[4:7]
                    ego['ax'], ego['ay'], ego['az'] = data[7:10]
                elif 'VGnss' in topic:
                    ego['vx'], ego['vy'], ego['vz'] = data[:3]
                    ego['vroll'], ego['vpitch'], ego['vyaw'] = data[3:6]

        info['cameras'] = cameras
        info['lidar_sensors'] = lidar_sensors
        info['ego'] = ego
        info_list.append(info)

    return info_list


def prepare_data_from_events(path, events, calibration):
    def update_topic(data):
        topic_dict[current_topic].append([time_stamp, *data])
    info_list = []
    topic_dict = {}
    root_path = path.parent.parent

    [topic_dict.update({k: []}) for k in calibration.keys()]
    topic_dict.update({'IMU': []})
    topic_dict.update({'GPS': []})
    main_topic = [k for k, v in calibration.items() if v['type'] == 'lidar' and v['index'] == 0][0]
    # find first lidar appearance
    index = events.index([event for event in events if main_topic in event][0])
    # n indices backward to have gps and imu data for the first entry
    #events = events[max(index, 0):]
    for event in events:
        event_entries = event.split()
        time_stamp = int(event_entries[0])
        event_type = event_entries[1]
        event_entries = event_entries[2:]
        if (event_type == 'CAM' or event_type == 'PCL') and len(event_entries) == 2:
            current_topic = event_entries[0]
            if len(topic_dict[current_topic]) > 0:
                # write info file based on timestamps
                if current_topic == main_topic:
                    # select last main topic
                    time_stamp_ref, frame_path = topic_dict[current_topic][0]
                    info = {'timestamp': time_stamp_ref, 'name': path.parent.stem, 'file_id': Path(frame_path).stem}
                    cameras = {}
                    lidar_sensors = {}
                    for topic, data in calibration.items():
                        if len(topic_dict[topic]) > 0:
                            # filling camera info
                            if data['type'] == 'camera':
                                idx = np.argmin(abs(np.array([e[0] for e in topic_dict[topic]]) - time_stamp_ref))
                                file_path = path / topic_dict[topic][idx][1]
                                cameras[topic] = {'intrinsics': calibration[topic]['intrinsic'],
                                                  'cam_to_ref': calibration[topic]['extrinsic'],
                                                  'path': Path(file_path).relative_to(root_path).__str__(),
                                                  'timestamp': topic_dict[topic][idx][0]}
                                topic_dict[topic] = topic_dict[topic][idx:]

                            # filling lidar info
                            elif data['type'] == 'lidar':
                                idx = np.argmin(abs(np.array([e[0] for e in topic_dict[topic]]) - time_stamp_ref))
                                file_path = path / topic_dict[topic][idx][1]
                                lidar_sensors[topic] = {'lidar_to_ref': calibration[topic]['extrinsic'],
                                                        'path': Path(file_path).relative_to(root_path).__str__(),
                                                        'timestamp': topic_dict[topic][idx][0]}
                                topic_dict[topic] = topic_dict[topic][idx:]
                            else:
                                raise NotImplementedError

                    gps = []
                    imu = []
                    if len(topic_dict['GPS']) > 0:
                        idx = np.argmin(abs(np.array([e[0] for e in topic_dict['GPS']]) - time_stamp_ref))
                        gps = topic_dict['GPS'][idx]
                        topic_dict['GPS'] = topic_dict['GPS'][idx:]
                    if len(topic_dict['IMU']) > 0:
                        idx = np.argmin(abs(np.array([e[0] for e in topic_dict['IMU']]) - time_stamp_ref))
                        imu = topic_dict['IMU'][idx]
                        topic_dict['IMU'] = topic_dict['IMU'][idx:]

                    info['cameras'] = cameras
                    info['lidar_sensors'] = lidar_sensors
                    if len(gps) > 0:
                        info['gps'] = gps
                    if len(imu) > 0:
                        info['imu'] = imu
                    topic_dict[current_topic] = []

                    info_list.append(info)
                # update current topic --> adds topic info to topic list
                update_topic([event_entries[1]])

            else:
                update_topic([event_entries[1]])
        elif event_type == 'GPS' and len(event_entries) == 6:
            current_topic = 'GPS'
            update_topic(event_entries)
        elif event_type == 'IMU' and len(event_entries) == 10:
            current_topic = 'IMU'
            update_topic(event_entries)
        else:
            raise NotImplementedError

    return info_list


def fill_avl_infos(path, sequence_list, count_inside_pts=True):
    avl_info_list = []
    if path.exists():
        process_bar = tqdm(sequence_list, desc='Fill infos')
        for seq_name in process_bar:
            process_bar.set_postfix_str(seq_name)
            unpacked_path = path / 'sequences' / seq_name / "unpacked"
            label_path = unpacked_path / 'labels.json'
            info_path = unpacked_path / 'info.pkl'
            event_path = unpacked_path / 'Event.dat'
            calibration_path = unpacked_path / 'calibration.yaml'
            metadata_path = unpacked_path / 'metadata.json'

            # load info/event file
            infos_dict = load_file(info_path, file_type='pickle')
            event_dat = load_file(event_path, file_type='dat')
            # load and organize lidar labels
            lidar_labels = load_file(label_path, file_type='json')
            # load metadata/calibration
            metadata = load_file(metadata_path, file_type='json')
            calibration_yaml = load_file(calibration_path, file_type='yaml')

            calibration = None
            if event_dat is not None:
                if metadata is not None:
                    calibration = load_calibration_from_metadata(metadata)
                elif calibration_yaml is not None:
                    calibration = load_calibration_from_yaml(calibration_yaml)

            #elif metadata is not None:
            #    calibration = load_calibration_from_metadata(metadata)
            #create label dict
            label_dict = {}
            if lidar_labels is not None:
                for label in lidar_labels['labels']:
                    file_id = Path(label['file_id']).stem
                    if file_id not in label_dict:
                        label_dict[file_id] = {'gt_boxes_lidar': [], 'name': [],
                                               'num_points_in_gt': [], 'gt_boxes_token': []}
                    if label['sensor_id'] == 'lidar' and label['label_type'] == '3d_bbox':
                        box = label['three_d_bbox']
                        bb = np.array([box['cx'], box['cy'], box['cz'], box['l'], box['w'], box['h'], box['rot_z']])
                        label_dict[file_id]['gt_boxes_lidar'].append(bb)
                        label_dict[file_id]['name'].append(label['label_category_id'])
                        label_dict[file_id]['gt_boxes_token'].append(label['label_id'])
                        if count_inside_pts:
                            frame_id = file_id.split("_")[-1]   #eg. 0001
                            lidar_path = unpacked_path / 'lidar' / ('%s.pkl' % frame_id)
                            points = load_file(lidar_path, file_type='pickle')
                            
                            corners_lidar = box_utils.boxes_to_corners_3d(np.expand_dims(bb, axis=0))[0]   #box_utils needs (N, 7) arraym [0] to flatten again
                            
                            flag = box_utils.in_hull(points[:, 0:3],
                                                    corners_lidar)
                            num_points_in_gt = flag.sum()
                            label_dict[file_id]['num_points_in_gt'].append(num_points_in_gt)
            
            
            # add labels to preprocessed infos (labelled data) and append to global info list
            if infos_dict is not None:
                for info_id, info in infos_dict.items():
                    #save point cloud data
                    save_info = {}
                    pc_info = {'num_features': 4, 
                               'lidar_idx': 'sequences/'+info["lidar_sensors"]["lidar_0"]["path"],
                               'timestamp': info["timestamp"],
                               'sequence_name': info["name"],
                               'file_id': info["file_id"],
                               'lidar_to_ref': info["lidar_sensors"]["lidar_0"]["lidar_to_ref"],
                               'world_to_lidar': info["lidar_sensors"]["lidar_0"]["world_to_lidar"]
                                }
                    save_info['point_cloud'] = pc_info
                    
                    #save label data
                    if info_id in label_dict:
                        labels = label_dict[info_id]
                        for k, v in labels.items():
                            labels[k] = np.array(v)
                        save_info["annos"] = labels
                    else:
                        save_info["annos"] = {'gt_boxes_lidar': np.array([]), 
                                     'name': np.array([]),
                                     'num_points_in_gt': np.array([]), 
                                     'gt_boxes_token': np.array([])}
                    avl_info_list.append(save_info)

            # prepare and add unlabelled data to global info list
            elif event_dat is not None:
                tmp_info_list = prepare_data_from_events(unpacked_path, event_dat, calibration)
                avl_info_list.extend(tmp_info_list)
            elif calibration is not None:
                tmp_info_list = prepare_extracted_event_data(unpacked_path, calibration)
                avl_info_list.extend(tmp_info_list)

    return avl_info_list
