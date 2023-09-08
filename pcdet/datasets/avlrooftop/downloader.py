import pickle
import json
import requests
import wget
import shutil

import numpy as np

from pathlib import Path
from tqdm import tqdm
from pyquaternion import Quaternion
from easydict import EasyDict

from .avl_utils import extract_frame

from .deepen_cfg import CLIENT_ID, REQUEST_DATASET_IDS, HEADER

def transform_matrix(translation: np.ndarray = np.array([0, 0, 0]),
                     rotation: Quaternion = Quaternion([1, 0, 0, 0]),
                     inverse: bool = False) -> np.ndarray:
    """
    Convert pose to transformation matrix.
    :param translation: <np.float32: 3>. Translation in x, y, z.
    :param rotation: Rotation in quaternions (w ri rj rk).
    :param inverse: Whether to compute inverse transform matrix.
    :return: <np.float32: 4, 4>. Transformation matrix.
    """
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


# def cart_to_hom(pts):
#     """
#     :param pts: (N, 3 or 2)
#     :return pts_hom: (N, 4 or 3)
#     """
#     pts_hom = np.concatenate((pts, np.ones((pts.shape[0], 1), dtype=np.float32)), axis=1)
#     return pts_hom
#
#
# def apply_transform(pts_lidar, transform):
#     pts_lidar_h = cart_to_hom(pts_lidar)
#     return np.einsum('ij,kj->ki', transform, pts_lidar_h)[..., :3]
#
#
# def world_to_gen(pts):
#     transform = np.array([
#         [1, 0, 0, 0],
#         [0, -1, 0, 0],
#         [0, 0, 1, 0],
#         [0, 0, 0, 1]])
#     pts_h = cart_to_hom(pts)
#     return np.einsum('ij,kj->ki', transform, pts_h)[..., :3]

def load_sequence_list(path):
    sequence_names = []
    if path.exists():
        with path.open(mode='r') as sf:
            sequence_names = sf.read().splitlines()
    return sequence_names


def web_request(url):
    result = requests.get(url=url, headers=HEADER)
    return result.json()


def dataset_url(dataset_id):
    url = 'https://tools.deepen.ai/api/v2/datasets/%s' % dataset_id
    url_dict = {
        'metadata': url,
        'files': url + '/files',
        'dataset':  url + '/uploaded_dataset_signed_media_link',
        'labels':   url + '/labels?filter_existing_categories=true&final=true'
    }
    return EasyDict(url_dict)


def download_and_extract_dataset(path, dataset_id, dataset_name, split, process_bar=None):
    # define paths
    unpacked_path = path / split / dataset_name / 'unpacked'
    unpacked_path.mkdir(exist_ok=True, parents=True)
    raw_download_path = path / split / dataset_name / 'raw'
    raw_download_path.mkdir(parents=True, exist_ok=True)

    file_name_archive = raw_download_path / 'json_data.zip'
    metadata_path = unpacked_path / 'metadata.json'
    label_path = unpacked_path / 'labels.json'
    info_path = unpacked_path / 'info.pkl'

    # skip if info is available
    if info_path.exists():
        return 0

    # fetch files of dataset
    url_dict =  dataset_url(dataset_id)
    files = web_request(url_dict.files)

    if not file_name_archive.exists():
        process_bar.set_description('Fetch data')
        ds_url = web_request(url_dict.dataset)['uploaded_dataset_signed_media_link']
        file_name_archive = wget.download(ds_url, out=str(raw_download_path))
    process_bar.set_description('Unzip data')
    shutil.unpack_archive(file_name_archive, raw_download_path)

    if not label_path.exists():
        process_bar.set_description('Fetch labels')
        labels = web_request(url_dict.labels)
        with label_path.open(mode='w') as fp:
            json.dump(labels, fp)

    if not metadata_path.exists():
        process_bar.set_description('Fetch metadata')
        metadata = web_request(url_dict.metadata)
        with metadata_path.open(mode='w') as fp:
            json.dump(metadata, fp)

    extracted_path_list = list(raw_download_path.glob('*'))
    extracted_file_list = [f.stem for f in extracted_path_list]

    info_dict = {}
    process_bar.set_description('Extract data')
    for file in files['files']:
        if 'lidar' in file['sensors']:
            source_path = extracted_path_list[extracted_file_list.index(Path(file['file_id']).stem)]
            frame_id = file['file_id'].split('_')[-1].split(".")[0]
            info = extract_frame(path / split, source_path, frame_id, dataset_name, unpacked_path)
            info_dict[info['file_id']] = info
        else:
            return 0

    with info_path.open(mode='wb') as fp:
        pickle.dump(info_dict, fp)

    if process_bar is not None:
        n_frames_total = int(process_bar.postfix.split('=')[-1])
        n_frames_total += len(info_dict.keys())
        process_bar.set_postfix({'frames_total': n_frames_total})
        process_bar.update()
    # print('Added %d items' % len(info_list))

    return len(info_dict.keys())


def download_datasets(path, sequence_list, split, n_workers=4):
    import concurrent.futures as futures

    dataset_id_list = web_request(REQUEST_DATASET_IDS)['datasets']
    print('Sequences: %d / %d' % (len(sequence_list), len(dataset_id_list)))

    pbar = tqdm(total=len(dataset_id_list))
    pbar.set_postfix({'frames_total': 0})

    def process_single_ds(pair):
        ds_id, ds_name = pair.values()
        if ds_name in sequence_list or len(sequence_list) == 0:
            return download_and_extract_dataset(path, ds_id, ds_name, split, process_bar=pbar)
        else:
            return 0

    with futures.ThreadPoolExecutor(n_workers) as executor:
        infos = list(executor.map(process_single_ds, dataset_id_list))
        print('Downloaded and extracted %d frames.' % sum(infos))

    return infos

#python -m pcdet.datasets.avlrooftop.downloader --root_path /data/AVLRooftop/ --sequence_file /data/AVLRooftop/training_sequences_small.txt --split sequences
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--root_path', type=str, default='/home/christian/big_data/data/avl/rooftopbox/',
                        help='defines the root_path')
    parser.add_argument('--sequence_file', type=str, default='training_sequences_10k.txt',
                        help='file of sequence names considered for download (in root folder)')
    parser.add_argument('--num_workers', type=int, default=5, help='number of parallel workers')
    parser.add_argument('--split', type=str, default='training', help='dataset split')
    args = parser.parse_args()

    root_path = Path(args.root_path)
    sequence_file_path = root_path / args.sequence_file
    # sequence list provided from AVL which contains sequences they use for training
    sequence_names = load_sequence_list(sequence_file_path)

    results = download_datasets(root_path,
                                sequence_names,
                                args.split,
                                args.num_workers)
    print('Done!')



