import argparse
import torch
from datetime import datetime
from pathlib import Path
import pickle

from pcdet.config import cfg, cfg_from_yaml_file

from gace.gace_utils.gace_data import GACEDataset
from gace.gace_utils.gace_utils import GACELogger, train_gace_model, evaluate_gace_model

import os
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/gace_demo.yaml', help='demo config file')
    parser.add_argument('--base_detector_ckpt', type=str ,help='base detector model weights')
    parser.add_argument('--gace_ckpt', type=str, default=None,help='gace model weights')
    parser.add_argument('--batch_size_dg', type=int, default=8, help='batch size for data generation (model inference)')
    parser.add_argument('--batch_size_gace', type=int, default=2048, help='batch size for GACE training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--gace_data_folder', type=str, default='gace_data/', help='folder for generated train/val data and model')
    parser.add_argument('--gace_output_folder', type=str, default='gace_output/',help='folder for gace output')
    parser.add_argument('--results_path', type=str, required=True, default=None, help='path to detection results for pseudo label generation')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    os.chdir('/home/cgriesbacher/thesis/3DTrans')
    args, cfg = parse_config()
    
    args.gace_output_folder = Path(args.gace_output_folder) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.gace_output_folder.mkdir(parents=True, exist_ok=True)

    # XXX: Fix this ugly hack
    cfg.MODEL = cfg.MODEL.MODEL

    logger = GACELogger(args.gace_output_folder)
    logger.gace_info('Demo for Geometry Aware Confidence Enhancement (GACE)')
    logger.gace_info(f'Dataset:\t {cfg.DATA_CONFIG.DATASET}')
    logger.gace_info(f'Base-Detector:\t {cfg.MODEL.NAME}')
    logger.gace_info(f'Data Folder:\t {args.gace_data_folder}')
    logger.gace_info(f'Output Folder:\t {args.gace_output_folder}')
    
    #overwrite data path in dataset cfg by removing the initial ..
    cfg.DATA_CONFIG.DATA_PATH = cfg.DATA_CONFIG.DATA_PATH[3:]

    #load results (pickle file)
    logger.gace_info(f'Load results from {args.results_path}')
    with open(args.results_path, 'rb') as f:
        results = pickle.load(f)

    gace_dataset_val = GACEDataset(args, cfg, results, logger, train=False)
    
    logger.gace_info(f'Load GACE model from {args.gace_ckpt}')
    gace_model = torch.load(args.gace_ckpt)
    logger.gace_info(f'GACE model loaded from {args.gace_ckpt}')

    
    logger.gace_info('Start evaluation with new confidence scores')
    result_str, result_str_old = evaluate_gace_model(gace_model, gace_dataset_val, args, cfg, logger, eval_old=True)
    if result_str_old is not None:
        logger.gace_info('Evaluation Results without GACE:')
        logger.gace_info(result_str_old)
    logger.gace_info('Evaluation Results including GACE:')
    logger.gace_info(result_str)

    return


if __name__ == '__main__':
    main()

