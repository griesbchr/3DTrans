import argparse
import torch
from datetime import datetime
from pathlib import Path

from pcdet.config import cfg, cfg_from_yaml_file

from gace.gace_utils.gace_data import GACEDataset
from gace.gace_utils.gace_utils import GACELogger, train_gace_model, evaluate_gace_model

import os
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/gace_demo.yaml', help='demo config file')
    parser.add_argument('--base_detector_ckpt', type=str, required=True,help='base detector model weights')
    parser.add_argument('--gace_ckpt', type=str, default=None,help='gace model weights')
    parser.add_argument('--batch_size_dg', type=int, default=8, help='batch size for data generation (model inference)')
    parser.add_argument('--batch_size_gace', type=int, default=2048, help='batch size for GACE training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--gace_data_folder', type=str, default='gace_data/', help='folder for generated train/val data and model')
    parser.add_argument('--gace_output_folder', type=str, default='gace_output/',help='folder for gace output')
    parser.add_argument('--results_path', type=str, default=None, help='path to detection results for pseudo label generation')
    parser.add_argument('--epochs', type=int, default=None, help='commandline epochs overwrite')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    #add command line arguments to cfg
    if args.epochs is not None:
        cfg.GACE.TRAIN.NUM_EPOCHS = args.epochs

    return args, cfg


def main():
    os.chdir('/home/cgriesbacher/thesis/3DTrans/gace')
    args, cfg = parse_config()
    
    if cfg.DATA_CONFIG_TAR is not None:
        filename = cfg.DATA_CONFIG.DATASET + "_" + cfg.DATA_CONFIG_TAR.DATASET + "_" + str(cfg.GACE.TRAIN.NUM_EPOCHS)
    else:
        filename = cfg.DATA_CONFIG.DATASET +  "_" + cfg.GACE.TRAIN.NUM_EPOCHS
    args.gace_output_folder = Path(args.gace_output_folder) / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + filename)
    args.gace_output_folder.mkdir(parents=True, exist_ok=True)

    cfg.MODEL = cfg.MODEL.MODEL

    logger = GACELogger(args.gace_output_folder)
    logger.gace_info('Demo for Geometry Aware Confidence Enhancement (GACE)')
    logger.gace_info(f'Dataset:\t {cfg.DATA_CONFIG.DATASET}')
    logger.gace_info(f'Base-Detector:\t {cfg.MODEL.NAME}')
    logger.gace_info(f'Data Folder:\t {args.gace_data_folder}')
    logger.gace_info(f'Output Folder:\t {args.gace_output_folder}')
    if args.gace_ckpt is not None:
        logger.gace_info(f'GACE model:\t {args.gace_ckpt}')
    if args.results_path is not None:
        logger.gace_info(f'Results:\t {args.results_path}')
    if cfg.DATA_CONFIG_TAR is not None:
        logger.gace_info(f'Target Dataset:\t {cfg.DATA_CONFIG_TAR.DATASET}')
        logger.gace_info(f'Target Data Path:\t {cfg.DATA_CONFIG_TAR.DATA_PATH}')
    logger.gace_info(f'Epochs:\t {cfg.GACE.TRAIN.NUM_EPOCHS}')
    #Load gace model
    if args.gace_ckpt is not None:
        args.gace_ckpt = "/home/cgriesbacher/thesis/3DTrans/gace/gace_output/2023-12-20_11-42-57/gace_model.pth"
        logger.gace_info(f'Load GACE model from {args.gace_ckpt}')
        gace_model = torch.load(args.gace_ckpt)
        logger.gace_info(f'GACE model loaded from {args.gace_ckpt}')
    else:
        #Train gace model
        gace_dataset_train = GACEDataset(args, cfg, logger, train=True)
        logger.gace_info('Start training confidence enhancement model')
        gace_model = train_gace_model(gace_dataset_train, args, cfg, logger)
        
        #store the trained model
        gace_model_path = args.gace_output_folder / 'gace_model.pth'
        torch.save(gace_model, gace_model_path)
        logger.gace_info(f'GACE model saved to {gace_model_path}')

    
    gace_dataset_val = GACEDataset(args, cfg, logger, train=False)
    
    
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


