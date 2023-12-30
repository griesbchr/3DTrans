import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler

from pcdet.utils import common_utils
from pcdet.utils.loss_utils import SigmoidFocalClassificationLoss
from gace.gace_utils.gace_model import GACEModel

import copy


def train_gace_model(cfg, dataset_train, logger, batch_size_gace=2048, num_workers=4):
    logger.info(f'Train GACE model for {cfg.SELF_TRAIN.GACE.TRAIN.NUM_EPOCHS} epochs')

    ip_dim = dataset_train.get_ip_dim()
    cp_dim = dataset_train.get_cp_dim()
    target_dim = dataset_train.get_target_dim()
    gace_model = GACEModel(cfg.SELF_TRAIN, ip_dim, cp_dim, target_dim)
    gace_model.cuda()
    gace_model.train()

    sampler_train = BatchSampler(
        RandomSampler(dataset_train), 
        batch_size=batch_size_gace, 
        drop_last=False)

    def my_collate(batch):
        return batch[0]

    dataloader_train = DataLoader(dataset_train, sampler=sampler_train,
                                  collate_fn=my_collate, num_workers=num_workers)
    
    iou_l1_loss = torch.nn.L1Loss()
    sigmoid_focal_loss = SigmoidFocalClassificationLoss(gamma=cfg.SELF_TRAIN.GACE.TRAIN.SFL_GAMMA, 
                                                        alpha=cfg.SELF_TRAIN.GACE.TRAIN.SFL_ALPHA)

    optimizer = torch.optim.Adam(gace_model.parameters(), lr=cfg.SELF_TRAIN.GACE.TRAIN.LR)

    total_iterations = len(dataloader_train) * cfg.SELF_TRAIN.GACE.TRAIN.NUM_EPOCHS

    p_bar_desc = f'[GACE] Training ({cfg.SELF_TRAIN.GACE.TRAIN.NUM_EPOCHS}) Epochs)'
    progress_bar = tqdm(total=total_iterations, desc=p_bar_desc, leave=True, 
                        dynamic_ncols=True)

    for epoch_count in range(cfg.SELF_TRAIN.GACE.TRAIN.NUM_EPOCHS):
        for ip_data, cp_data, nb_ip_data, cat, iou in dataloader_train:
            ip_data = ip_data.cuda()
            cp_data = cp_data.cuda()
            nb_ip_data = nb_ip_data.cuda()
            cat = cat.cuda()
            iou = iou.cuda()

            # forward instance specific features of neighbors first
            f_n_I = gace_model.H_I(nb_ip_data)
                
            # forward instance specific features of current detection
            f_I = gace_model.H_I(ip_data)
                
            # forward context features
            f_n_C = gace_model.H_C(torch.cat([cp_data, f_n_I.detach()], dim=1))

            # merge instance-specific and context feature vectors
            gace_output = gace_model.H_F(torch.cat([f_I, f_n_C], dim=1))

            sf_loss = sigmoid_focal_loss(gace_output[:, 0, ...], cat[:, None, None], 
                                         torch.ones_like(cat)[:, None, None])
            sf_loss = sf_loss.mean()
                                            
            iou_loss = iou_l1_loss(torch.sigmoid(gace_output[:, 1, ...]).flatten(), iou)

            loss = sf_loss + cfg.SELF_TRAIN.GACE.TRAIN.IOU_LOSS_W * iou_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
            progress_bar.update()
        
    progress_bar.close()

    return gace_model
