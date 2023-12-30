import torch
import os
import glob
import tqdm
import numpy as np
import torch.distributed as dist
from pcdet.config import cfg
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils, commu_utils, memory_ensemble_utils
import pickle as pkl
import re
from copy import deepcopy

from gace.gace_utils.gace_dataloader import GACEDataset as GACEEEvalDataset
from gace.gace_utils.gace_dataloader_train import GACEDataset as GACETrainDataset
from gace.gace_utils.gace_utils_train import train_gace_model



#PSEUDO_LABELS = {}
from multiprocessing import Manager

PSEUDO_LABELS = Manager().dict() #for multiple GPU training
NEW_PSEUDO_LABELS = {}


def check_already_exsit_pseudo_label(ps_label_dir, start_epoch):
    """
    if we continue training, use this to directly
    load pseudo labels from exsiting result pkl

    if exsit, load latest result pkl to PSEUDO LABEL
    otherwise, return false and

    Args:
        ps_label_dir: dir to save pseudo label results pkls.
        start_epoch: start epoc
    Returns:

    """
    # support init ps_label given by cfg
    if start_epoch == 0 and cfg.SELF_TRAIN.get('INIT_PS', None):
        if os.path.exists(cfg.SELF_TRAIN.INIT_PS):
            print ("********LOADING PS FROM:", cfg.SELF_TRAIN.INIT_PS)
            init_ps_label = pkl.load(open(cfg.SELF_TRAIN.INIT_PS, 'rb'))
            PSEUDO_LABELS.update(init_ps_label)

            if cfg.LOCAL_RANK == 0:
                ps_path = os.path.join(ps_label_dir, "ps_label_e0.pkl")
                with open(ps_path, 'wb') as f:
                    pkl.dump(PSEUDO_LABELS, f)

            return cfg.SELF_TRAIN.INIT_PS

    ps_label_list = glob.glob(os.path.join(ps_label_dir, 'ps_label_e*.pkl'))
    if len(ps_label_list) == 0:
        return

    ps_label_list.sort(key=os.path.getmtime, reverse=True)
    for cur_pkl in ps_label_list:
        num_epoch = re.findall('ps_label_e(.*).pkl', cur_pkl)
        assert len(num_epoch) == 1

        # load pseudo label and return
        if int(num_epoch[0]) <= start_epoch:
            latest_ps_label = pkl.load(open(cur_pkl, 'rb'))
            PSEUDO_LABELS.update(latest_ps_label)
            return cur_pkl

    return None

def print_pseudo_label_stats(model, val_loader, rank, leave_pbar, ps_label_dir, cur_epoch, source_loader):
    pass   

def store_det_annos(det_annos, det_annos_gace):
    global det_annos_gobal
    global det_annos_gace_global
    
    #init global var det_annos and det_annos_gace if not initialized
    if not("det_annos_gobal" in globals()):
        det_annos_gobal = []
        det_annos_gace_global = []

    #check if each dict entry is cpu, if not, convert to cpu
    for i in range(len(det_annos)):
        for key in det_annos[i].keys():
            if isinstance(det_annos[i][key], torch.Tensor):
                det_annos[i][key] = det_annos[i][key].cpu().numpy()
    for i in range(len(det_annos_gace)):
        for key in det_annos_gace[i].keys():
            if isinstance(det_annos_gace[i][key], torch.Tensor):
                det_annos_gace[i][key] = det_annos_gace[i][key].cpu().numpy()

    #store det_annos and det_annos_gace
    det_annos_gobal += det_annos
    det_annos_gace_global += det_annos_gace

    return

def save_det_annos(path, cur_epoch):

    global det_annos_gobal
    global det_annos_gace_global

    #save det_annos and det_annos_gace to path
    path_no_gace = os.path.join(path, "det_annos_e{}.pkl".format(cur_epoch))
    path_gace = os.path.join(path, "det_annos_gace_e{}.pkl".format(cur_epoch))
    with open(path_no_gace, 'wb') as f:
        pkl.dump(det_annos_gobal, f)
    with open(path_gace, 'wb') as f:
        pkl.dump(det_annos_gace_global, f)
    
    #reset det_annos and det_annos_gace
    det_annos_gobal = []
    det_annos_gace_global = []
    return


def save_pseudo_label_epoch(model, source_loader, val_loader, rank, leave_pbar, ps_label_dir, cur_epoch, logger=None):
    """
    Generate pseudo label with given model.

    Args:
        model: model to predict result for pseudo label
        val_loader: data_loader to predict pseudo label
        rank: process rank
        leave_pbar: tqdm bar controller
        ps_label_dir: dir to save pseudo label
        cur_epoch
    """
    val_dataloader_iter = iter(val_loader)
    total_it_each_epoch = len(val_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                         desc='generate_ps_e%d' % cur_epoch, dynamic_ncols=True)

    pos_ps_meter = common_utils.AverageMeter()
    ign_ps_meter = common_utils.AverageMeter()

    # Since the model is eval status, some object-level data augmentation methods such as 
    # 'random_object_rotation', 'random_object_scaling', 'normalize_object_size' are not used 
    model.eval()

    # init gace
    if cfg.SELF_TRAIN.get('GACE', None) and cfg.SELF_TRAIN.GACE.ENABLED:
        logger.info('GACE enabled')

        if not getattr(cfg.SELF_TRAIN.GACE, 'CKPT', None):
            logger.info('training GACE model')

            #init gace dataloader and preprocess data
            prev_nms_thresh = model.model_cfg.POST_PROCESSING.NMS_CONFIG.NMS_THRESH
            model.model_cfg.POST_PROCESSING.NMS_CONFIG.NMS_THRESH = 0.1

            gace_train_dataset = GACETrainDataset(cfg, model, source_loader, logger, train=True)

            model.model_cfg.POST_PROCESSING.NMS_CONFIG.NMS_THRESH = prev_nms_thresh

            #train gace model
            batch_size_gace = getattr(cfg.SELF_TRAIN.GACE, 'BATCH_SIZE', None)
            num_workers_gace = getattr(cfg.SELF_TRAIN.GACE, 'NUM_WORKERS', None)
            gace_model = train_gace_model(cfg, gace_train_dataset, logger, batch_size_gace=batch_size_gace, num_workers=num_workers_gace)

        else: 
            # load gace model
            #check if ckpt exists
            assert os.path.exists(cfg.SELF_TRAIN.GACE.CKPT), "GACE ckpt not found"
            gace_model = torch.load(cfg.SELF_TRAIN.GACE.CKPT)
            logger.info(f'GACE model loaded from {cfg.SELF_TRAIN.GACE.CKPT}')
        gace_model.cuda()   
        gace_model.eval()

        #init gace dataset
        gace_eval_dataset = GACEEEvalDataset(cfg, logger, train=False)    

        #init flag to store detections
        store_detections = cfg.SELF_TRAIN.GACE.STORE_DETECTIONS 

    for cur_it in range(total_it_each_epoch):
        try:
            target_batch = next(val_dataloader_iter)
        except StopIteration:
            target_dataloader_iter = iter(val_loader)
            target_batch = next(target_dataloader_iter)

        # generate gt_boxes for target_batch and update model weights
        with torch.no_grad():
            load_data_to_gpu(target_batch)
            pred_dicts, ret_dict = model(target_batch)

        # add gace score correction
        if cfg.SELF_TRAIN.get('GACE', None) and cfg.SELF_TRAIN.GACE.ENABLED:
            
            ip_data, cp_data, nb_ip_data, cat, iou = gace_eval_dataset.process_data_batch(target_batch, pred_dicts)

            # predict updated scores
            with torch.no_grad():
                f_n_I = gace_model.H_I(nb_ip_data)
                f_I = gace_model.H_I(ip_data)
                f_n_C = gace_model.H_C(torch.cat([cp_data, f_n_I.detach()], dim=1))
                gace_output = gace_model.H_F(torch.cat([f_I, f_n_C], dim=1))
                    
                scores = torch.sigmoid(gace_output[:, 0, ...]).flatten()
            
            if store_detections: 
                pred_dicts_old = deepcopy(pred_dicts)

            # update pseudo label
            det_count = 0
            for b_idx in range(len(pred_dicts)):
                pred_dicts[b_idx]['pred_scores'] = scores[det_count:det_count+pred_dicts[b_idx]['pred_scores'].shape[0]]
                det_count += pred_dicts[b_idx]['pred_scores'].shape[0]

            if store_detections:
                det_annos = val_loader.dataset.generate_prediction_dicts(deepcopy(target_batch), deepcopy(pred_dicts_old), cfg.CLASS_NAMES)
                det_annos_gace = val_loader.dataset.generate_prediction_dicts(deepcopy(target_batch), deepcopy(pred_dicts), cfg.CLASS_NAMES)
                store_det_annos(det_annos, det_annos_gace)

        pos_ps_batch, ign_ps_batch = save_pseudo_label_batch(
            target_batch, pred_dicts=pred_dicts,
            need_update=(cfg.SELF_TRAIN.get('MEMORY_ENSEMBLE', None) and
                         cfg.SELF_TRAIN.MEMORY_ENSEMBLE.ENABLED and
                         cur_epoch > 0)
        )


        # log to console and tensorboard
        pos_ps_meter.update(pos_ps_batch)
        ign_ps_meter.update(ign_ps_batch)
        disp_dict = {'pos_ps_box': "{:.3f}({:.3f})".format(pos_ps_meter.val, pos_ps_meter.avg),
                     'ign_ps_box': "{:.3f}({:.3f})".format(ign_ps_meter.val, ign_ps_meter.avg)}

        if rank == 0:
            pbar.update()
            pbar.set_postfix(disp_dict)
            pbar.refresh()

        #if cur_it == 1:
        #    break

    if rank == 0:
        pbar.close()

    if cfg.SELF_TRAIN.get('GACE', None) and cfg.SELF_TRAIN.GACE.ENABLED and cfg.SELF_TRAIN.GACE.STORE_DETECTIONS:
        save_det_annos(ps_label_dir, cur_epoch)

    gather_and_dump_pseudo_label_result(rank, ps_label_dir, cur_epoch)
    print(len(PSEUDO_LABELS))


def gather_and_dump_pseudo_label_result(rank, ps_label_dir, cur_epoch):
    commu_utils.synchronize()

    if dist.is_initialized():
        part_pseudo_labels_list = commu_utils.all_gather(NEW_PSEUDO_LABELS)

        new_pseudo_label_dict = {}
        for pseudo_labels in part_pseudo_labels_list:
            new_pseudo_label_dict.update(pseudo_labels)

        NEW_PSEUDO_LABELS.update(new_pseudo_label_dict)

    # dump new pseudo label to given dir
    if rank == 0:
        ps_path = os.path.join(ps_label_dir, "ps_label_e{}.pkl".format(cur_epoch))
        with open(ps_path, 'wb') as f:
            pkl.dump(NEW_PSEUDO_LABELS, f)

    commu_utils.synchronize()
    PSEUDO_LABELS.clear()
    PSEUDO_LABELS.update(NEW_PSEUDO_LABELS)
    NEW_PSEUDO_LABELS.clear()


def save_pseudo_label_batch(input_dict,
                            pred_dicts=None,
                            need_update=True):
    """
    Save pseudo label for give batch.
    If model is given, use model to inference pred_dicts,
    otherwise, directly use given pred_dicts.

    Args:
        input_dict: batch data read from dataloader
        pred_dicts: Dict if not given model.
            predict results to be generated pseudo label and saved
        need_update: Bool.
            If set to true, use consistency matching to update pseudo label
    """
    pos_ps_meter = common_utils.AverageMeter()
    ign_ps_meter = common_utils.AverageMeter()

    batch_size = len(pred_dicts)
    for b_idx in range(batch_size):
        pred_cls_scores = pred_iou_scores = None
        if 'pred_boxes' in pred_dicts[b_idx]:
            # Exist predicted boxes passing self-training score threshold
            pred_boxes = pred_dicts[b_idx]['pred_boxes'].detach().cpu().numpy()
            pred_labels = pred_dicts[b_idx]['pred_labels'].detach().cpu().numpy()
            pred_scores = pred_dicts[b_idx]['pred_scores'].detach().cpu().numpy()
            if 'pred_cls_scores' in pred_dicts[b_idx]:
                pred_cls_scores = pred_dicts[b_idx]['pred_cls_scores'].detach().cpu().numpy()
            if 'pred_iou_scores' in pred_dicts[b_idx]:
                pred_iou_scores = pred_dicts[b_idx]['pred_iou_scores'].detach().cpu().numpy()

            # remove boxes under negative threshold
            if cfg.SELF_TRAIN.get('NEG_THRESH', None):
                labels_remove_scores = np.array(cfg.SELF_TRAIN.NEG_THRESH)[pred_labels - 1]
                remain_mask = pred_scores >= labels_remove_scores
                pred_labels = pred_labels[remain_mask]
                pred_scores = pred_scores[remain_mask]
                pred_boxes = pred_boxes[remain_mask]
                if 'pred_cls_scores' in pred_dicts[b_idx]:
                    pred_cls_scores = pred_cls_scores[remain_mask]
                if 'pred_iou_scores' in pred_dicts[b_idx]:
                    pred_iou_scores = pred_iou_scores[remain_mask]

            labels_ignore_scores = np.array(cfg.SELF_TRAIN.SCORE_THRESH)[pred_labels - 1]
            ignore_mask = pred_scores < labels_ignore_scores
            pred_labels[ignore_mask] = -1

            gt_box = np.concatenate((pred_boxes,
                                     pred_labels.reshape(-1, 1),
                                     pred_scores.reshape(-1, 1)), axis=1)

        else:
            # no predicted boxes passes self-training score threshold
            gt_box = np.zeros((0, 9), dtype=np.float32)

        gt_infos = {
            'gt_boxes': gt_box,
            'cls_scores': pred_cls_scores,
            'iou_scores': pred_iou_scores,
            'memory_counter': np.zeros(gt_box.shape[0])
        }

        # record pseudo label to pseudo label dict
        if need_update:
            ensemble_func = getattr(memory_ensemble_utils, cfg.SELF_TRAIN.MEMORY_ENSEMBLE.NAME)
            gt_infos = ensemble_func(PSEUDO_LABELS[input_dict['frame_id'][b_idx]],
                                     gt_infos, cfg.SELF_TRAIN.MEMORY_ENSEMBLE)

        if gt_infos['gt_boxes'].shape[0] > 0:
            ign_ps_meter.update((gt_infos['gt_boxes'][:, 7] < 0).sum())
        else:
            ign_ps_meter.update(0)
        pos_ps_meter.update(gt_infos['gt_boxes'].shape[0] - ign_ps_meter.val)

        NEW_PSEUDO_LABELS[input_dict['frame_id'][b_idx]] = gt_infos

    return pos_ps_meter.avg, ign_ps_meter.avg


def load_ps_label(frame_id):
    """
    :param frame_id: file name of pseudo label
    :return gt_box: loaded gt boxes (N, 9) [x, y, z, w, l, h, ry, label, scores]
    """
    if frame_id in PSEUDO_LABELS:
        gt_box = PSEUDO_LABELS[frame_id]['gt_boxes']
    else:
        raise ValueError('Cannot find pseudo label for frame: %s' % frame_id)

    return gt_box
