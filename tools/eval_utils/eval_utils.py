import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    ret_dict = {}

    #check if there is results.pkl in result_dir
    if (result_dir / 'result.pkl').exists():
        det_annos = pickle.load(open(result_dir / 'result.pkl', 'rb'))
    else:
        if cfg.LOCAL_RANK == 0:
            progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
        start_time = time.time()

        for i, batch_dict in enumerate(dataloader):
            load_data_to_gpu(batch_dict)
            with torch.no_grad():
                pred_dicts, ret_dict = model(batch_dict)
            disp_dict = {}

            statistics_info(cfg, ret_dict, metric, disp_dict)
            annos = dataset.generate_prediction_dicts(
                batch_dict, pred_dicts, class_names,
                output_path=final_output_dir if save_to_file else None
            )
            det_annos += annos
            if cfg.LOCAL_RANK == 0:
                progress_bar.set_postfix(disp_dict)
                progress_bar.update()

        if cfg.LOCAL_RANK == 0:
            progress_bar.close()

        if dist_test:
            rank, world_size = common_utils.get_dist_info()
            det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
            metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

        logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
        sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
        logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

        if cfg.LOCAL_RANK != 0:
            return {}

        if dist_test:
            for key, val in metric[0].items():
                for k in range(1, world_size):
                    metric[0][key] += metric[k][key]
            metric = metric[0]

        gt_num_cnt = metric['gt_num']
        for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
            cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
            cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
            logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
            logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
            ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
            ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

        with open(result_dir / 'result.pkl', 'wb') as f:
            pickle.dump(det_annos, f)

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    with open(result_dir / 'eval_results.pkl', 'wb') as f:
        pickle.dump(result_dict, f)

    logger.info(result_str)
    ret_dict.update(result_dict)

    plot_pr_curve(result_dict, result_dir, cfg)

    logger.info('****************Evaluation done.*****************')
    return ret_dict

def plot_pr_curve(result_dict, result_dir, cfg):
    import matplotlib.pyplot as plt
    import os
    import seaborn as sns

    #find class names
    class_names = list(set([key.split('_')[0] for key in result_dict if 'AP' in key]))

    #sort class names alphabetically inverse
    class_names.sort(reverse=True)

    #do a subplot for each class in sns for the precision recall curve
    #to two seperate plots side by side for level 1 and level 2
    #the dimentionality of the plot grid thus is 2 x len(class_names)q
    fig, axs = plt.subplots(len(class_names),2, figsize=(20, 10))
    fig.suptitle('Precision Recall for ' + cfg.TAG + "-" + cfg.EXP_GROUP_PATH + "-" + cfg.extra_tag + "@" + cfg.DATA_CONFIG.DATASET, fontsize=16)
    for i, class_name in enumerate(class_names):
        for j, level in enumerate(['1', '2']):
            #get all keys that contain the class name and level
            keys = [key for key in result_dict if class_name in key and level in key and "PR" in key]
            
            #get first value that contain the class name and level
            values = [result_dict[key] for key in keys][0]

            #get recall and precision values
            recalls = values[:,1]
            precisions = values[:,0]

            #plot recall vs. precision
            axs[i, j].plot(recalls, precisions)
            axs[i, j].set_title(class_name + ' Level ' + level)
            axs[i, j].set_xlabel('Recall')
            axs[i, j].set_ylabel('Precision')
            axs[i, j].set_ylim([0, 1.01])
            axs[i, j].set_xlim([0, 1.01])
            axs[i, j].grid(True)

            #todo plot confusion matrix for new waymo eval

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig_path = os.path.join(result_dir, 'pr_curve.png')
    plt.savefig(fig_path)
    print('Saved pr_curve.png to %s' % fig_path)
    plt.close(fig)   

def eval_one_epoch_parallel(cfg, model, show_db, dataloader_s1, dataloader_s2, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0
    if show_db == 1:
        dataset = dataloader_s1.dataset
        class_names = dataset.class_names
        det_annos = []
    elif show_db == 2:
        dataset = dataloader_s2.dataset
        class_names = dataset.class_names
        det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        if show_db == 1:
            progress_bar = tqdm.tqdm(total=len(dataloader_s1), leave=True, desc='eval', dynamic_ncols=True)
        elif show_db == 2:
            progress_bar = tqdm.tqdm(total=len(dataloader_s2), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()

    if show_db == 1:
        dataloader_iter_2 = iter(dataloader_s2)
        for i, batch_1 in enumerate(dataloader_s1):
            try:
                batch_2 = next(dataloader_iter_2)
            except StopIteration:
                dataloader_iter_2 = iter(dataloader_s2)
                batch_2 = next(dataloader_iter_2)

            batch_dict = common_utils.merge_two_batch_dict(batch_1, batch_2) 

            load_data_to_gpu(batch_dict)
            with torch.no_grad():
                pred_dicts, ret_dict, _, _ = model(batch_dict)
            disp_dict = {}

            statistics_info(cfg, ret_dict, metric, disp_dict)
            annos = dataset.generate_prediction_dicts(
                batch_dict, pred_dicts, class_names,
                output_path=final_output_dir if save_to_file else None
            )
            det_annos += annos
            if cfg.LOCAL_RANK == 0:
                progress_bar.set_postfix(disp_dict)
                progress_bar.update()
    elif show_db == 2:
        dataloader_iter_1 = iter(dataloader_s1)
        for i, batch_2 in enumerate(dataloader_s2):
            try:
                batch_1 = next(dataloader_iter_1)
            except StopIteration:
                dataloader_iter_1 = iter(dataloader_s1)
                batch_1 = next(dataloader_iter_1)

            batch_dict = common_utils.merge_two_batch_dict(batch_1, batch_2) 

            load_data_to_gpu(batch_dict)
            with torch.no_grad():
                _, _, pred_dicts, ret_dict = model(batch_dict)
            disp_dict = {}

            statistics_info(cfg, ret_dict, metric, disp_dict)
            annos = dataset.generate_prediction_dicts(
                batch_dict, pred_dicts, class_names,
                output_path=final_output_dir if save_to_file else None
            )
            det_annos += annos
            if cfg.LOCAL_RANK == 0:
                progress_bar.set_postfix(disp_dict)
                progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
