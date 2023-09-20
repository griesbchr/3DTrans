#!/bin/bash
#################CROSS DATASET TESTING#################

TRAIN_DATASET=zod
MODEL=second
CFG_TAG=full_2epochs_trunc
EPOCH=2

EVAL_DATASET=avlrooftop

NUM_GPUS=2
BATCHSIZE=4
NUM_WORKERS=8


#-----------------------------------------------------
EVAL_DATASET_CFG_PATH=cfgs/dataset_configs/$EVAL_DATASET/OD/${EVAL_DATASET}_dataset.yaml

ROOT_PATH=/home/cgriesbacher/thesis/3DTrans
RUN_PATH=${TRAIN_DATASET}_models/$MODEL/$CFG_TAG
CHECKPOINT_PATH=$ROOT_PATH/output/$RUN_PATH/ckpt/checkpoint_epoch_$EPOCH.pth
CFG_PATH=$ROOT_PATH/output/$RUN_PATH/$MODEL.yaml

#multi gpu training
cd "/home/cgriesbacher/thesis/3DTrans/tools"
bash scripts/dist_test.sh $NUM_GPUS --cfg_file $CFG_PATH --ckpt $CHECKPOINT_PATH --batch_size $BATCHSIZE --workers $NUM_WORKERS --extra_tag $CFG_TAG --crosseval_dataset_cfg $EVAL_DATASET_CFG_PATH

#single gpu training for debugging
#python test.py --cfg_file $CFG_PATH --ckpt $CHECKPOINT_PATH --batch_size 6 --workers 6 --extra_tag $CFG_TAG