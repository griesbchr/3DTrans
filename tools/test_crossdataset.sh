#!/bin/bash
#################CROSS DATASET TESTING#################

TRAIN_DATASET=zod
MODEL=centerpoint
CFG_TAG=D16_100epochs_4classes
EPOCH=100

EVAL_DATASET=avlrooftop

BATCHSIZE=8
NUM_WORKERS=4


#-----------------------------------------------------
EVAL_DATASET_CFG_PATH=cfgs/dataset_configs/$EVAL_DATASET/OD/${EVAL_DATASET}_dataset.yaml

ROOT_PATH=/home/cgriesbacher/thesis/3DTrans
RUN_PATH=${TRAIN_DATASET}_models/$MODEL/$CFG_TAG
CHECKPOINT_PATH=$ROOT_PATH/output/$RUN_PATH/ckpt/checkpoint_epoch_$EPOCH.pth
CFG_PATH=$ROOT_PATH/output/$RUN_PATH/$MODEL.yaml

#multi gpu training
NUM_GPUS=1
cd "/home/cgriesbacher/thesis/3DTrans/tools"
bash scripts/dist_test.sh $NUM_GPUS --cfg_file $CFG_PATH --ckpt $CHECKPOINT_PATH --batch_size $BATCHSIZE --workers $NUM_WORKERS --extra_tag $CFG_TAG --crosseval_dataset_cfg $EVAL_DATASET_CFG_PATH --eval_tag $EVAL_DATASET

#single gpu training for debugging
#python test.py --cfg_file $CFG_PATH --ckpt $CHECKPOINT_PATH --batch_size $BATCHSIZE --workers $NUM_WORKERS --extra_tag $CFG_TAG --crosseval_dataset_cfg $EVAL_DATASET_CFG_PATH --eval_tag $EVAL_DATASET