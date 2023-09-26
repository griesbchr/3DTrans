#!/bin/bash
#################TESTING#################

#CONFIG_FILE=kitti_models/second
#CONFIG_FILE=kitti_models/IA-SSD
DATASET=zod
MODEL=second
CFG_TAG=full_4epochs_normintensity
EPOCH=4

NUM_GPUS=2
BATCHSIZE=4
NUM_WORKERS=4

#----------------------------------------------------
ROOT_PATH=/home/cgriesbacher/thesis/3DTrans
RUN_PATH=${DATASET}_models/$MODEL/$CFG_TAG
CHECKPOINT_PATH=$ROOT_PATH/output/$RUN_PATH/ckpt/checkpoint_epoch_$EPOCH.pth
CFG_PATH=$ROOT_PATH/output/$RUN_PATH/$MODEL.yaml

#multi gpu training
cd "/home/cgriesbacher/thesis/3DTrans/tools"
bash scripts/dist_test.sh $NUM_GPUS --cfg_file $CFG_PATH --ckpt $CHECKPOINT_PATH --batch_size $BATCHSIZE --workers $NUM_WORKERS --extra_tag $CFG_TAG

#single gpu training for debugging
#python test.py --cfg_file $CFG_PATH --ckpt $CHECKPOINT_PATH --batch_size 6 --workers 6 --extra_tag $CFG_TAG