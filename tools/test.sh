#!/bin/bash
#################TESTING#################
# Test a single detector on its training dataset

#CONFIG_FILE=kitti_models/second
#CONFIG_FILE=kitti_models/IA-SSD
DATASET=avlrooftop
MODEL=centerpoint
CFG_TAG=D1_100epochs_truck
EPOCH=100

NUM_GPUS=1
BATCHSIZE=4
NUM_WORKERS=4

#----------------------------------------------------
ROOT_PATH=/home/cgriesbacher/thesis/3DTrans
RUN_PATH=${DATASET}_models/$MODEL/$CFG_TAG
CHECKPOINT_PATH=$ROOT_PATH/output/$RUN_PATH/ckpt/checkpoint_epoch_$EPOCH.pth
CFG_PATH=$ROOT_PATH/output/$RUN_PATH/$MODEL.yaml


#single gpu training for debugging
#python test.py --cfg_file $CFG_PATH --ckpt $CHECKPOINT_PATH --batch_size 6 --workers 6 --extra_tag $CFG_TAG

#multi gpu training
cd "/home/cgriesbacher/thesis/3DTrans/tools"
bash scripts/dist_test.sh $NUM_GPUS --cfg_file $CFG_PATH --ckpt $CHECKPOINT_PATH --batch_size $BATCHSIZE --workers $NUM_WORKERS --extra_tag $CFG_TAG

