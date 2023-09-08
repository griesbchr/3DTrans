#!/bin/bash
#################TESTING#################

#CONFIG_FILE=kitti_models/second
#CONFIG_FILE=kitti_models/IA-SSD
DATASET=avl_rooftop
MODEL=second
CFG_TAG=full_2epochs
EPOCH=2

ROOT_PATH=/home/cgriesbacher/thesis/3DTrans
CONFIG_FILE=${DATASET}_models/$MODEL/$CFG_TAG
CHECKPOINT_PATH=$ROOT_PATH/output/$CONFIG_FILE/ckpt/checkpoint_epoch_$EPOCH.pth
CFG_PATH=$ROOT_PATH/output/$CONFIG_FILE/$MODEL.yaml

#multi gpu training
cd "/home/cgriesbacher/thesis/3DTrans/tools"
bash scripts/dist_test.sh 2 --cfg_file $CFG_PATH --ckpt $CHECKPOINT_PATH --batch_size 6 --workers 6 --extra_tag $CFG_TAG

#single gpu training for debugging
#python test.py --cfg_file $CFG_PATH --ckpt $CHECKPOINT_PATH --batch_size 6 --workers 6 --extra_tag $CFG_TAG