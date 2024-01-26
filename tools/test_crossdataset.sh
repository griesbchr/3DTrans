#!/bin/bash
#################CROSS DATASET TESTING#################
# Test a single detector on a single arbitrary dataset

TRAIN_DATASET=avltruck
MODEL=pvrcnnpp_sn2zod
EXTRA_TAG=D6_10epochs_ft_D6_50epochs_ros
EPOCH=10

EVAL_DATASETS=(zod)

BATCHSIZE=4
NUM_WORKERS=2


#loop over eval datasets
for EVAL_DATASET in "${EVAL_DATASETS[@]}"
do
    #-----------------------------------------------------
    EVAL_DATASET_CFG_PATH=cfgs/dataset_configs/$EVAL_DATASET/DA/${EVAL_DATASET}_dataset.yaml

    ROOT_PATH=/home/cgriesbacher/thesis/3DTrans
    RUN_PATH=${TRAIN_DATASET}_models/$MODEL/$EXTRA_TAG
    CHECKPOINT_PATH=$ROOT_PATH/output/$RUN_PATH/ckpt/checkpoint_epoch_$EPOCH.pth
    CFG_PATH=$ROOT_PATH/output/$RUN_PATH/$MODEL.yaml

    #multi gpu training
    NUM_GPUS=1
    cd "/home/cgriesbacher/thesis/3DTrans/tools"
    bash scripts/dist_test.sh $NUM_GPUS --cfg_file $CFG_PATH --ckpt $CHECKPOINT_PATH --batch_size $BATCHSIZE --workers $NUM_WORKERS --extra_tag $EXTRA_TAG --crosseval_dataset_cfg $EVAL_DATASET_CFG_PATH --eval_tag $EVAL_DATASET

    #single gpu training for debugging
    #python test.py --cfg_file $CFG_PATH --ckpt $CHECKPOINT_PATH --batch_size $BATCHSIZE --workers $NUM_WORKERS --extra_tag $EXTRA_TAG --crosseval_dataset_cfg $EVAL_DATASET_CFG_PATH --eval_tag $EVAL_DATASET
done    