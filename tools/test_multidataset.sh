#!/bin/bash
#################MULTI DATASET TESTING#################
# Test a single detector on multiple arbitrary datasets

TRAIN_DATASET=(avltruck avlrooftop zod)

MODEL=pointpillar

CFG_TAG=(D6_50epochs D1_50epochs D16_50epochs)
EPOCH=50

EVAL_DATASETS=(avltruck avlrooftop zod)

BATCHSIZE=8
NUM_WORKERS=4


#-----------------------------------------------------

for i in "${!TRAIN_DATASET[@]}"; do
    TRAIN_DATASET=${TRAIN_DATASET[$i]}
    CFG_TAG=${CFG_TAG[$i]}

    for EVAL_DATASET in "${EVAL_DATASETS[@]}";
    do
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
    done
done


