#################TRAINING#################


TRAIN_DATASET=avlrooftop
MODEL=pvrcnnpp
EXTRA_TAG=D10_1epochs
EPOCHS=1
SUBSAMPLE=10
CKPT_SAVE_INTERVAL=1
BATCHSIZE=4
WORKERS=4

CONFIG_FILE=${TRAIN_DATASET}_models/$MODEL.yaml


#single gpu training
python train.py --cfg_file cfgs/$CONFIG_FILE --extra_tag $EXTRA_TAG  --epochs $EPOCHS --subsample $SUBSAMPLE --ckpt_save_interval $CKPT_SAVE_INTERVAL --batch_size $BATCHSIZE --workers $WORKERS --no_eval "True"

#multi gpu training
#bash scripts/dist_train.sh $NUM_GPUS --cfg_file cfgs/$CONFIG_FILE --extra_tag $EXTRA_TAG  --epochs $EPOCHS --subsample $SUBSAMPLE --ckpt_save_interval $CKPT_SAVE_INTERVAL --batch_size $BATCHSIZE

#multi dataset eval
NUM_GPUS=1
EVAL_BATCHSIZE=4
EVAL_WORKERS=4


EVAL_DATASETS=(avltruck
              avlrooftop
              zod)



# Check if the epochs is divisible by the ckpt_save_interval to ensure the latest checkpoint is saved.
if [[ $((EPOCHS % CKPT_SAVE_INTERVAL)) -ne 0 ]]; then
    echo "$EPOCHS is not divisible by $CKPT_SAVE_INTERVAL."
    echo "The latest checkpoint would not be saved!"
    exit 1
fi


#-----------------------------------------------------
for EVAL_DATASET in "${EVAL_DATASETS[@]}";
do
    EVAL_DATASET_CFG_PATH=cfgs/dataset_configs/$EVAL_DATASET/OD/${EVAL_DATASET}_dataset.yaml

    ROOT_PATH=/home/cgriesbacher/thesis/3DTrans
    RUN_PATH=${TRAIN_DATASET}_models/$MODEL/$EXTRA_TAG
    CHECKPOINT_PATH=$ROOT_PATH/output/$RUN_PATH/ckpt/checkpoint_epoch_$EPOCHS.pth
    CFG_PATH=$ROOT_PATH/output/$RUN_PATH/$MODEL.yaml

    #multi gpu training
    NUM_GPUS=1
    cd "/home/cgriesbacher/thesis/3DTrans/tools"
    bash scripts/dist_test.sh $NUM_GPUS --cfg_file $CFG_PATH --ckpt $CHECKPOINT_PATH --batch_size $EVAL_BATCHSIZE --workers $EVAL_WORKERS --extra_tag $EXTRA_TAG --crosseval_dataset_cfg $EVAL_DATASET_CFG_PATH --eval_tag $EVAL_DATASET

    #single gpu training for debugging
    #python test.py --cfg_file $CFG_PATH --ckpt $CHECKPOINT_PATH --batch_size $EVAL_BATCHSIZE --workers $EVAL_WORKERS --extra_tag $EXTRA_TAG --crosseval_dataset_cfg $EVAL_DATASET_CFG_PATH --eval_tag $EVAL_DATASET
done