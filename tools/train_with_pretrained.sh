#################FINETUNING#################
# Finetune a single detector on an arbitrary dataset


DATASET=avltruck
MODEL=pvrcnnpp_nogtsampling
EXTRA_TAG=EmptyFrames_0epochs_ft_D16_50epochs_nogtsampling
PRETRAINED=/home/cgriesbacher/thesis/3DTrans/output/zod_models/pvrcnnpp/D16_50epochs/ckpt/checkpoint_epoch_50.pth

EPOCHS=0
CKPT_SAVE_INTERVAL=0
BATCHSIZE=4
WORKERS=4

NUM_GPUS=1
#export CUDA_VISIBLE_DEVICES=0,1

CONFIG_FILE=${DATASET}_models/$MODEL.yaml

SUBSAMPLEFACTOR=6

#single gpu training
python train.py --cfg_file cfgs/$CONFIG_FILE --extra_tag $EXTRA_TAG  --epochs $EPOCHS --ckpt_save_interval $CKPT_SAVE_INTERVAL --batch_size $BATCHSIZE --workers $WORKERS --pretrained_model $PRETRAINED --no_eval "True" --set DATA_CONFIG.SUBSAMPLEFACTOR $SUBSAMPLEFACTOR
#################CROSS DATASET TESTING#################
# Test a single detector on a single arbitrary dataset
OUTPUT_FOLDER=output

TRAIN_DATASET=avltruck
MODEL=pvrcnnpp_nogtsampling
EXTRA_TAG=EmptyFrames_0epochs_ft_D16_50epochs_nogtsampling
EPOCH=0

EVAL_DATASET=avltruck

BATCHSIZE=4
NUM_WORKERS=2


#-----------------------------------------------------
EVAL_DATASET_CFG_PATH=cfgs/dataset_configs/$EVAL_DATASET/OD/${EVAL_DATASET}_dataset.yaml

ROOT_PATH=/home/cgriesbacher/thesis/3DTrans
RUN_PATH=${TRAIN_DATASET}_models/$MODEL/$EXTRA_TAG
CHECKPOINT_PATH=$ROOT_PATH/$OUTPUT_FOLDER/$RUN_PATH/ckpt/checkpoint_epoch_$EPOCH.pth
CFG_PATH=$ROOT_PATH/$OUTPUT_FOLDER/$RUN_PATH/$MODEL.yaml

#multi gpu training
NUM_GPUS=1
cd "/home/cgriesbacher/thesis/3DTrans/tools"

bash scripts/dist_test.sh $NUM_GPUS --cfg_file $CFG_PATH --ckpt $CHECKPOINT_PATH --batch_size $BATCHSIZE --workers $NUM_WORKERS --extra_tag $EXTRA_TAG --crosseval_dataset_cfg $EVAL_DATASET_CFG_PATH --eval_tag $EVAL_DATASET

#single gpu training for debugging
#python test.py --cfg_file $CFG_PATH --ckpt $CHECKPOINT_PATH --batch_size $BATCHSIZE --workers $NUM_WORKERS --extra_tag $EXTRA_TAG --crosseval_dataset_cfg $EVAL_DATASET_CFG_PATH --eval_tag $EVAL_DATASET


