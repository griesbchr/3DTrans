#################FINETUNING#################
# Finetune a single detector on an arbitrary dataset with self-training

DATASET=zod
MODEL=pvrcnnpp_ST++truck
EXTRA_TAG=D16_30epochs_STzod_ft_D6_50epochs_2
PRETRAINED=/home/cgriesbacher/thesis/3DTrans/output/zod_models/pvrcnnpp_ros/D16_50epochs/ckpt/checkpoint_epoch_50.pth

EPOCHS=30
CKPT_SAVE_INTERVAL=1
BATCHSIZE=4
WORKERS=2

CONFIG_FILE=${DATASET}_models/$MODEL.yaml


#single gpu training
python train_st.py --cfg_file cfgs/$CONFIG_FILE --extra_tag $EXTRA_TAG  --epochs $EPOCHS --ckpt_save_interval $CKPT_SAVE_INTERVAL --batch_size $BATCHSIZE --workers $WORKERS --pretrained_model $PRETRAINED

#multi gpu training
#NUM_GPUS=1
#bash scripts/dist_train.sh $NUM_GPUS --cfg_file cfgs/$CONFIG_FILE --extra_tag $EXTRA_TAG  --epochs $EPOCHS --ckpt_save_interval $CKPT_SAVE_INTERVAL --batch_size $BATCHSIZE --workers $WORKERS




#################CROSS DATASET TESTING#################
# Test a single detector on a single arbitrary dataset
OUTPUT_FOLDER=output

TRAIN_DATASET=zod
MODEL=pvrcnnpp_ST++truck
EXTRA_TAG=D16_30epochs_STzod_ft_D6_50epochs_2
EPOCHS=(1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 22 24 26 28 30)

EVAL_DATASET=avltruck

BATCHSIZE=4
NUM_WORKERS=2


#loop over eval datasets
for EPOCH in "${EPOCHS[@]}"
do
    #-----------------------------------------------------
    EVAL_DATASET_CFG_PATH=cfgs/dataset_configs/$EVAL_DATASET/DA/${EVAL_DATASET}_dataset.yaml

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
done    







#################FINETUNING#################
# Finetune a single detector on an arbitrary dataset with self-training

DATASET=zod
MODEL=pvrcnnpp_ST++truck_1
EXTRA_TAG=D16_30epochs_STzod_ft_D6_50epochs_3
PRETRAINED=/home/cgriesbacher/thesis/3DTrans/output/zod_models/pvrcnnpp_ros/D16_50epochs/ckpt/checkpoint_epoch_50.pth

EPOCHS=30
CKPT_SAVE_INTERVAL=1
BATCHSIZE=4
WORKERS=2

CONFIG_FILE=${DATASET}_models/$MODEL.yaml


#single gpu training
python train_st.py --cfg_file cfgs/$CONFIG_FILE --extra_tag $EXTRA_TAG  --epochs $EPOCHS --ckpt_save_interval $CKPT_SAVE_INTERVAL --batch_size $BATCHSIZE --workers $WORKERS --pretrained_model $PRETRAINED

#multi gpu training
#NUM_GPUS=1
#bash scripts/dist_train.sh $NUM_GPUS --cfg_file cfgs/$CONFIG_FILE --extra_tag $EXTRA_TAG  --epochs $EPOCHS --ckpt_save_interval $CKPT_SAVE_INTERVAL --batch_size $BATCHSIZE --workers $WORKERS








#################CROSS DATASET TESTING#################
# Test a single detector on a single arbitrary dataset
OUTPUT_FOLDER=output

TRAIN_DATASET=zod
MODEL=pvrcnnpp_ST++truck_1
EXTRA_TAG=D16_30epochs_STzod_ft_D6_50epochs_3
EPOCHS=(1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 22 24 26 28 30)

EVAL_DATASET=avltruck

BATCHSIZE=4
NUM_WORKERS=2


#loop over eval datasets
for EPOCH in "${EPOCHS[@]}"
do
    #-----------------------------------------------------
    EVAL_DATASET_CFG_PATH=cfgs/dataset_configs/$EVAL_DATASET/DA/${EVAL_DATASET}_dataset.yaml

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
done    

