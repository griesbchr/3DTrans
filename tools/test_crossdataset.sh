#################CROSS DATASET TESTING#################
# Test a single detector on a single arbitrary dataset
OUTPUT_FOLDER=output

TRAIN_DATASET=zod
MODEL=pvrcnnpp_STrooftop
EXTRA_TAG=D1_10epochs_STrooftop_ft_D16_50epochs_ros
EPOCHS=(1 2 3 4 5 6 7 8 9 10)

EVAL_DATASET=avlrooftop
EVAL_DATASET_EXTRA_TAG=""

BATCHSIZE=4
NUM_WORKERS=2


#loop over eval datasets
#for i in "${!TRAIN_DATASETS[@]}"; do
#    TRAIN_DATASET=${TRAIN_DATASETS[$i]}
#    EXTRA_TAG=${EXTRA_TAG[$i]}

for i in "${!EPOCHS[@]}"; do
    EPOCH=${EPOCHS[$i]}
    #MODEL=${MODELS[$i]}
    #TRAIN_DATASET=${TRAIN_DATASETS[$i]}
    #EXTRA_TAG=${EXTRA_TAGS[$i]}

    #-----------------------------------------------------
    EVAL_DATASET_CFG_PATH=cfgs/dataset_configs/$EVAL_DATASET/OD/${EVAL_DATASET}${EVAL_DATASET_EXTRA_TAG}_dataset.yaml                   
    #EVAL_DATASET_CFG_PATH=cfgs/dataset_configs/$EVAL_DATASET/OD/${EVAL_DATASET}${EVAL_DATASET_EXTRA_TAG}_dataset_nogtsampling.yaml     #use this to eval ON downsampled zenseact dataset 

    ROOT_PATH=/home/cgriesbacher/thesis/3DTrans
    RUN_PATH=${TRAIN_DATASET}_models/$MODEL/$EXTRA_TAG
    CHECKPOINT_PATH=$ROOT_PATH/$OUTPUT_FOLDER/$RUN_PATH/ckpt/checkpoint_epoch_$EPOCH.pth
    CFG_PATH=$ROOT_PATH/$OUTPUT_FOLDER/$RUN_PATH/$MODEL.yaml

    #multi gpu training
    NUM_GPUS=1
    cd "/home/cgriesbacher/thesis/3DTrans/tools"
    
    bash scripts/dist_test.sh $NUM_GPUS --cfg_file $CFG_PATH --ckpt $CHECKPOINT_PATH --batch_size $BATCHSIZE --workers $NUM_WORKERS --extra_tag $EXTRA_TAG --crosseval_dataset_cfg $EVAL_DATASET_CFG_PATH --eval_tag $EVAL_DATASET${EVAL_DATASET_EXTRA_TAG}

    #single gpu training for debugging
    #python test.py --cfg_file $CFG_PATH --ckpt $CHECKPOINT_PATH --batch_size $BATCHSIZE --workers $NUM_WORKERS --extra_tag $EXTRA_TAG --crosseval_dataset_cfg $EVAL_DATASET_CFG_PATH --eval_tag $EVAL_DATASET
done    