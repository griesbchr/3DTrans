#################TRAINING#################
# Train a single detector on an arbitrary dataset
set +e  # Suppress exit on errors


#######
DATASET=zod
MODEL=second_feat3
EXTRA_TAG=D16_100epochs

EPOCHS=100
CKPT_SAVE_INTERVAL=100
BATCHSIZE=8
WORKERS=4

CONFIG_FILE=${DATASET}_models/$MODEL.yaml


#single gpu training
#python train.py --cfg_file cfgs/$CONFIG_FILE --extra_tag $EXTRA_TAG  --epochs $EPOCHS --ckpt_save_interval $CKPT_SAVE_INTERVAL --batch_size $BATCHSIZE --workers $WORKERS
#multi gpu training
#NUM_GPUS=1
#bash scripts/dist_train.sh $NUM_GPUS --cfg_file cfgs/$CONFIG_FILE --extra_tag $EXTRA_TAG  --epochs $EPOCHS --subsample $SUBSAMPLE --ckpt_save_interval $CKPT_SAVE_INTERVAL --batch_size $BATCHSIZE --workers $WORKERS


#################CROSS DATASET TESTING#################
# Test a single detector on a single arbitrary dataset
OUTPUT_FOLDER=output

TRAIN_DATASET=zod
MODEL=second
EXTRA_TAG=D16_100epochs
EPOCHS=(100)

EVAL_DATASET=avltruck
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
    EVAL_DATASET_CFG_PATH=cfgs/dataset_configs/$EVAL_DATASET/OD/${EVAL_DATASET}${EVAL_DATASET_EXTRA_TAG}_dataset_feat3.yaml                   
    #EVAL_DATASET_CFG_PATH=cfgs/dataset_configs/$EVAL_DATASET/OD/${EVAL_DATASET}${EVAL_DATASET_EXTRA_TAG}_dataset_nogtsampling.yaml     #use this to eval ON downsampled zenseact dataset 

    echo "Testing $MODEL on $EVAL_DATASET_CFG_PATH with $EXTRA_TAG"
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




#################CROSS DATASET TESTING#################
# Test a single detector on a single arbitrary dataset
OUTPUT_FOLDER=output

TRAIN_DATASET=zod
MODEL=second
EXTRA_TAG=D16_100epochs
EPOCHS=(100)


EVAL_DATASET=zod
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
    EVAL_DATASET_CFG_PATH=cfgs/dataset_configs/$EVAL_DATASET/OD/${EVAL_DATASET}${EVAL_DATASET_EXTRA_TAG}_dataset_feat3.yaml                   
    #EVAL_DATASET_CFG_PATH=cfgs/dataset_configs/$EVAL_DATASET/OD/${EVAL_DATASET}${EVAL_DATASET_EXTRA_TAG}_dataset_nogtsampling.yaml     #use this to eval ON downsampled zenseact dataset 

    echo "Testing $MODEL on $EVAL_DATASET_CFG_PATH with $EXTRA_TAG"
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





#################CROSS DATASET TESTING#################
# Test a single detector on a single arbitrary dataset
OUTPUT_FOLDER=output

TRAIN_DATASET=zod
MODEL=second
EXTRA_TAG=D16_100epochs
EPOCHS=(100)

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
    EVAL_DATASET_CFG_PATH=cfgs/dataset_configs/$EVAL_DATASET/OD/${EVAL_DATASET}${EVAL_DATASET_EXTRA_TAG}_dataset_feat3.yaml                   
    #EVAL_DATASET_CFG_PATH=cfgs/dataset_configs/$EVAL_DATASET/OD/${EVAL_DATASET}${EVAL_DATASET_EXTRA_TAG}_dataset_nogtsampling.yaml     #use this to eval ON downsampled zenseact dataset 

    echo "Testing $MODEL on $EVAL_DATASET_CFG_PATH with $EXTRA_TAG"
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

