#################SELFTRAINING(FINETUNING)#################
# Selftrain a pretrained detector (source) on an arbitrary (target) dataset


DATASET=avlrooftop
MODEL=pvrcnnpp_ST++truck_gace
EXTRA_TAG=D6_15epochs_STzod_ft_D1_50epochs
PRETRAINED=/home/griesbacher/thesis/3DTrans/output/avlrooftop_models/pvrcnnpp_ros/D1_50epochs/ckpt/checkpoint_epoch_50.pth
EPOCHS=15
CKPT_SAVE_INTERVAL=1

NUM_GPUS=1
export CUDA_VISIBLE_DEVICES=0

BATCH_SIZE=8
WORKERS_PER_GPU=4

CONFIG_FILE=${DATASET}_models/$MODEL.yaml

#bash scripts/ST/dist_train_st.sh $NUM_GPUS --cfg_file cfgs/$CONFIG_FILE --extra_tag $EXTRA_TAG  --epochs $EPOCHS --ckpt_save_interval $CKPT_SAVE_INTERVAL --batch_size $BATCH_SIZE --workers $WORKERS_PER_GPU --pretrained_model $PRETRAINED --no_eval "True"


#python train_st.py --cfg_file cfgs/$CONFIG_FILE --extra_tag $EXTRA_TAG  --epochs $EPOCHS --ckpt_save_interval $CKPT_SAVE_INTERVAL --batch_size $BATCH_SIZE --workers $WORKERS_PER_GPU --pretrained_model $PRETRAINED --no_eval "True"



TRAIN_DATASET=avlrooftop
MODEL=pvrcnnpp_ST++truck_gace
CFG_TAG=D6_15epochs_STzod_ft_D1_50epochs
EPOCHS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)

EVAL_DATASET=avltruck
EVAL_DATASET_TAGS=""
NUM_GPUS=1
export CUDA_VISIBLE_DEVICES=0
BATCHSIZE=8
NUM_WORKERS=2

#loop over eval datasets
for EPOCH in "${EPOCHS[@]}"
do
    #EVAL_DATASET_CFG_PATH=cfgs/dataset_configs/$EVAL_DATASET/OD/${EVAL_DATASET}${EVAL_DATASET_TAG}_dataset_nogtsampling.yaml
    EVAL_DATASET_CFG_PATH=cfgs/dataset_configs/$EVAL_DATASET/OD/${EVAL_DATASET}${EVAL_DATASET_TAG}_dataset.yaml

    ROOT_PATH=/home/griesbacher/thesis/3DTrans
    RUN_PATH=${TRAIN_DATASET}_models/$MODEL/$CFG_TAG
    CHECKPOINT_PATH=$ROOT_PATH/output/$RUN_PATH/ckpt/checkpoint_epoch_$EPOCH.pth
    CFG_PATH=$ROOT_PATH/output/$RUN_PATH/$MODEL.yaml

    #eval tag is EVAL_DATASET+EVAL_DATASET_TAG
    EVAL_TAG=${EVAL_DATASET}${EVAL_DATASET_TAG}
    #multi gpu training
    cd "/home/griesbacher/thesis/3DTrans/tools"
    bash scripts/dist_test.sh $NUM_GPUS --cfg_file $CFG_PATH --ckpt $CHECKPOINT_PATH --batch_size $BATCHSIZE --workers $NUM_WORKERS --extra_tag $CFG_TAG --crosseval_dataset_cfg $EVAL_DATASET_CFG_PATH --eval_tag $EVAL_TAG

    #single gpu training for debugging
    #python test.py --cfg_file $CFG_PATH --ckpt $CHECKPOINT_PATH --batch_size 6 --workers 6 --extra_tag $CFG_TAG
done
