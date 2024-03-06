#SUBSAMPLEFACTORSS=("6000 6000 1000 1000 16000 16000"        #1h
#                    "3000 3000 500 500 8000 8000"           #2h
#                    "1200 1200 200 200 3200 3200")           #4h
#                    "600 600 100 100 1600 1600"             #8h
#                    "300 300 50 50 800 800")                #16h

SUBSAMPLEFACTORSS=( "60 60 10 10 160 160")

EXPERIMENT_IDXS=(0 1 2 3 4)
for j in "${!EXPERIMENT_IDXS[@]}"; do
EXPERIMENT_IDX=${EXPERIMENT_IDXS[$j]}


for i in "${!SUBSAMPLEFACTORSS[@]}"; do
    SUBSAMPLEFACTORSSTR=${SUBSAMPLEFACTORSS[$i]}

    IFS=' ' read -r -a SUBSAMPLEFACTORS <<< "$SUBSAMPLEFACTORSSTR"
    #################FINETUNING#################
    # Finetune multiple detector on multiple arbitrary datasets, where each detector is finetuned on a single dataset (no sweep) 

    DATASETS=(avltruck avltruck avlrooftop avlrooftop zod zod)
    MODEL=pvrcnnpp_nogtsampling
    #SUBSAMPLEFACTORS=(600 600 100 100 1600 1600)

    PRETRAINED_DATASETS=(avlrooftop zod avltruck zod avltruck avlrooftop)
    PRETRAINED_MODEL=pvrcnnpp
    PRETRAINED_CFG_TAGS=(D1_50epochs D16_50epochs D6_50epochs D16_50epochs D6_50epochs D1_50epochs)
    PRETRAINED_CKPT_EPOCH=50

    NUM_GPUS=1
    #export CUDA_VISIBLE_DEVICES=2,3
    EPOCHS=50
    CKPT_SAVE_INTERVAL=50
    BATCHSIZE=4
    WORKERS=4



    for i in "${!DATASETS[@]}"; do
        DATASET=${DATASETS[$i]}
        PRETRAINED_CFG_TAG=${PRETRAINED_CFG_TAGS[$i]}
        PRETRAINED_DATASET=${PRETRAINED_DATASETS[$i]}
        PRETRAINED=/home/cgriesbacher/thesis/3DTrans/output/${PRETRAINED_DATASET}_models/$PRETRAINED_MODEL/$PRETRAINED_CFG_TAG/ckpt/checkpoint_epoch_$PRETRAINED_CKPT_EPOCH.pth

        SUBSAMPLEFACTOR=${SUBSAMPLEFACTORS[$i]}
        EXTRA_TAG=D${SUBSAMPLEFACTOR}_50epochs_ft_${PRETRAINED_CFG_TAG}_RND${EXPERIMENT_IDX}
        CONFIG_FILE=${DATASET}_models/$MODEL.yaml

        #single gpu training
        #python train.py --cfg_file cfgs/$CONFIG_FILE --extra_tag $EXTRA_TAG  --epochs $EPOCHS --ckpt_save_interval $CKPT_SAVE_INTERVAL --batch_size $BATCHSIZE --workers $WORKERS --pretrained_model $PRETRAINED

        #multi gpu training
        bash scripts/dist_train.sh $NUM_GPUS --cfg_file cfgs/$CONFIG_FILE --extra_tag $EXTRA_TAG  --epochs $EPOCHS --ckpt_save_interval $CKPT_SAVE_INTERVAL --batch_size $BATCHSIZE --workers $WORKERS --no_eval "True" --pretrained_model $PRETRAINED --set DATA_CONFIG.SUBSAMPLEFACTOR $SUBSAMPLEFACTOR

    done

    OUTPUT_FOLDER=output

    TRAIN_DATASETS=(avltruck avltruck avlrooftop avlrooftop zod zod)
    MODEL=pvrcnnpp_nogtsampling
    #EXTRA_TAG=D1200_50epochs_ft_D1_50epochs
    EPOCH=50

    EVAL_DATASETS=(avltruck avltruck avlrooftop avlrooftop zod zod)

    BATCHSIZE=4
    NUM_WORKERS=2

    NUM_GPUS=1
    #export CUDA_VISIBLE_DEVICES=2,3

    for i in "${!DATASETS[@]}"; do
        EVAL_DATASET=${EVAL_DATASETS[$i]}
        TRAIN_DATASET=${TRAIN_DATASETS[$i]}
        PRETRAINED_CFG_TAG=${PRETRAINED_CFG_TAGS[$i]}
        SUBSAMPLEFACTOR=${SUBSAMPLEFACTORS[$i]}

        EXTRA_TAG=D${SUBSAMPLEFACTOR}_50epochs_ft_${PRETRAINED_CFG_TAG}_RND${EXPERIMENT_IDX}
        #-----------------------------------------------------
        EVAL_DATASET_CFG_PATH=cfgs/dataset_configs/$EVAL_DATASET/OD/${EVAL_DATASET}_dataset.yaml

        ROOT_PATH=/home/cgriesbacher/thesis/3DTrans
        RUN_PATH=${TRAIN_DATASET}_models/$MODEL/$EXTRA_TAG
        CHECKPOINT_PATH=$ROOT_PATH/$OUTPUT_FOLDER/$RUN_PATH/ckpt/checkpoint_epoch_$EPOCH.pth
        CFG_PATH=$ROOT_PATH/$OUTPUT_FOLDER/$RUN_PATH/$MODEL.yaml

        #multi gpu training

        cd "/home/cgriesbacher/thesis/3DTrans/tools"

        bash scripts/dist_test.sh $NUM_GPUS --cfg_file $CFG_PATH --ckpt $CHECKPOINT_PATH --batch_size $BATCHSIZE --workers $NUM_WORKERS --extra_tag $EXTRA_TAG --crosseval_dataset_cfg $EVAL_DATASET_CFG_PATH --eval_tag $EVAL_DATASET
    done


done

done