

TRAINING_FRAMES_LIST=("['011054','053362','054946','053497','052226','028063','003378','027010','015711','000095','022633','015711','065076','082177','063922','076836','067479','061583','051907','051907','051900','074549','074549','055431','051900','055431','055431','051907','055431','055431','066342','075935','044690','003885','019406','064524','018235','031092','065607','010426']" "['004743','002170','006261','004809','052206','005818','003814','002724','001097','061233','065607','073540','067479','053459','051999','071400','052574','055431','055431','051907','051900','051900','055431','051900','055431','051907','074549','074549','074549','074549','066342','051900','033888','010724','008378','002851','004787','054585','023747','007857']" "['007366','056516','019239','034142','005818','054585','017486','027010','015711','027010','094330','079721','052691','059405','062871','071879','070551','063368','074549','074549','055431','051900','051907','051900','051907','055431','051900','055431','055431','055431','088823','075935','031525','019094','009659','026401','012213','032821','015319','051907']" )  
SUBSAMPLEFACTORSS=("2500 2500")

EXPERIMENT_IDXS=(0 1)
for j in "${!EXPERIMENT_IDXS[@]}"; do
EXPERIMENT_IDX=${EXPERIMENT_IDXS[$j]}
TRAINING_FRAMES=${TRAINING_FRAMES_LIST[$j]}

for i in "${!SUBSAMPLEFACTORSS[@]}"; do
    SUBSAMPLEFACTORSSTR=${SUBSAMPLEFACTORSS[$i]}

    IFS=' ' read -r -a SUBSAMPLEFACTORS <<< "$SUBSAMPLEFACTORSSTR"
    #################FINETUNING#################
    # Finetune multiple detector on multiple arbitrary datasets, where each detector is finetuned on a single dataset (no sweep) 

    DATASETS=(zod zod)
    MODEL=pvrcnnpp_nogtsampling
    #SUBSAMPLEFACTORS=(600 600 100 100 1600 1600)

    PRETRAINED_DATASETS=(avlrooftop avltruck)
    PRETRAINED_MODEL=pvrcnnpp
    PRETRAINED_CFG_TAGS=(D1_50epochs D6_50epochs)
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
        EXTRA_TAG=D${SUBSAMPLEFACTOR}_${EPOCHS}epochs_ft_${PRETRAINED_CFG_TAG}_FRAMESELECT${EXPERIMENT_IDX}
        CONFIG_FILE=${DATASET}_models/$MODEL.yaml

        #single gpu training
        #python train.py --cfg_file cfgs/$CONFIG_FILE --extra_tag $EXTRA_TAG  --epochs $EPOCHS --ckpt_save_interval $CKPT_SAVE_INTERVAL --batch_size $BATCHSIZE --workers $WORKERS --pretrained_model $PRETRAINED

        #multi gpu training
        #bash scripts/dist_train.sh $NUM_GPUS --cfg_file cfgs/$CONFIG_FILE --extra_tag $EXTRA_TAG  --epochs $EPOCHS --ckpt_save_interval $CKPT_SAVE_INTERVAL --batch_size $BATCHSIZE --workers $WORKERS --no_eval "True" --pretrained_model $PRETRAINED --set DATA_CONFIG.SUBSAMPLEFACTOR $SUBSAMPLEFACTOR DATA_CONFIG.SET_TRAINING_FRAMES "True" DATA_CONFIG.TRAINING_FRAMES $TRAINING_FRAMES

    done

    OUTPUT_FOLDER=output

    TRAIN_DATASETS=(zod zod)
    MODEL=pvrcnnpp_nogtsampling
    #EXTRA_TAG=D1200_50epochs_ft_D1_50epochs
    EPOCH=50

    EVAL_DATASETS=(zod zod)

    BATCHSIZE=4
    NUM_WORKERS=2

    NUM_GPUS=1
    #export CUDA_VISIBLE_DEVICES=2,3

    for i in "${!DATASETS[@]}"; do
        EVAL_DATASET=${EVAL_DATASETS[$i]}
        TRAIN_DATASET=${TRAIN_DATASETS[$i]}
        PRETRAINED_CFG_TAG=${PRETRAINED_CFG_TAGS[$i]}
        SUBSAMPLEFACTOR=${SUBSAMPLEFACTORS[$i]}

        EXTRA_TAG=D${SUBSAMPLEFACTOR}_${EPOCHS}epochs_ft_${PRETRAINED_CFG_TAG}_FRAMESELECT${EXPERIMENT_IDX}
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



