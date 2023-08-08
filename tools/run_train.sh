#################TRAINING#################

#CONFIG_FILE=kitti_models/second.yaml
#CONFIG_FILE=avltruck_models/IA-SSD.yaml
#CONFIG_FILE=avltruck_models/second.yaml
#DATASET=zod
DATASET=avltruck
#MODEL=IA-SSD
MODEL=second
#MODEL=pointpillar_1x
EXTRA_TAG=full_1epochs
EPOCHS=1
SUBSAMPLE=1
CKPT_SAVE_INTERVAL=1

CONFIG_FILE=${DATASET}_models/$MODEL.yaml


#multi gpu training
bash scripts/dist_train.sh 2 --cfg_file cfgs/$CONFIG_FILE --extra_tag $EXTRA_TAG  --epochs $EPOCHS --subsample $SUBSAMPLE --ckpt_save_interval $CKPT_SAVE_INTERVAL

