#################TRAINING#################

#CONFIG_FILE=kitti_models/second.yaml
#CONFIG_FILE=kitti_models/IA-SSD.yaml
CONFIG_FILE=avltruck_models/second.yaml
EXTRA_TAG=D5_5epochs
EPOCHS=5
SUBSAMPLE=5
CKPT_SAVE_INTERVAL=1

#multi gpu training
bash scripts/dist_train.sh 2 --cfg_file cfgs/$CONFIG_FILE --extra_tag $EXTRA_TAG  --epochs $EPOCHS --subsample $SUBSAMPLE --ckpt_save_interval $CKPT_SAVE_INTERVAL

