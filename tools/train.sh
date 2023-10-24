#################TRAINING#################

#CONFIG_FILE=kitti_models/second.yaml
#CONFIG_FILE=avltruck_models/IA-SSD.yaml
#CONFIG_FILE=avltruck_models/second.yaml
DATASET=avlrooftop
#DATASET=avltruck
#DATASET=avlrooftop
#MODEL=IA-SSD
MODEL=dsvt_pillar
#MODEL=pointpillar_1x
EXTRA_TAG=D1_100epochs_delete
EPOCHS=100
SUBSAMPLE=1
CKPT_SAVE_INTERVAL=100
BATCHSIZE=4

CONFIG_FILE=${DATASET}_models/$MODEL.yaml


#single gpu training
python train.py --cfg_file cfgs/$CONFIG_FILE --extra_tag $EXTRA_TAG  --epochs $EPOCHS --subsample $SUBSAMPLE --ckpt_save_interval $CKPT_SAVE_INTERVAL --batch_size $BATCHSIZE

#multi gpu training
#NUM_GPUS=1
#bash scripts/dist_train.sh $NUM_GPUS --cfg_file cfgs/$CONFIG_FILE --extra_tag $EXTRA_TAG  --epochs $EPOCHS --subsample $SUBSAMPLE --ckpt_save_interval $CKPT_SAVE_INTERVAL --batch_size $BATCHSIZE

