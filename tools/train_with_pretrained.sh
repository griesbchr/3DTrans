#################FINETUNING#################
# Finetune a single detector on an arbitrary dataset


DATASET=avlrooftop
MODEL=pvrcnnpp_ros_rbus
EXTRA_TAG=D1_50epochs_D1_2epochs_R2
PRETRAINED=/home/cgriesbacher/thesis/3DTrans/output/avlrooftop_models/pvrcnnpp_ros/D1_50epochs/ckpt/checkpoint_epoch_50.pth

EPOCHS=2
CKPT_SAVE_INTERVAL=2
BATCHSIZE=8
WORKERS=4

NUM_GPUS=1
#export CUDA_VISIBLE_DEVICES=0,1

CONFIG_FILE=${DATASET}_models/$MODEL.yaml


#single gpu training
#python train.py --cfg_file cfgs/$CONFIG_FILE --extra_tag $EXTRA_TAG  --epochs $EPOCHS --ckpt_save_interval $CKPT_SAVE_INTERVAL --batch_size $BATCHSIZE --workers $WORKERS --pretrained_model $PRETRAINED

#multi gpu training
bash scripts/dist_train.sh $NUM_GPUS --cfg_file cfgs/$CONFIG_FILE --extra_tag $EXTRA_TAG  --epochs $EPOCHS --ckpt_save_interval $CKPT_SAVE_INTERVAL --batch_size $BATCHSIZE --workers $WORKERS

