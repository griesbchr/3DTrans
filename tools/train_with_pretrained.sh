#################FINETUNING#################
# Finetune a single detector on an arbitrary dataset


DATASET=zod
MODEL=pvrcnnpp_ros_rbds
EXTRA_TAG=D16_10epochs_rbds0.5_ft_D6_50epochs_ros
PRETRAINED=/home/cgriesbacher/thesis/3DTrans/output/zod_models/pvrcnnpp_ros/D6_50epochs/ckpt/checkpoint_epoch_50.pth

EPOCHS=10
CKPT_SAVE_INTERVAL=10
BATCHSIZE=16
WORKERS=4

NUM_GPUS=2
export CUDA_VISIBLE_DEVICES=0,1

CONFIG_FILE=${DATASET}_models/$MODEL.yaml


#single gpu training
#python train.py --cfg_file cfgs/$CONFIG_FILE --extra_tag $EXTRA_TAG  --epochs $EPOCHS --ckpt_save_interval $CKPT_SAVE_INTERVAL --batch_size $BATCHSIZE --workers $WORKERS --pretrained_model $PRETRAINED

#multi gpu training
bash scripts/dist_train.sh $NUM_GPUS --cfg_file cfgs/$CONFIG_FILE --extra_tag $EXTRA_TAG  --epochs $EPOCHS --ckpt_save_interval $CKPT_SAVE_INTERVAL --batch_size $BATCHSIZE --workers $WORKERS

