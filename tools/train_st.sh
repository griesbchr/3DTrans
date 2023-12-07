#################FINETUNING#################
# Finetune a single detector on an arbitrary dataset


DATASET=zod
MODEL=pvrcnnpp_STtruck
EXTRA_TAG=D6_30epochs_STtruck_ft_D16_50epochs_ros
PRETRAINED=/home/cgriesbacher/thesis/3DTrans/output_okeanos/output/zod_models/pvrcnnpp_ros/D16_50epochs/ckpt/checkpoint_epoch_50.pth

EPOCHS=30
CKPT_SAVE_INTERVAL=1
BATCHSIZE=4
WORKERS=4

CONFIG_FILE=${DATASET}_models/$MODEL.yaml


#single gpu training
python train_st.py --cfg_file cfgs/$CONFIG_FILE --extra_tag $EXTRA_TAG  --epochs $EPOCHS --ckpt_save_interval $CKPT_SAVE_INTERVAL --batch_size $BATCHSIZE --workers $WORKERS --pretrained_model $PRETRAINED

#multi gpu training
#NUM_GPUS=1
#bash scripts/dist_train.sh $NUM_GPUS --cfg_file cfgs/$CONFIG_FILE --extra_tag $EXTRA_TAG  --epochs $EPOCHS --ckpt_save_interval $CKPT_SAVE_INTERVAL --batch_size $BATCHSIZE --workers $WORKERS

