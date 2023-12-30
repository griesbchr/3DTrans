#################FINETUNING#################
# Finetune a single detector on an arbitrary dataset with self-training

DATASET=avltruck
MODEL=pvrcnnpp_STzod
EXTRA_TAG=D16_15epochs_STzod_ft_D6_50epochs_gace
PRETRAINED=/home/cgriesbacher/thesis/3DTrans/output_okeanos/output/avltruck_models/pvrcnnpp_ros/D6_50epochs/ckpt/checkpoint_epoch_50.pth

EPOCHS=15
CKPT_SAVE_INTERVAL=1
BATCHSIZE=4
WORKERS=4

CONFIG_FILE=${DATASET}_models/$MODEL.yaml


#single gpu training
python train_st.py --cfg_file cfgs/$CONFIG_FILE --extra_tag $EXTRA_TAG  --epochs $EPOCHS --ckpt_save_interval $CKPT_SAVE_INTERVAL --batch_size $BATCHSIZE --workers $WORKERS --pretrained_model $PRETRAINED

#multi gpu training
#NUM_GPUS=1
#bash scripts/dist_train.sh $NUM_GPUS --cfg_file cfgs/$CONFIG_FILE --extra_tag $EXTRA_TAG  --epochs $EPOCHS --ckpt_save_interval $CKPT_SAVE_INTERVAL --batch_size $BATCHSIZE --workers $WORKERS

