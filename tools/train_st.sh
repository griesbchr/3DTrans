#################FINETUNING#################
# Finetune a single detector on an arbitrary dataset


DATASET=avltruck
MODEL=pvrcnnpp_STrooftop
EXTRA_TAG=D1_60epochs_STrooftop_ft_D6_50epochs
PRETRAINED=/home/cgriesbacher/thesis/3DTrans/output/avltruck_models/pvrcnnpp/D6_50epochs/ckpt/checkpoint_epoch_50.pth

EPOCHS=60
CKPT_SAVE_INTERVAL=60
BATCHSIZE=4
WORKERS=4

CONFIG_FILE=${DATASET}_models/$MODEL.yaml


#single gpu training
python train_st.py --cfg_file cfgs/$CONFIG_FILE --extra_tag $EXTRA_TAG  --epochs $EPOCHS --ckpt_save_interval $CKPT_SAVE_INTERVAL --batch_size $BATCHSIZE --workers $WORKERS --pretrained_model $PRETRAINED

#multi gpu training
#NUM_GPUS=1
#bash scripts/dist_train.sh $NUM_GPUS --cfg_file cfgs/$CONFIG_FILE --extra_tag $EXTRA_TAG  --epochs $EPOCHS --ckpt_save_interval $CKPT_SAVE_INTERVAL --batch_size $BATCHSIZE --workers $WORKERS
