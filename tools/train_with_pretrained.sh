#################FINETUNING#################
# Finetune a single detector on an arbitrary dataset


DATASET=avltruck
MODEL=pvrcnnpp_sn2rooftop
EXTRA_TAG=D6_1epochs_sn2rooftop_ft_D6_50epochs
SUBSAMPLE=6
PRETRAINED=/home/cgriesbacher/thesis/3DTrans/output/avltruck_models/pvrcnnpp/D6_50epochs/ckpt/checkpoint_epoch_50.pth

EPOCHS=1
CKPT_SAVE_INTERVAL=1
BATCHSIZE=4
WORKERS=4

CONFIG_FILE=${DATASET}_models/$MODEL.yaml


#single gpu training
python train.py --cfg_file cfgs/$CONFIG_FILE --extra_tag $EXTRA_TAG  --epochs $EPOCHS --subsample $SUBSAMPLE --ckpt_save_interval $CKPT_SAVE_INTERVAL --batch_size $BATCHSIZE --workers $WORKERS --pretrained_model $PRETRAINED

#multi gpu training
#NUM_GPUS=1
#bash scripts/dist_train.sh $NUM_GPUS --cfg_file cfgs/$CONFIG_FILE --extra_tag $EXTRA_TAG  --epochs $EPOCHS --subsample $SUBSAMPLE --ckpt_save_interval $CKPT_SAVE_INTERVAL --batch_size $BATCHSIZE --workers $WORKERS

