#################TRAINING#################
# Train a single detector on an arbitrary dataset
set +e  # Suppress exit on errors

#######
DATASET=avltruck
MODEL=pvrcnnpp
EXTRA_TAG=D1_50epochs_dsnorm

EPOCHS=1
CKPT_SAVE_INTERVAL=1
BATCHSIZE=4
WORKERS=4

CONFIG_FILE=${DATASET}_models/$MODEL.yaml


#single gpu training
python train.py --cfg_file cfgs/$CONFIG_FILE --extra_tag $EXTRA_TAG  --epochs $EPOCHS --ckpt_save_interval $CKPT_SAVE_INTERVAL --batch_size $BATCHSIZE --workers $WORKERS

#multi gpu training
#NUM_GPUS=1
#bash scripts/dist_train.sh $NUM_GPUS --cfg_file cfgs/$CONFIG_FILE --extra_tag $EXTRA_TAG  --epochs $EPOCHS --subsample $SUBSAMPLE --ckpt_save_interval $CKPT_SAVE_INTERVAL --batch_size $BATCHSIZE --workers $WORKERS

