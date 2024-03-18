#################TRAINING#################
# Train a single detector on an arbitrary dataset
set +e  # Suppress exit on errors

#######
DATASET=avltruck
MODEL=pvrcnnpp_nogtsampling
EXTRA_TAG=D1_50epochs_debug

EPOCHS=50
CKPT_SAVE_INTERVAL=50
BATCHSIZE=4
WORKERS=4

CONFIG_FILE=${DATASET}_models/$MODEL.yaml


#single gpu training
python train.py --cfg_file cfgs/$CONFIG_FILE --extra_tag $EXTRA_TAG  --epochs $EPOCHS --ckpt_save_interval $CKPT_SAVE_INTERVAL --batch_size $BATCHSIZE --workers $WORKERS --set DATA_CONFIG.SET_TRAINING_FRAMES "True" DATA_CONFIG.TRAINING_FRAMES "['sequences/Motorway_dgt_2021-08-20-10-29-08_0_s0/dataset/logical_frame_000001.json','sequences/Motorway_dgt_2021-08-20-10-29-08_0_s0/dataset/logical_frame_000005.json','sequences/Motorway_dgt_2021-08-20-10-29-08_0_s0/dataset/logical_frame_000009.json','sequences/Motorway_dgt_2021-08-20-10-29-08_0_s0/dataset/logical_frame_000013.json','sequences/Motorway_dgt_2021-08-20-10-29-08_0_s0/dataset/logical_frame_000017.json','sequences/Motorway_dgt_2021-08-20-10-29-08_0_s0/dataset/logical_frame_000020.json']"

#multi gpu training
#NUM_GPUS=1
#bash scripts/dist_train.sh $NUM_GPUS --cfg_file cfgs/$CONFIG_FILE --extra_tag $EXTRA_TAG  --epochs $EPOCHS --subsample $SUBSAMPLE --ckpt_save_interval $CKPT_SAVE_INTERVAL --batch_size $BATCHSIZE --workers $WORKERS

