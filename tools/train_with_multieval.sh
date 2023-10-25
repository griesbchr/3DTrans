#################TRAINING#################

#CONFIG_FILE=kitti_models/second.yaml
#CONFIG_FILE=avltruck_models/IA-SSD.yaml
#CONFIG_FILE=avltruck_models/second.yaml
DATASET=avltruck
#DATASET=avltruck
#DATASET=avlrooftop
#MODEL=IA-SSD
MODEL=second_4classes
#MODEL=pointpillar_1x
EXTRA_TAG=D6_100epochs
EPOCHS=100
SUBSAMPLE=6
CKPT_SAVE_INTERVAL=100
BATCHSIZE=16

CONFIG_FILE=${DATASET}_models/$MODEL.yaml


#single gpu training
python train.py --cfg_file cfgs/$CONFIG_FILE --extra_tag $EXTRA_TAG  --epochs $EPOCHS --subsample $SUBSAMPLE --ckpt_save_interval $CKPT_SAVE_INTERVAL --batch_size $BATCHSIZE --no_eval True

#multi gpu training
#bash scripts/dist_train.sh $NUM_GPUS --cfg_file cfgs/$CONFIG_FILE --extra_tag $EXTRA_TAG  --epochs $EPOCHS --subsample $SUBSAMPLE --ckpt_save_interval $CKPT_SAVE_INTERVAL --batch_size $BATCHSIZE

#multi dataset eval
NUM_GPUS=1
EVAL_BATCHSIZE=8
EVAL_WORKERS=4
python multi_dataset_eval.py --source_dataset $DATASET --model $MODEL --cfg $EXTRA_TAG --epoch $EPOCHS --num_gpus $NUM_GPUS --batch_size $EVAL_BATCHSIZE --workers $EVAL_WORKERS

