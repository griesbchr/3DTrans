#################TRAINING#################

#CONFIG_FILE=kitti_models/second
#CONFIG_FILE=kitti_models/IA-SSD
CONFIG_FILE=avltruck_models/second
EPOCH=30

CHECKPOINT=/home/cgriesbacher/thesis/3DTrans/output/home/cgriesbacher/thesis/3DTrans/tools/cfgs/$CONFIG_FILE/default/ckpt/checkpoint_epoch_$EPOCH.pth

#multi gpu training
bash scripts/dist_test.sh 2 --cfg_file /home/cgriesbacher/thesis/3DTrans/tools/cfgs/$CONFIG_FILE.yaml --ckpt $CHECKPOINT

#single gpu training for debugging
#python train.py --cfg_file cfgs/kitti_models/second.yaml    