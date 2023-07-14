#################TRAINING#################

#CONFIG_FILE=kitti_models/second.yaml
#CONFIG_FILE=kitti_models/IA-SSD.yaml
CONFIG_FILE=avltruck_models/second.yaml

#multi gpu training
bash scripts/dist_train.sh 2 --cfg_file /home/cgriesbacher/thesis/3DTrans/tools/cfgs/$CONFIG_FILE

#single gpu training for debugging
#python train.py --cfg_file cfgs/kitti_models/second.yaml    