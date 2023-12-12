# workdir is 3DTrans/

BASE_DETECTOR=/home/cgriesbacher/thesis/3DTrans/output_okeanos/output/zod_models/pvrcnnpp_ros/D16_50epochs/ckpt/checkpoint_epoch_50.pth
CFG_FILE=gace/cfgs/gace_zod.yaml
DATA_FOLDER=gace/gace_data

#run gace
python gace_demo.py --base_detector_ckpt $BASE_DETECTOR --cfg_file $CFG_FILE --gace_data_folder $DATA_FOLDER
