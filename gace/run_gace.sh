# workdir is 3DTrans/

BASE_DETECTOR=/home/cgriesbacher/thesis/3DTrans/output_okeanos/output/zod_models/pvrcnnpp_ros/D16_50epochs/ckpt/checkpoint_epoch_50.pth
CFG_FILE=gace/cfgs/gace_zod.yaml
DATA_FOLDER=gace/gace_data
GRACE_CKPT_PATH="//home/cgriesbacher/thesis/3DTrans/gace_output/2023-12-12_10-03-19/gace_model.pth"

#run gace
python gace_demo.py --base_detector_ckpt $BASE_DETECTOR --cfg_file $CFG_FILE --gace_data_folder $DATA_FOLDER --gace_ckpt $GRACE_CKPT_PATH
