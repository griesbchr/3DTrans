# workdir is 3DTrans/

BASE_DETECTOR=/home/cgriesbacher/thesis/3DTrans/output_okeanos/output/avlrooftop_models/pvrcnnpp_ros/D1_50epochs/ckpt/checkpoint_epoch_50.pth
CFG_FILE=cfgs/gace_avlrooftop_avltruck.yaml
DATA_FOLDER=gace_data
EPOCHS=(5 10 15 20 25 30 35 40)

for EPOCH in "${EPOCHS[@]}";do
    #run gace
    python gace_demo.py --base_detector_ckpt $BASE_DETECTOR --cfg_file $CFG_FILE --gace_data_folder $DATA_FOLDER --epochs $EPOCH # --gace_ckpt activate
done