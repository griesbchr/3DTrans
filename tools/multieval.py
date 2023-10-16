import subprocess

import time

source_dataset = "zod"
model = "centerpoint"
cfg = "D16_100epochs_4classes"
epoch = "100"


root_path= "/home/cgriesbacher/thesis/3DTrans"
run_path = source_dataset+"_models/"+model+"/"+cfg

checkpoint_file = root_path+"/output/"+run_path+"/ckpt/checkpoint_epoch_"+epoch+".pth"

#check if checkpoint exists
import os.path
if not os.path.isfile(checkpoint_file):
    print("checkpoint file not found")
    exit()

target_datasets = ["avltruck", "avlrooftop", "zod"]

for target_dataset in target_datasets:

    eval_dataset_cfg_path="cfgs/dataset_configs/"+target_dataset+"/OD/"+target_dataset+"_dataset.yaml"

    #print time and current model and epoch 
    print(time.strftime("%H:%M:%S", time.localtime()) + " ##################################" + model + "(" + cfg + ") trained on"+ source_dataset + "@" +target_dataset+ "##################################")

    #build config path
    config_file = "cfgs/" + source_dataset + "_models/"+model+".yaml"

    #build extra-tag
    extra_tag = cfg

    #build ckpt_save_interval
    ckpt_save_interval = epoch     #only save last checkpoint

    num_gpus = "2"
    batch_size = "8"    
    workers = "2"

    #build command
    command = "bash scripts/dist_test.sh "+num_gpus+" --cfg_file " + config_file + " --ckpt "+checkpoint_file+" --batch_size "+batch_size+" --workers "+workers+" --extra_tag "+cfg+" --crosseval_dataset_cfg "+eval_dataset_cfg_path+" --eval_tag " + target_dataset

    print(time.strftime("%H:%M:%S", time.localtime()))
    print(command)

    subprocess.call(command, shell=True)

