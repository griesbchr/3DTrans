import subprocess

import time
import argparse

def main():

    source_dataset = "avltruck"
    model = "second_4classes"
    cfg = "D6_100epochs"
    epoch = "100"

    num_gpus = "1"
    batch_size = "8"    
    workers = "4"

    args = parse_config()
    args_dict = {
        'source_dataset': source_dataset,
        'model': model,
        'cfg': cfg,
        'epoch': epoch,
        'num_gpus': num_gpus,
        'batch_size': batch_size,
        'workers': workers
    }
    #overwrite with default values if not set via commandline
    for arg_name, arg_value in args_dict.items():
        if getattr(args, arg_name) is None:
            setattr(args, arg_name, arg_value)

    root_path= "/home/cgriesbacher/thesis/3DTrans"
    run_path = source_dataset+"_models/"+model+"/"+cfg

    checkpoint_file = root_path+"/output/"+run_path+"/ckpt/checkpoint_epoch_"+epoch+".pth"

    #check if checkpoint exists
    import os.path
    if not os.path.isfile(checkpoint_file):
        print("checkpoint file not found:", checkpoint_file)
        exit()

    target_datasets = ["avltruck", "avlrooftop", "zod"]

    for target_dataset in target_datasets:

        eval_dataset_cfg_path="cfgs/dataset_configs/"+target_dataset+"/OD/"+target_dataset+"_dataset.yaml"

        #print time and current model and epoch 
        print(time.strftime("%H:%M:%S", time.localtime()) + " ##################################" + model + "(" + cfg + ") trained on"+ source_dataset + "@" +target_dataset+ "##################################")

        #build config path
        config_file = "cfgs/" + source_dataset + "_models/"+model+".yaml"

        #build command
        command = "bash scripts/dist_test.sh "+num_gpus+" --cfg_file " + config_file + " --ckpt "+checkpoint_file+" --batch_size "+batch_size+" --workers "+workers+" --extra_tag "+cfg+" --crosseval_dataset_cfg "+eval_dataset_cfg_path+" --eval_tag " + target_dataset

        print(time.strftime("%H:%M:%S", time.localtime()))
        print(command)

        subprocess.call(command, shell=True)

def parse_config():
    parser = argparse.ArgumentParser(description='multi_dataset_eval')
    parser.add_argument('--source_dataset', type=str, default=None, help='source dataset')
    parser.add_argument('--model', type=str, default=None, help='model')
    parser.add_argument('--cfg', type=str, default=None, help='cfg')
    parser.add_argument('--epoch', type=str, default=None, help='epoch')
    parser.add_argument('--num_gpus', type=str, default=None, help='num_gpus')
    parser.add_argument('--batch_size', type=str, default=None, help='batch_size')
    parser.add_argument('--workers', type=str, default=None, help='workers')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
