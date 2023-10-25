import subprocess

import time
import argparse
import ast # Used to safely parse the string into a list

def main():

    target_dataset = ["avltruck", "avlrooftop", "zod"]
    source_datasets = ["avltruck", "avlrooftop", "zod"]
    models = ["second_4classes", "second_4classes", "second_4classes"]
    cfgs = ["D6_100epochs", "D1_100epochs_old_split", "D16_100epochs"]
    epochs = "100"

    num_gpus = "1"
    batch_size = "8"    
    workers = "4"

    args = parse_config()
    args_dict = {
        'source_datasets': source_datasets,
        'models': models,
        'cfgs': cfgs,
        'epochs': epochs,
        'num_gpus': num_gpus,
        'batch_size': batch_size,
        'workers': workers
    }
    #overwrite with default values if not set via commandline
    for arg_name, arg_value in args_dict.items():
        if getattr(args, arg_name) is None:
            setattr(args, arg_name, arg_value)
        else:
            setattr(args, arg_name, ast.literal_eval(getattr(args, arg_name)))

    root_path= "/home/cgriesbacher/thesis/3DTrans"

    eval_dataset_cfg_path="cfgs/dataset_configs/"+target_dataset+"/OD/"+target_dataset+"_dataset.yaml"

    not_found = False
    for source_dataset, model,cfg in zip(args.source_datasets, args.models, args.cfgs):
        run_path = source_dataset+"_models/"+model+"/"+cfg
        checkpoint_file = root_path+"/output/"+run_path+"/ckpt/checkpoint_epoch_"+epochs+".pth"
        #check if checkpoint exists
        import os.path
        if not os.path.isfile(checkpoint_file):
            print("checkpoint file not found:", checkpoint_file)
            not_found = True
    if not_found: exit()

    for target_dataset in args.target_datasets:
        for source_dataset, model,cfg in zip(args.source_datasets, args.models, args.cfgs):

            run_path = source_dataset+"_models/"+model+"/"+cfg

            checkpoint_file = root_path+"/output/"+run_path+"/ckpt/checkpoint_epoch_"+epochs+".pth"

            #check if checkpoint exists
            import os.path
            if not os.path.isfile(checkpoint_file):
                print("checkpoint file not found:", checkpoint_file)
                continue

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
    parser = argparse.ArgumentParser(description='multi_detector_eval')
    parser.add_argument('--target_datasets', type=str, default=None, help='target datasets')
    parser.add_argument('--source_datasets', type=str, default=None, help='source datasets')
    parser.add_argument('--models', type=str, default=None, help='model')
    parser.add_argument('--cfgs', type=str, default=None, help='cfg')
    parser.add_argument('--epochs', type=str, default=None, help='epochs')
    parser.add_argument('--num_gpus', type=str, default=None, help='num_gpus')
    parser.add_argument('--batch_size', type=str, default=None, help='batch_size')
    parser.add_argument('--workers', type=str, default=None, help='workers')
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    main()
