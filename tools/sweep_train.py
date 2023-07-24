import subprocess

import time

dataset = "avltruck"
models = ["second", "IA-SSD"]
epochs = ["1", "5", "10", "20", "40", "80"]
sampling = "1"

for model in models:
    for epoch  in epochs:
        #print time and current model and epoch 
        print(time.strftime("%H:%M:%S", time.localtime()) + " ##################################" + model + "(" + epoch + ")" + "##################################")

        #build config path
        config_file = "cfgs/" + dataset + "_models/"+model+".yaml"

        #build extra-tag
        extra_tag = "D"+sampling+"_"+epoch+"epochs"

        #build ckpt_save_interval
        ckpt_save_interval = epoch     #only save last checkpoint

        #force batch size to 2
        batch_size = "2"    #just in case of CUDA out of memory

        #build command
        command = "bash scripts/dist_train.sh 2 --cfg_file " + config_file + " --extra_tag "+ extra_tag +  " --epochs " + epoch + " --subsample "+sampling+" --ckpt_save_interval " + ckpt_save_interval # + " --batch_size " + batch_size
        
        print(time.strftime("%H:%M:%S", time.localtime()))
        print(command)

        subprocess.call(command, shell=True)

        #wait for 60 minutes to cool down the GPU
        time.sleep(3600)
