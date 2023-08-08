'''
loads  db infos from pickle files and shifts them in z direction
'''
#%% imports
import pickle   
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
#%% load pickle file "avl_dbinfos.pkl" which is in AVLTruck folder

info_file_paths = {"avl_dbinfos_train": "/data/AVLTruck/avl_dbinfos_train.pkl",
              "avl_infos_val": "/data/AVLTruck/avl_infos_val.pkl",
              "avl_infos_train": "/data/AVLTruck/avl_infos_train.pkl"}
#load pickle file from info_file_paths
avl_dbinfos = {}
for key, value in info_file_paths.items():
    with open(value, 'rb') as f:
        avl_dbinfos[key] = pkl.load(f)

dbinfos_train = avl_dbinfos["avl_dbinfos_train"]
infos_val = avl_dbinfos["avl_infos_val"]
infos_train = avl_dbinfos["avl_infos_train"]

lidar_z_shift = -3.4097

# %% shift dbinfos_train

for key in dbinfos_train:
    instance_list = dbinfos_train[key]
    for instance in instance_list:
        instance["box3d_lidar"][2] -= lidar_z_shift

# %% shift infos_val and infos_train

for info in infos_val:
    if "annos" not in info:
        continue
    info["annos"]["location"][:, 2] -= lidar_z_shift
    info["annos"]["gt_boxes_lidar"][:, 2] -= lidar_z_shift

for info in infos_train:
    if "annos" not in info:
        continue
    info["annos"]["location"][:, 2] -= lidar_z_shift
    info["annos"]["gt_boxes_lidar"][:, 2] -= lidar_z_shift

# %% save shifted dbinfos_train, infos_val and infos_train

with open("/data/AVLTruck/avl_dbinfos_train.pkl", "wb") as f:
    pkl.dump(dbinfos_train, f)

with open("/data/AVLTruck/avl_infos_val.pkl", "wb") as f:
    pkl.dump(infos_val, f)

with open("/data/AVLTruck/avl_infos_train.pkl", "wb") as f:
    pkl.dump(infos_train, f)

