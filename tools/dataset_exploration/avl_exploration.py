#%% imports
from pcdet.datasets.avltruck.avl_dataset import AVLDataset
import yaml
from easydict import EasyDict
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

#%% import avl dataset 
dataset_path = '/home/cgriesbacher/thesis/3DTrans/tools/cfgs/dataset_configs/avltruck/OD/avltruck_dataset.yaml'
dataset_cfg = EasyDict(yaml.safe_load(open(dataset_path)))

class_names = ['Vehicle_Drivable_Car', 'Vehicle_Drivable_Van', 'Vehicle_Ridable_Motorcycle','Vehicle_Ridable_Bicycle',
            'Human', 'LargeVehicle_Bus', 'LargeVehicle_TruckCab', 'LargeVehicle_Truck', 'Trailer']
root_path = Path("/data/AVLTruck")
dataset = AVLDataset(dataset_cfg=dataset_cfg,
                         class_names=class_names,
                         root_path=root_path,
                         training=False)


# %% get training split and load training lidar
dataset.set_split('train')
training_frames = dataset.sample_id_list

# %% estimating z offset by fitting the ground plane
from fit_ground_plane import fit_ground_plane 

n_samples = 10
sample_list = np.random.choice(list(training_frames), n_samples, replace=False)

z_offsets = []
roll_angls = []
pitch_angls = []
for frame_id in tqdm(sample_list, desc="Fitting ground plane..."):
    points = dataset.get_lidar(frame_id)
    roll_angl, pitch_angle, z_offset = fit_ground_plane(points, verbose=False)
    #disregard of roll angle or pitch angle are over 10 degrees
    if abs(roll_angl) > 20 or abs(pitch_angle) > 20:
        continue
    roll_angls.append(roll_angl)
    pitch_angls.append(pitch_angle)
    z_offsets.append(z_offset)

# show the distribution of z offsets
plt.hist(z_offsets, bins=100)
plt.title("Distribution of z offsets")
plt.xlabel("z offset [m]")
plt.ylabel("count")
plt.show()

# show the distribution of roll angles
plt.hist(roll_angls, bins=100)
plt.title("Distribution of roll angles")
plt.xlabel("roll angle [deg]")
plt.ylabel("count")
plt.show()

# show the distribution of pitch angles
plt.hist(pitch_angls, bins=100)
plt.title("Distribution of pitch angles")
plt.xlabel("pitch angle [deg]")
plt.ylabel("count")
plt.show()


#find mode of z offsets
bin_size = 0.01     #1 cm

# Calculate the histogram and bin edges
hist, bin_edges = np.histogram(z_offsets, bins=np.arange(min(z_offsets), max(z_offsets) + bin_size, bin_size))

# Find the bin(s) with the highest frequency (the mode)
modes = bin_edges[:-1][hist == hist.max()]

# If there are multiple modes (multiple bins with the same highest frequency), you can get them all
print("most occuring z-offet in", n_samples, "random samples: ", modes)
#most occuring z-offet in 100 random samples:  [-3.40970763]

#%% load lidar frame
frame_path = "/data/AVLTruck/sequences/CityStreet_dgt_2021-07-07-13-20-04_0_s0/dataset/logical_frame_000002.npy"

#load npy frame
frame = np.load(frame_path, allow_pickle=True)

