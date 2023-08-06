#%% imports
# imports for plotting
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = [20, 10]

import numpy as np

# import the ZOD DevKit
from zod import ZodFrames
from zod import ZodSequences

# import default constants
import zod.constants as constants
from zod.constants import Camera, Lidar, Anonymization, AnnotationProject

# import useful data classes
from zod.data_classes import LidarData

# NOTE! Set the path to dataset and choose a version
dataset_root = "/data/zod/"
version = "full"  # "mini" or "full"

# initialize ZodFrames
zod_frames = ZodFrames(dataset_root=dataset_root, version=version)


# %% train and val files

# get default training and validation splits
training_frames = zod_frames.get_split(constants.TRAIN)
validation_frames = zod_frames.get_split(constants.VAL)

# print the number of training and validation frames
print(f"Number of training frames: {len(training_frames)}")
print(f"Number of validation frames: {len(validation_frames)}")

# %% view frame
zod_frame = zod_frames[62592]

# we can access the metadata of a frame
metadata = zod_frame.metadata

# print a subsample of meta data
print(f"Frame id: {metadata.frame_id}")
print(f"Country Code: {metadata.country_code}")
print(f"Time of day: {metadata.time_of_day}")
print(f"Number of vehicles in the frame: {metadata.num_vehicles}")

# %% annotations
# get a new frame
zod_frame = zod_frames["029229"]

# get the object annotations
annotations = zod_frame.get_annotation(AnnotationProject.OBJECT_DETECTION)

# get a single annotation object by index
idx = 31
print(f"Annotation: {annotations[idx].name}")

annotation_3d = annotations[idx].box3d
print(annotation_3d)

# %% lidar data

zod_frame = zod_frames[62592]

# get the lidar core-frame
lidar_core_frame = zod_frame.info.get_key_lidar_frame()
print(lidar_core_frame)

# load the lidar data
pc = lidar_core_frame.read()

# LidarData dataclass is a wrapper around several numpy arrays
assert isinstance(pc, LidarData)

# alternatively, we can use helper functions on the frame itself
assert zod_frame.get_lidar()[0] == pc
assert zod_frame.get_lidar_frames()[0].read() == pc

print(f"Points: {pc.points.shape}")  # x, y, z
print(f"Timestamps: {pc.timestamps.shape}")
print(f"Intensity: {pc.intensity.shape}")


# %% estimating z offset by fitting the ground plane
from fit_ground_plane import fit_ground_plane 
from tqdm import tqdm

n_samples = 500
sample_list = np.random.choice(list(training_frames), n_samples, replace=False)

z_offsets = []
roll_angls = []
pitch_angls = []
for frame_id in tqdm(sample_list, desc="Fitting ground plane..."):
    zod_frame = zod_frames[frame_id]
    lidar_core_frame = zod_frame.info.get_key_lidar_frame()
    pc = lidar_core_frame.read()
    roll_angl, pitch_angle, z_offset = fit_ground_plane(pc.points, verbose=False)
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

# most occuring z-offet in 200 random samples:  [-2.08702246]


# %%
# Visualize LiDAR and objects in Bird's Eye View
from zod.visualization.lidar_bev import BEVBox

zod_frame = zod_frames["009158"]

# get the LiDAR point cloud
pcd = zod_frame.get_lidar()[0]

# get the object annotations
object_annotations = zod_frame.get_annotation(AnnotationProject.OBJECT_DETECTION)

import numpy as np

bev = BEVBox()
bev_image = bev(
    np.hstack((pcd.points, pcd.intensity[:, None])),
    (
        np.array([obj.name for obj in object_annotations if obj.box3d]),
        np.concatenate(
            [obj.box3d.center[None, :] for obj in object_annotations if obj.box3d], axis=0
        ),
        np.concatenate(
            [obj.box3d.size[None, :] for obj in object_annotations if obj.box3d], axis=0
        ),
        np.array([obj.box3d.orientation for obj in object_annotations if obj.box3d]),
    ),
)
# %%  open all frames to check if they can be loaded
from tqdm import tqdm

faulty_frames = []
for frame_id in tqdm(validation_frames, desc="Loading frames..."):
    zod_frame = zod_frames[frame_id]
    lidar_core_frame = zod_frame.info.get_key_lidar_frame()
    try:
        pc = lidar_core_frame.read()
    except:
        print("could not load frame: ", frame_id)
        faulty_frames.append(frame_id)
        continue

print("faulty frames: ", faulty_frames)

# %%
