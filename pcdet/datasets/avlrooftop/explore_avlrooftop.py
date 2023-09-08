#%% imports
import pickle
from pathlib import Path
import orjson
import shutil

# %%

data_path = Path("/data/AVLRooftop")

#print list of folders in data_path
folders = [f.stem for f in (data_path/"sequences").iterdir() if f.is_dir()]
sequence_categories_list = ["_".join(folder.split("_")[:3]) for folder in folders]
sequence_categories = list(set(sequence_categories_list))
print(sequence_categories)



# %%
info_path = Path("/data/AVLRooftop/sequences/CITY_Normal_junction_20200309145430_1/unpacked/info.pkl")
with open(info_path, 'rb') as f:
    info = pickle.load(f)
print(info["camera_front__0001"]["lidar_sensors"]["lidar_0"].keys())
# %% read training_seq_file
training_seq_filepath = Path("/data/AVLRooftop/training_sequences_10k.txt")
training_seq_file = open(training_seq_filepath, "r")
training_seq_list = training_seq_file.readlines()
training_seq_list = [seq.strip() for seq in training_seq_list]
training_seq_categories_list = ["_".join(seq.split("_")[:3]) for seq in training_seq_list]
training_seq_categories = list(set(training_seq_categories_list))
print("number of categories", len(training_seq_categories))

categories_count = {i:training_seq_categories_list.count(i) for i in training_seq_categories_list}
print(categories_count)

training_seq_frame_list = ["_".join(seq.split("_")[:4]) for seq in training_seq_list]
training_seq_frames = list(set(training_seq_frame_list))
print("number of overall sequences", len(training_seq_frame_list))
print("number of individual sequences", len(training_seq_frames))
# %% read a metadata file
metadata_path = Path("/data/AVLRooftop/sequences/CITY_Normal_junction_20200309145430_1/unpacked/metadata.json")

with open(str(metadata_path), 'r') as f:
    data = orjson.loads(f.read())

#print(data.keys())

print("labelling infos", data["labelling_profile"]["features"].keys())
#labelling infos dict_keys(['box', 'lane', 'polygon', 'point', '3d_bbox', '3d_polyline', '3d_polygon', '3d_point', '3d_instance_point'])
print("labelling infos", data["labelling_profile"]["features"]["point"].keys())
print("labelling infos", data["labelling_profile"]["features"]["3d_bbox"].keys())
print("labelling infos", data["labelling_profile"]["features"]["3d_point"].keys())
print("labelling infos", data["labelling_profile"]["features"]["3d_instance_point"].keys())

# %% check content of lidar file
lidar_path = Path("/data/AVLRooftop/sequences/CITY_Normal_junction_20200309145430_1/unpacked/lidar/0000.pkl")

lidar_file = open(lidar_path, "rb")
lidar_data = pickle.load(lidar_file)
print(lidar_data.keys())
# %% open info file 
info_path = Path("/data/AVLTruck/avl_infos_train.pkl")

infos_file = open(info_path, "rb")
infos = pickle.load(infos_file)
print(infos[0])
# %%

# iterate over all folders contained in sequences /data/AVLRooftop/sequences and move the content of /raw to /data/AVLRooftop/download/sequences/SEQUENCE_NAME
root_path = Path("/data/AVLRooftop/")
sequences_path = root_path / "sequences"
download_path = root_path / "download"
download_sequences_path = download_path / "sequences"
download_sequences_path.mkdir(exist_ok=True)
for sequence in sequences_path.iterdir():
    sequence_name = sequence.stem
    print(sequence_name)
    download_sequence_path = download_sequences_path / sequence_name
    download_sequence_path.mkdir(exist_ok=True)
    target_file = sequence/"raw"/"json_data.zip"
    if target_file.exists():
        #copy file to download_sequence_path
        print("copying file")
        shutil.copy(target_file, download_sequence_path)
    else:
        print("no file found")

            
# %% iterate over sequences and remove unpacked folder
root_path = Path("/data/AVLRooftop/")
sequences_path = root_path / "sequences"
for sequence in sequences_path.iterdir():
    sequence_name = sequence.stem
    print(sequence_name)
    unpacked_path = sequence/"unpacked"
    if unpacked_path.exists():
        shutil.rmtree(unpacked_path)
    else:
        print("no unpacked folder found")




# %% view info file
info_path = Path("/data/AVLTruck/avl_infos_train.pkl")
with open(info_path, 'rb') as f:
    info = pickle.load(f)
print(info[0])

# %%
