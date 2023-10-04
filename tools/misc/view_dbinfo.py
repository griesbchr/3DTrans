#%% imports
import pickle   
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%% load pickle file "avl_dbinfos.pkl" which is in AVLTruck folder
#load pickle file "avl_dbinfos.pkl" which is in AVLTruck folder
with open('../../data/avltruck/avl_dbinfos_train.pkl', 'rb') as f:
    #store as dataframe
    avl_dbinfos = pickle.load(f)

#dict_keys(['Vehicle_Drivable_Car', 'Human', 'LargeVehicle_Bus', 'Other', 'Vehicle_Ridable_Bicycle', 'Vehicle_Ridable_Motorcycle', 'Dont_Care', 'LargeVehicle_Truck', 'LargeVehicle_TruckCab', 'Trailer', 'Vehicle_Drivable_Van'])
classes = ['Vehicle_Drivable_Car', 'Human', 'LargeVehicle_Bus', 'Other', 
           'Vehicle_Ridable_Bicycle', 'Vehicle_Ridable_Motorcycle', 
           'Dont_Care', 'LargeVehicle_Truck', 'LargeVehicle_TruckCab', 
           'Trailer', 'Vehicle_Drivable_Van']

# create empty dataframe
gt_df = pd.DataFrame()

#loop over all classes and store in dataframe
for i in range(len(classes)):
    gt_df = pd.concat([gt_df, pd.DataFrame(avl_dbinfos[classes[i]])], ignore_index=True)

# Split the 'box3d_lidar' column into separate columns
gt_df[['x', 'y', 'z', 'l', 'w', 'h', 'theta']] = pd.DataFrame(gt_df['box3d_lidar'].tolist())

# Drop the original 'box3d_lidar' column
df = gt_df.drop('box3d_lidar', axis=1)

# rename col name to class
df.rename(columns={'name':'class'}, inplace=True)

# "image_idx" contains 'sequences/CityStreet_dgt_2021-07-08-15-24-00_0_s0/dataset/logical_frame_000020.json', please make a col containing CityStreet_dgt_2021-07-08-15-24-00_0_s0
df['sequence'] = df['image_idx'].str.split('/').str[1]

#rename image_idx to frame_idx
df.rename(columns={'image_idx':'frame_idx'}, inplace=True)

print("Number of instances per class:")
print(df.groupby('class')['num_points_in_gt'].count())

#%% load pickle file "avl_dbinfos.pkl" which is in AVLRooftop folder
#load pickle file "avl_dbinfos.pkl" which is in AVLRooftop folder
with open('../../data/avlrooftop/avl_dbinfos_train.pkl', 'rb') as f:
    #store as dataframe
    avl_dbinfos = pickle.load(f)

classes = ['Vehicle_Ridable_Bicycle', 'Vehicle_Ridable_Motorcycle', 
             'LargeVehicle_Truck', 'LargeVehicle_TruckCab', 
             'Trailer', 'LargeVehicle_Bus', 'LargeVehicle_Bus_Bendy', 
             'Vehicle_Drivable_Van', 'Vehicle_Drivable_Car', 
             'Human', 
             'PPObject_Stroller']

# create empty dataframe
gt_df = pd.DataFrame()

#loop over all classes and store in dataframe
for i in range(len(classes)):
    gt_df = pd.concat([gt_df, pd.DataFrame(avl_dbinfos[classes[i]])], ignore_index=True)

# Split the 'box3d_lidar' column into separate columns
gt_df[['x', 'y', 'z', 'l', 'w', 'h', 'theta']] = pd.DataFrame(gt_df['box3d_lidar'].tolist())

# Drop the original 'box3d_lidar' column
df = gt_df.drop('box3d_lidar', axis=1)

# rename col name to class
df.rename(columns={'name':'class'}, inplace=True)

# "image_idx" contains 'sequences/CityStreet_dgt_2021-07-08-15-24-00_0_s0/dataset/logical_frame_000020.json', please make a col containing CityStreet_dgt_2021-07-08-15-24-00_0_s0
df['sequence'] = df['image_idx'].str.split('/').str[1]

#rename image_idx to frame_idx
df.rename(columns={'image_idx':'frame_idx'}, inplace=True)

print("Number of instances per class:")
print(df.groupby('class')['num_points_in_gt'].count())
#%% load pickle file "zod_dbinfos.pkl" which is in zod folder

with open('../../data/zod/zod_dbinfos_train_full.pkl', 'rb') as f:
    #store as dataframe
    zod_dbinfos = pickle.load(f)

classes =  ["Vehicle_Car", "Vehicle_Van", "Vehicle_Truck", "Vehicle_Trailer", 
            "Vehicle_Bus", "Vehicle_HeavyEquip", "Vehicle_TramTrain",
            "VulnerableVehicle_Bicycle", "VulnerableVehicle_Motorcycle",
            "Pedestrian"]

# create empty dataframe
gt_df = pd.DataFrame()

#loop over all classes and store in dataframe
for i in range(len(classes)):
    gt_df = pd.concat([gt_df, pd.DataFrame(zod_dbinfos[classes[i]])], ignore_index=True)

# Split the 'box3d_lidar' column into separate columns
gt_df[['x', 'y', 'z', 'l', 'w', 'h', 'theta']] = pd.DataFrame(gt_df['box3d_lidar'].tolist())

# Drop the original 'box3d_lidar' column
df = gt_df.drop('box3d_lidar', axis=1)

# split "path" by "/", take the last element, split by "_", take the first element
df['frame_idx'] = df['path'].str.split('/').str[-1].str.split('_').str[0]


# rename col name to class
df.rename(columns={'name':'class'}, inplace=True)

print("Number of instances per class:")
print(df.groupby('class')['num_points_in_gt'].count())


#%% analyze the data
# print average number of points for each class
print("Average number of points for each class:")
print(df.groupby('class')['num_points_in_gt'].mean())

#prinnt mode of number of points for each class
print("Mode of number of points for each class:")
print(df.groupby('class')['num_points_in_gt'].agg(pd.Series.mode))

# plot histogram of number of points for each class, make a seperate plot for each class and plot all on one figure, label each plot with the class name
sns.set_style("darkgrid")
sns.set_context("paper")
sns.set(font_scale=1.5)
plt.figure(figsize=(20, 10))
for i in range(len(classes)):
    plt.subplot(3,4,i+1)
    
    # calc limit with 99% inliers
    upper_limit = np.percentile(df[df['class'] == classes[i]]['num_points_in_gt'], 95)
    
    #only use boxes with more than 5 points
    lower_limit = 5
    plt.hist(df[df['class'] == classes[i]]['num_points_in_gt'], bins=100, range=(lower_limit, upper_limit))
    plt.title(classes[i])
    plt.xlabel('num_points_in_gt')
    plt.ylabel('count')
plt.suptitle('Number of points in ground truth histogram')
plt.tight_layout()
plt.show()


# %% plot bar chart for number of instances per class, tile the x labels so they are readable
sns.set_style("darkgrid")
sns.set_context("paper")
sns.set(font_scale=1.5)
plt.figure(figsize=(20, 10))
agg_df = df.groupby('class')['num_points_in_gt'].count()
classes_plot = agg_df.index
plt.bar(classes_plot, df.groupby('class')['num_points_in_gt'].count())
plt.xticks(rotation=45, ha='right')
plt.xlabel('class')
plt.ylabel('count')
#print the number on top of each bar
for i in range(len(classes_plot)):
    plt.text(i, agg_df[i], agg_df[i], ha='center', va='bottom')
plt.suptitle('Number of instances per class')
plt.tight_layout()
plt.show()


# %% plot histogram with distance (can be calculated with x and y) for each class, make a seperate plot for each class and plot all on one figure, label each plot with the class name
sns.set_style("darkgrid")
sns.set_context("paper")
sns.set(font_scale=1.5)
plt.figure(figsize=(20, 10))
for i in range(len(classes)):
    plt.subplot(3,4,i+1)
    plt.hist(np.sqrt(df[df['class'] == classes[i]]['x']**2 + df[df['class'] == classes[i]]['y']**2), bins=100)
    plt.title(classes[i])
    plt.xlabel('distance')
    plt.ylabel('count')
plt.suptitle('Distance Histogram')
plt.tight_layout()
#name plot
plt.show()

# %% plot histogram with numnber of sequences in dataframe
sns.set_style("darkgrid")
sns.set_context("paper")
sns.set(font_scale=1.5)
plt.figure(figsize=(20, 10))
plt.hist(df['sequence'].value_counts(), bins=100)
plt.title('Number of sequences in dataframe')
plt.xlabel('number of sequences')
plt.ylabel('count')
plt.tight_layout()
plt.show()

#%% plot histogram with number of instances per frame
sns.set_style("darkgrid")
sns.set_context("paper")
sns.set(font_scale=1.5)
plt.figure(figsize=(20, 10))
plt.hist(df['frame_idx'].value_counts(), bins=100)
plt.title('Number of instances per frame')
plt.xlabel('number of instances')
plt.ylabel('count')
plt.tight_layout()
plt.show()

#%% print median value for {'l', 'w', 'h', 'theta'} and all the classes
print("Median value for {'l', 'w', 'h', 'theta'} and all the classes:")
print(df.groupby('class')['l', 'w', 'h', 'theta'].mean())

# print median value for {'x', 'y', 'z',} and all the classes
print("Median value for {'x', 'y', 'z',} and all the classes:")
print(df.groupby('class')['x', 'y', 'z'].mean())

# %%print distr
metrics = ['l', 'w', 'h', 'theta', 'x', 'y', 'z']

for metric in metrics:
    sns.set_style("darkgrid")
    sns.set_context("paper")
    sns.set(font_scale=1.5)
    plt.figure(figsize=(20, 10))
    for i in range(len(classes)):
        plt.subplot(3,4,i+1)
        plt.hist(df[df['class'] == classes[i]][metric], bins=100)
        plt.title(classes[i])
        plt.xlabel(metric)
        plt.ylabel('count')        
    plt.suptitle(metric + ' Histogram')
    plt.tight_layout()
    plt.show()


#%% filter rows by scene 
lidar_idx = "sequences/CityThoroughfare_dgt_2021-07-15-12-21-17_0_s0/dataset/logical_frame_000009.json"
#number of rows with scene name
print("Number of rows with scene name:")
print(df[df['frame_idx'] == lidar_idx].shape[0])


#%% load pickle file "avl_dbinfos.pkl" which is in AVLTruck folder
#load pickle file "avl_dbinfos.pkl" which is in AVLTruck folder
with open('/data/AVLTruck/avl_infos_train.pkl', 'rb') as f:
    #store as dataframe
    avl_train_infos = pickle.load(f)

# Convert the list of dictionaries directly into a DataFrame
avl_train_df = pd.DataFrame(avl_train_infos)

# Split the sub-dictionaries into separate columns
avl_train_df = pd.concat([avl_train_df.drop(columns=[key]).join(avl_train_df[key].apply(pd.Series)) for key in avl_train_df.columns], axis=1)

# Add a new index column to preserve the list location
avl_train_df['index'] = [idx for idx, _ in enumerate(avl_train_infos)]
avl_train_df.set_index('index', inplace=True)
            
lidar_idx = "sequences/CityThoroughfare_dgt_2021-07-15-12-21-17_0_s0/dataset/logical_frame_000009.json"
scene_df = avl_train_df[avl_train_df['lidar_idx'] == lidar_idx]


# %% given limits for x, y and z, calculate the percentage of rowss per class (from df) that are outside the limits for each class
#limits
x_min = -75
x_max = 125
y_min = -75
y_max = 75
z_min = -2
z_max = 4

#calculate percentage of classes that are outside the limits for each class
print("Percentage of classes that are outside the limits for each class:")
for i in range(len(classes)):
    print(classes[i],"\t\t\t" ,df[(df['class'] == classes[i]) & ((df['x'] < x_min) | (df['x'] > x_max) | (df['y'] < y_min) | (df['y'] > y_max) | (df['z'] < z_min) | (df['z'] > z_max))].shape[0] / df[df['class'] == classes[i]].shape[0])

# %% print the total number of instances that are inside the limits for every frame

#calculate a df containing only the instances that are inside the limit
df_limit = df[(df['x'] > x_min) & (df['x'] < x_max) & (df['y'] > y_min) & (df['y'] < y_max) & (df['z'] > z_min) & (df['z'] < z_max)]

#calculate the number of instances per frame
print("Number of instances per frame:")
print(df_limit['frame_idx'].value_counts())


# %% view point distribution within each class using a 2d (x,y) histogram with marginal histograms. 
# The marginal histograms should be normalized by the total number of instances in each class.

#plot 2d histogram with marginal histograms
sns.set_style("darkgrid")
sns.set_context("paper")
sns.set(font_scale=1.5)
plt.figure(figsize=(20, 10))
for i in range(len(classes)):
    plt.subplot(3,4,i+1)
    sns.histplot(data=df[df['class'] == classes[i]], x="x", y="y", bins=100, cbar=True)
    plt.title(classes[i])
    plt.xlabel('x')
    plt.ylabel('y')
plt.suptitle('2D Histogram with marginal histograms')
plt.tight_layout()
plt.show()
