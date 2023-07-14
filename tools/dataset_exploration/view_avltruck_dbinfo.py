#%% imports
import pickle   
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%% load pickle file "avl_dbinfos.pkl" which is in AVLTruck folder
#load pickle file "avl_dbinfos.pkl" which is in AVLTruck folder
with open('/data/AVLTruck/avl_dbinfos_train.pkl', 'rb') as f:
    #store as dataframe
    avl_dbinfos = pickle.load(f)

#dict_keys(['Vehicle_Drivable_Car', 'Human', 'LargeVehicle_Bus', 'Other', 'Vehicle_Ridable_Bicycle', 'Vehicle_Ridable_Motorcycle', 'Dont_Care', 'LargeVehicle_Truck', 'LargeVehicle_TruckCab', 'Trailer', 'Vehicle_Drivable_Van'])
classes = ['Vehicle_Drivable_Car', 'Human', 'LargeVehicle_Bus', 'Other', 'Vehicle_Ridable_Bicycle', 'Vehicle_Ridable_Motorcycle', 'Dont_Care', 'LargeVehicle_Truck', 'LargeVehicle_TruckCab', 'Trailer', 'Vehicle_Drivable_Van']

#create empty dataframe
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

#%% print number of instances per class
print("Number of instances per class:")
print(df.groupby('class')['num_points_in_gt'].count())

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
plt.hist(df['image_idx'].value_counts(), bins=100)
plt.title('Number of instances per frame')
plt.xlabel('number of instances')
plt.ylabel('count')
plt.tight_layout()
plt.show()



