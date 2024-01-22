import json
import numpy as np
from matplotlib import pyplot as plt
import os
import glob
from scipy.spatial.transform import Rotation
from sklearn.cluster import KMeans, DBSCAN
from tqdm import tqdm


def read_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def plot_polar_image(phi, theta, beam_labels):
    #beam labels to int
    beam_labels = beam_labels.astype(int)

    #convert beam label to color
    colors = [0, 0.5, 1]
    #alternate colors for visualization
    color = np.zeros((beam_labels.shape[0]))
    for i in range(beam_labels.shape[0]):
        color[i] = colors[beam_labels[i] % 3]
        
    #plot polar image
    plt.figure()
    plt.scatter(phi, theta, s=0.1, c=color, cmap='jet')
    plt.xlabel('Theta (radians)')
    plt.ylabel('Phi (radians)')
    #plt.ylim([-0.1, 0.1])
    plt.show()

def beam_label(theta, beam, method='kmeans'):
    if method == 'kmeans++':
        estimator=KMeans(n_clusters=beam, init='k-means++', n_init="auto")
    elif method == 'kmeans':
        estimator=KMeans(n_clusters=beam, init='random', n_init='auto')
    elif method == 'dbscan':
        estimator=DBSCAN(eps=0.1, min_samples=10)
   
    res=estimator.fit_predict(theta.reshape(-1, 1))
    label=estimator.labels_

    #sort beam index based on the mean theta of each beam,
    #so that the beam index is consistent across different scans
    mean_theta = np.zeros((beam))
    for i in range(beam):
        mean_theta[i] = np.mean(theta[label == i])
    sorted_inds = np.argsort(mean_theta)
    sorted_labels = np.ones_like(label) * (-1)
    for i in range(beam):
        sorted_labels[label == sorted_inds[i]] = i  

    return sorted_labels 

def compute_angles(pc_np):
    tan_theta = pc_np[:, 2] / (pc_np[:, 0]**2 + pc_np[:, 1]**2)**(0.5)
    theta = np.arctan(tan_theta)

    sin_phi = pc_np[:, 1] / (pc_np[:, 0]**2 + pc_np[:, 1]**2)**(0.5)
    phi_ = np.arcsin(sin_phi)

    cos_phi = pc_np[:, 0] / (pc_np[:, 0]**2 + pc_np[:, 1]**2)**(0.5)
    phi = np.arccos(cos_phi)

    phi[phi_ < 0] = 2*np.pi - phi[phi_ < 0]
    phi[phi == 2*np.pi] = 0

    return theta, phi

def process_json_file(filepath):
    data = read_json(filepath)
    points = data['points']

    # lidar transforms
    device_position_dict = data['device_position']
    device_heading_dict = data['device_heading']
    ##
    device_position = np.array([device_position_dict['x'], 
                                device_position_dict['y'], 
                                device_position_dict['z']])
    device_q = Rotation.from_quat(
        [device_heading_dict['x'], 
            device_heading_dict['y'], 
            device_heading_dict['z'], 
            device_heading_dict['w']])

    R_ref_to_waymo_lidar = device_q.as_matrix() 

    lidar_sensors = {}
    # extract and dump lidar data if not available
    pcl_ref_pts = []
    lidar_sensors_pts = {
        "lidar_front": [],
        "lidar_right": [],
        "lidar_left":  []
    }

    for point in data['points']:
        #check if poin["s"] is in lidar_sensors_pts.keys()
        if point["s"] in lidar_sensors_pts.keys():    
            lidar_sensors_pts[point["s"]].append(np.array([point['x'], point['y'], point['z'], point['i']]))
        else:
            print("skipping frame", filepath, "because of unknown lidar sensor",point["s"])

    for key in lidar_sensors_pts.keys():
        pcl_ref_pts = lidar_sensors_pts[key]
        points = np.array(pcl_ref_pts)
        points[:,:3] = (points[:,:3] - device_position) @ R_ref_to_waymo_lidar

        points[:, 3] /= 255
        points[:, 3] = np.tanh(points[:, 3])
        
        lidar_sensors_pts[key] = points

    #compute beam labels for main lidar
    points = lidar_sensors_pts["lidar_front"]
    theta, phi = compute_angles(points)
    beam = 32
    method = 'kmeans++'
    beam_labels = beam_label(theta, beam, method)

    #append beam labels to points
    points = np.hstack((points, beam_labels.reshape(-1, 1)))
    lidar_sensors_pts["lidar_front"] = points

    #append  beam label -1 to other lidars
    for key in lidar_sensors_pts.keys():
        if key != "lidar_front":
            points = lidar_sensors_pts[key]
            points = np.hstack((points, np.ones((points.shape[0], 1)) * (-1)))
            lidar_sensors_pts[key] = points

    return lidar_sensors_pts

def main():
    sequences_file_path = "/media/data_8T/AVLRooftop/sequences/"
    #fetch all sequences   


    sequences = os.listdir(sequences_file_path)

    for sequence in tqdm(sequences):
        #go to raw folder
        raw_folder = os.path.join(sequences_file_path, sequence, 'raw')

        #read all files that end with .json

        frame_json_filepaths = glob.glob(os.path.join(raw_folder, '*.json'))

        for i, frame_json_filepath in enumerate(frame_json_filepaths):
            frame_id = os.path.splitext(os.path.basename(frame_json_filepath))[0].split("_")[-1]

            lidar_point_data = process_json_file(frame_json_filepath)

            #compute angles
            #theta, phi = compute_angles(lidar_data["lidar_front"])
            #plot 
            #plot_polar_image(phi, theta, lidar_data["lidar_front"][:, 4])

            #save lidar data to unpacked/lidar
            lidar_folder = os.path.join(sequences_file_path, sequence, 'unpacked', 'lidar')
            os.makedirs(lidar_folder, exist_ok=True)

            #concat lidar data
            lidar_data = np.vstack((lidar_point_data["lidar_front"], 
                                    lidar_point_data["lidar_right"], 
                                    lidar_point_data["lidar_left"]))

            #save lidar data
            np.save(os.path.join(lidar_folder, frame_id + '_beamlabels.npy'), lidar_data)

if __name__ == "__main__":
    main()