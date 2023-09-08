## Setup AVLRooftop dataset

1. Download and unzip dataset  
   `python -m pcdet.datasets.avlrooftop.downloader --root_path /data/AVLRooftop/ --sequence_file /data/AVLRooftop/training_sequences_10k.txt --split sequences`  
   Downloads data into the following structure
   <pre> Folder structure:   
    - AVLRooftop (ROOT_PATH)
      - sequences 
         - SEQUENCE_NAME
           - raw
             - json_data.zip
           - unpacked
             - camera_0
               - 0000.png
               - 0001.png
               - ...
             - camera_1
               - ...
             - camera_2
               - ...
             - camera_3
               - ...
             - lidar
               - 0000.pkl
               - 0001.pkl
               - ...
             - info.pkl
             - labels.json
             - metadata.json
   </pre>

   The info file contains the metadata such as lidar transformations etc
   The lidar data is already transformed to the waymo-like coordinate system (see description below).
   (z=0 NOT equal to groundplane!)
   The labels are not yet transformed!

2. Create training and test split  
   `python -m pcdet.datasets.avlrooftop.avlrooftop_dataset split_avl_data /data/AVLRooftop/training_sequences_10k.txt`
   <pre> Folder structure:   
    - AVLRooftop (ROOT_PATH)
      - train.txt
      - val.txt
   </pre>
3. Create infos and groundtruth database  
   `cd 3DTrans/`  
   `python -m pcdet.datasets.avlrooftop.avlrooftop_dataset create_avl_infos tools/cfgs/dataset_configs/avlrooftop/OD/avlrooftop_dataset.yaml`
   Creates info files that contains the transformed labels and information about the pointcloud
   ```
    {'point_cloud':
        {
            'num_features': 4,
            'lidar_idx': 'sequences/SEQNENCE_NAME/dataset/unpacked/lidar/XXXX.pkl'
        },
    'annos':
        {
            'name': (Nx1),
            'gt_boxes_lidar': (Nx7),
            'num_points_in_gt':(Nx1),
        }
        }
   ```
   While the raw labels are stored in the reference coordinate frame, the labels in the info.pkl is already transformed to the waymo-like coordinate frame.

## Coordinate Frames

### Reference coordinate frame

Lidar points and labels are originally stored in the reference coordinate frame. The origin is an arbitrary world location.

### AVL Lidar Frame

To transform lidar points to the AVL Lidar Frame, subtract the device location and rotate by `R_ref_to_avl_lidar`. The origin is at the Lidar location, but x points backwards in this frame.

### Waymo Lidar Frame

`R_ref_to_waymo_lidar` rotates the coordinte system such that x points forward. The origin is still at the lidar location (NOT z=0=groundplane!)
`points[:,:3] = (points[:,:3] - device_location) @ R_ref_to_waymo_lidar` directly transforms from the reference coordinate frame to the waymo-like coordinate frame (x pointing forward)

### Remark on labels

To transform the bbox labels (format `[x,y,z,l,w,h,theta]`), transform `[x,y,z]` like regular points.
The transformed `theta` is `theta_transformed = -theta - device_rotation` (device rotation in z-axis). I am not sure why the **minus sign** for theta is neccessary!
