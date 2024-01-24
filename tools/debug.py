

import numpy as np
import open3d as o3d
from open3d.cuda.pybind.visualization import O3DVisualizer
from open3d.visualization import gui
import torch
import pickle


def show_scene(batch_dict, pred_dicts=None, pts_feature=None, batch=0):
    
    pts_coors = batch_dict["points"]

    gt_boxes = batch_dict["gt_boxes"]
    
    # Check if it's a PyTorch Tensor
    if isinstance(pts_coors, torch.Tensor):
        if pts_coors.device.type == 'cuda':
            # to cpu
            pts_coors = pts_coors.cpu().numpy()
        else:
            # to numpy
            pts_coors = pts_coors.numpy()

    #if type of points is not float64, convert
    if pts_coors.dtype != np.float64:
        pts_coors = pts_coors.astype(np.float64)

    # if point dim is 4, select rows where first dim is 0
    if pts_coors.shape[1] == 4:
        pts_coors = pts_coors[pts_coors[:,0]==batch][:,1:]

    # if point dim is 5, select rows where first dim is 0 and leace out intensity
    if pts_coors.shape[1] == 5:
        pts_coors = pts_coors[pts_coors[:,0]==batch][:,1:4]

    ###check point cloud validity###
    #check if number of points matches in features and coordinate array
    if pts_feature: assert pts_coors.shape[0] == pts_feature.shape[0], "different number of points in coordinate and feature array"

    #check if numpy array
    assert type(pts_coors) == type(np.array([])), "type of pts is not numpy array"

    #check if pts has two dimenstions
    assert len(pts_coors.shape) == 2, "pts has more than two dims. Should be (#points, 3)"

    #check if last dimention is 3
    assert pts_coors.shape[-1] == 3, "last dimension is not 3, but {}".format(pts_coors.shape[-1])

    ###set up vizualization###
    #create visualization
    vis = o3d.visualization.Visualizer()   
    vis.create_window()

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.82, 0.82, 0.82])
    opt.point_size = 1.0

    #create point cloud
    pcd = o3d.geometry.PointCloud()

    # From numpy to Open3D
    pcd.points = o3d.utility.Vector3dVector(pts_coors)    
    
    
    if pts_feature:
        # TODO: implement feature to color 
        #    color_array = np.array(o3d.visualization.ColorMapColor(colormap, color_values))  # generate colors
        #pcd.colors = o3d.utility.Vector3dVector(color_array / 255.0)
        pass
    else:   # select color based on z-value of points
        opt.point_color_option = o3d.visualization.PointColorOption.ZCoordinate


    vis.add_geometry(pcd)

    #add bounding boxes if results is not none
    if pred_dicts:
        boxes = pred_dicts[batch]["pred_boxes"]
        # Check if it's a PyTorch Tensor
        if isinstance(boxes, torch.Tensor):
            if boxes.device.type == 'cuda':
                # to cpu
                boxes = boxes.detach().cpu().numpy()
            else:
                # to numpy
                boxes = boxes.numpy()
        green_color = np.asarray([0, 1, 0])
        show_boxes_o3d(boxes, vis, color=green_color)

    #add ground truth boxes
    if gt_boxes is not None:
        gt_boxes = gt_boxes[batch]
        # Check if it's a PyTorch Tensor
        if isinstance(gt_boxes, torch.Tensor):
            if gt_boxes.device.type == 'cuda':
                # to cpu
                gt_boxes = gt_boxes.cpu().numpy()
            else:
                # to numpy
                gt_boxes = gt_boxes.numpy()
        red_color = np.asarray([1, 0, 0])
        show_boxes_o3d(gt_boxes, vis, color=red_color)
    vis.run()
    vis.destroy_window()

'''
adds bounding boxes to open3d visualization
boxes: numpy array of shape (#boxes, 7)

'''
def show_boxes_o3d(boxes, vis, color=[0, 0, 1], static_color=[0, 1, 0], static=None):
    for bid, box in enumerate(boxes):
        center = box[0:3]
        lwh = box[3:6]
        axis_angles = np.array([0, 0, box[6] + 1e-10])
        rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
        box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)
        line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

        lines = np.asarray(line_set.lines)
        lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

        line_set.lines = o3d.utility.Vector2iVector(lines)
        c = color
        if static is not None and not static[bid]:
            c = static_color
        line_set.paint_uniform_color(c)
        vis.add_geometry(line_set)
        
def pkl(arr, name):
    with open(name+".pkl", "wb") as f: 
        pickle.dump(arr, f)

def main():
    pass

if __name__ == '__main__':
    main()