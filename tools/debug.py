

import numpy as np
import open3d as o3d
from open3d.cuda.pybind.visualization import O3DVisualizer
from open3d.visualization import gui

import pickle


def show_scene(pts_coors, result=None, pts_feature=None):
    '''
    make sure to open display first if X11 forwarding is used!

    pts_coors: numpy array of type float64, shape (#points, 3)
    pts_feature: numpy array of shape (#points, XXXX)

    
    the following conversions might be useful
    pts = points[0][:,:3].cpu().numpy().astype(np.float64)
    '''

    ###check point cloud validity###
    #check if number of points matches in features and coordinate array
    if pts_feature: assert pts_coors.shape[0] == pts_feature.shape[0], "different number of points in coordinate and feature array"

    #check if numpy array
    assert type(pts_coors) == type(np.array([])), "type of pts is not numpy array"

    #check if pts has two dimenstions
    assert len(pts_coors.shape) == 2, "pts has more than two dims. Should be (#points, 3)"

    #check if last dimention is 3
    assert pts_coors.shape[-1] == 3, "last dimension is not 3"

    ###set up vizualization###
    #create visualization
    vis = o3d.visualization.Visualizer()   
    vis.create_window()

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
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
    if result:
        lidar_boxes = result['boxes_3d']
        scores = result['scores_3d'].detach().cpu().numpy()
        labels = result['labels_3d'].detach().cpu().numpy()
        boxes = lidar_boxes.tensor.detach().cpu().numpy()
        show_boxes_o3d(boxes, vis)
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