"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d as o3d
import torch
import matplotlib
import numpy as np

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba



def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = o3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis


def fit_ground_plane_ransac(points, n_iterations=100, inlier_threshold=0.1):
    best_plane = None
    max_inliers = 0

    for _ in range(n_iterations):
        # Randomly select three non-collinear points to form a plane
        indices = np.random.choice(points.shape[0], 3, replace=False)
        sample_points = points[indices]

        # Calculate the coefficients of the plane equation (A, B, C, D)
        p1, p2, p3 = sample_points
        normal_vector = np.cross(p2 - p1, p3 - p1)
        A, B, C = normal_vector
        D = -np.dot(normal_vector, p1)

        # Calculate the distance of all points to the estimated plane
        distances = np.abs(np.dot(points, np.array([A, B, C])) + D) / np.linalg.norm([A, B, C])

        # Count the number of inliers
        inliers = np.sum(distances < inlier_threshold)

        # Update the best plane if this iteration has more inliers
        if inliers > max_inliers:
            max_inliers = inliers
            best_plane = (A, B, C, D)

    return best_plane

def draw_plane(coefficients, xlim, ylim, resolution=0.1):
    A, B, C, D = coefficients
    x_range = np.arange(xlim[0], xlim[1], resolution)
    y_range = np.arange(ylim[0], ylim[1], resolution)
    xx, yy = np.meshgrid(x_range, y_range)
    zz = (-A * xx - B * yy - D) / C

    points = np.column_stack((xx.flatten(), yy.flatten(), zz.flatten()))
    plane = o3d.geometry.PointCloud()
    plane.points = o3d.utility.Vector3dVector(points)

    return plane

def plane_coefficients_to_euler(coefficients):
    A, B, C, _ = coefficients

    # Extract the normal vector from plane coefficients
    normal_vector = np.array([A, B, C])

    # Normalize the normal vector to get unit normal
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    # Calculate roll, pitch, and yaw angles
    roll = np.arctan2(normal_vector[1], normal_vector[2])  # Rotation around x-axis
    pitch = np.arctan2(-normal_vector[0], np.sqrt(normal_vector[1]**2 + normal_vector[2]**2))  # Rotation around y-axis
    yaw = 0  # We assume yaw to be 0, as the ground plane does not have a rotation around the z-axis

    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)



def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True, fit_ground_plane=False):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 2.0
    #light grey color
    light_grey = np.array([0.7, 0.7, 0.7])
    vis.get_render_option().background_color = light_grey

    # draw origin
    if draw_origin:
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = o3d.geometry.PointCloud()
    pts.points = o3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is not None:
        pts.colors = o3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (1, 0, 0))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    if fit_ground_plane:
        xlim= [-50, 50]
        ylim= [-50, 50]
        plane_coefficients = fit_ground_plane_ransac(points[:, :3])
        print("Best Plane Coefficients:", plane_coefficients)
        
        # Assuming you have the plane coefficients in the 'plane_coefficients' variable
        roll_angle, pitch_angle, yaw_angle = plane_coefficients_to_euler(plane_coefficients)
        print("Roll Angle (x-axis):", roll_angle)
        print("Pitch Angle (y-axis):", pitch_angle)
        print("z-offset of the plane:", -plane_coefficients[3]/plane_coefficients[2])        
        plane_point_cloud = draw_plane(plane_coefficients, xlim, ylim)

        # Set the color of the plane to light grey
        grey_color = [0.7, 0.7, 0.7]  # RGB color values for light grey
        plane_point_cloud.paint_uniform_color(grey_color)
        
        vis.add_geometry(plane_point_cloud, point_show_normal=True)


    vis.run()
    vis.destroy_window()
