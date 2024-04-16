"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d as o3d
import torch
import matplotlib
import numpy as np

from PIL import Image, ImageFont, ImageDraw
from pyquaternion import Quaternion

box_colormap = [
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
    [1, 1, 1],
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

#https://github.com/isl-org/Open3D/issues/2
def text_3d(text, pos, direction=None, degree=0.0, font='/usr/share/fonts/truetype/freefont/FreeMono.ttf', font_size=200, density=10):
    """
    Generate a 3D text point cloud used for visualization.
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param font: Name of the font - change it according to your system
    :param font_size: size of the font
    :return: o3d.geoemtry.PointCloud object
    """
    if direction is None:
        direction = (1., 0., 0.)

    font_obj = ImageFont.truetype(font, font_size)
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    pcd.points = o3d.utility.Vector3dVector(indices / 1000)

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    
    quaternion_z = Quaternion(axis=[0, 0, 1], angle=np.pi)
    trans = (quaternion_z * Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
             Quaternion(axis=direction, degrees=degree)).transformation_matrix
    trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)
    return pcd


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

        if score is not None:
            corners = box3d.get_box_points()
            #vis.add_3d_label(corners[5], '%.2f' % score[i])
            #label = o3d.visualization.gui.Label3D(str(score[i]),corners[5])
            vis.add_geometry(text_3d(str(np.round(score[i], 4)),corners[3]))
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

def add_line(vis, start, end, color=(0, 0, 0)):
    
    # Create a LineSet object
    line_set = o3d.geometry.LineSet()

    # Set the points of the line
    line_set.points = o3d.utility.Vector3dVector([start, end])

    # Define the connection between the points (indices of the points list)
    lines = [[0, 1]]  # Connects the first point to the second point
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color(color)

    # Visualize the line
    vis.add_geometry(line_set)

def add_circle(vis, radius, num_points, color=(0, 0, 0)):
    # Generate points for the circle
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.zeros(num_points)  # Assuming the circle is in the XY plane

    # Create a point cloud
    circle_points2 = np.vstack((x, y, z)).T  # Combine x, y, z coordinates
    circle_pcd2 = o3d.geometry.PointCloud()
    circle_pcd2.points = o3d.utility.Vector3dVector(circle_points2)
    circle_pcd2.paint_uniform_color(color)

    vis.add_geometry(circle_pcd2)


def draw_scenes(points, gt_boxes=None, det_boxes=None, det_labels=None, det_scores=None, point_colors=None, draw_origin=True, fit_ground_plane=False, image_path=None, view_control=None):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(det_boxes, torch.Tensor):
        det_boxes = det_boxes.cpu().numpy()

    #if points dim is 5, remove batch dim
    if len(points.shape) == 5:
        points = points[:,1:]
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 2.0
    #light grey color
    light_grey = np.array([0.7, 0.7, 0.7])
    white = np.array([1, 1, 1])
    vis.get_render_option().background_color = white

    # draw origin
    if draw_origin:
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = o3d.geometry.PointCloud()
    pts.points = o3d.utility.Vector3dVector(points[:, :3])

    if point_colors is not None:
        if (point_colors.shape[-1] == 1 or len(point_colors.shape) == 1):
            from matplotlib import cm
            colormap = cm.get_cmap('viridis')
            point_colors = colormap(point_colors.squeeze())[:, :3] 
        pts.colors = o3d.utility.Vector3dVector(point_colors)
        
    vis.add_geometry(pts)


    #add line in x directiron
    # Define the points for the line in the X direction
    point_start = np.array([0, 0, 0])  # Starting point of the line
    point_end = np.array([100, 0, 0])  # Ending point of the line, 1 unit along the X axis
    #add_line(vis, point_start, point_end)

    # Circle parameters
    radius = 75  # 50 meters
    num_points = 5000  # Number of points to generate
    add_circle(vis, radius, num_points)

    # Circle parameters
    radius = 123  # 50 meters
    num_points = 20000  # Number of points to generate
    add_circle(vis, radius, num_points)


    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0.99, 0.0, 0.0))

    if det_boxes is not None:
        vis = draw_box(vis, det_boxes, (0.0, 0.99, 0.0), det_labels, det_scores)

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

    #read file PinholeCameraParameters.json
    camera_params = o3d.io.read_pinhole_camera_parameters("./visual_utils/PinholeCameraParameters.json")
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(camera_params)
    ctr.rotate(0.0, -np.pi / 2.0, 0.0)
    vis.update_renderer()
    vis.poll_events()
    
    if view_control is not None and image_path is not None:
        #insert image view here, can be copied by pressing Ctrl+C in open3d window and paste in editor file
        #ctr.set_front(view_control["front"])
        #ctr.set_lookat(view_control["lookat"])
        #ctr.set_up(view_control["up"])
        #ctr.set_zoom(view_control["zoom"])
        #vis.update_renderer()
        #vis.poll_events()
        
        #create image path if not exist
        import os
        image_path_dir = os.path.dirname(image_path)
        if not os.path.exists(image_path_dir):
            os.makedirs(image_path_dir)

        vis.capture_screen_image(image_path, do_render=True)
        print("image saved to: 3DTrans/tools/"+image_path)
    vis.run()
    vis.destroy_window()
