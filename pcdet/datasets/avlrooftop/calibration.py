import numpy as np


class Calibration(object):
    def __init__(self, intrinsic, lidar_to_world, lidar_to_cam, shape_image):

        self.intrinsic = intrinsic
        self.lidar_to_world = lidar_to_world
        self.lidar_to_cam = lidar_to_cam

        # self.extrinsic = np.reshape(calibration['extrinsic'], [4, 4])
        # self.width = calibration['width']
        # self.height = calibration['height']
        # self.rolling_shutter_direction = calibration['rolling_shutter_direction']
        # self.metadata = calibration['metadata']
        self.shape_image = shape_image
        # self.pose_vehicle = calibration['pose_vehicle']

        # Camera intrinsics and extrinsics
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)

    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.concatenate((pts, np.ones((pts.shape[0], 1), dtype=np.float32)), axis=1)
        return pts_hom

    @property
    def vehicle_to_world(self):
        return self.lidar_to_world

    @property
    def camera_model(self):
        camera_model = np.array([
            [self.intrinsic[0], 0, self.intrinsic[1], 0],
            [0, self.intrinsic[2], self.intrinsic[3], 0],
            [0, 0, 1, 0]], dtype=np.float32)
        return camera_model

    # KITTI compatibility
    @property
    def P2(self):
        return self.camera_model

    # @property
    # def velo_to_cam_transform(self):
    #     axes_transformation = np.array([
    #         [0., -1., 0., 0.],
    #         [0., 0., -1., 0.],
    #         [1., 0., 0., 0.],
    #         [0., 0., 0., 1.]], dtype=np.float32)
    #
    #     # vehicle to camera transformation
    #     vehicle_to_camera = np.matmul(axes_transformation, np.linalg.inv(self.extrinsic))
    #
    #     return vehicle_to_camera

    # KITTI compatibility
    @property
    def Tr_velo_to_cam(self):
        return self.lidar_to_cam

    @property
    def cam_to_velo_transform(self):
        return np.linalg.inv(self.lidar_to_cam)

    def rect_to_lidar(self, pts_rect):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)  # (N, 4)
        pts_lidar = np.einsum('ij,kj->ki', self.cam_to_velo_transform, pts_rect_hom)
        return pts_lidar[:, 0:3]

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        pts_rect = np.einsum('ij,kj->ki', self.lidar_to_cam, pts_lidar_hom)
        return pts_rect[:, 0:3]

    def lidar_to_world(self, pts_lidar):
        pts_lidar_h = self.cart_to_hom(pts_lidar)
        return np.einsum('ij,kj->ki', self.vehicle_to_world, pts_lidar_h)[..., :3]

    def world_to_lidar(self, pts_lidar):
        pts_lidar_h = self.cart_to_hom(pts_lidar)
        return np.einsum('ij,kj->ki', np.linalg.inv(self.vehicle_to_world), pts_lidar_h)[..., :3]

    def rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_2d_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]  # depth in rect camera coord

        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth
