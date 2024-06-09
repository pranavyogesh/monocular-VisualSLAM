import os
import numpy as np
import cv2
from tqdm import tqdm
from lib.visualization import plotting
from lib.visualization.video import play_trip

class VisualOdometry():
    def __init__(self, data_dir):
        self.K, self.P = self._load_calib(os.path.join(data_dir, 'calib.txt'))
        self.gt_poses = self._load_poses(os.path.join(data_dir, 'poses.txt'))
        self.images = self._load_images(os.path.join(data_dir, 'image_l'))
        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        self.point_cloud = []  # Initialize an empty list to accumulate 3D points

    @staticmethod
    def _load_calib(filepath):
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P

    @staticmethod
    def _load_poses(filepath):
        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses

    @staticmethod
    def _load_images(filepath):
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

    @staticmethod
    def _form_transf(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, i):
        keypoints1, descriptors1 = self.orb.detectAndCompute(self.images[i - 1], None)
        keypoints2, descriptors2 = self.orb.detectAndCompute(self.images[i], None)
        matches = self.flann.knnMatch(descriptors1, descriptors2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                good.append(m)
        q1 = np.float32([keypoints1[m.queryIdx].pt for m in good])
        q2 = np.float32([keypoints2[m.trainIdx].pt for m in good])
        return q1, q2

    def get_pose(self, q1, q2):
        Essential, mask = cv2.findEssentialMat(q1, q2, self.K)
        R, t = self.decomp_essential_mat(Essential, q1, q2)
        return self._form_transf(R, t)

    def decomp_essential_mat(self, E, q1, q2):
        R1, R2, t = cv2.decomposeEssentialMat(E)
        T1 = self._form_transf(R1, np.ndarray.flatten(t))
        T2 = self._form_transf(R2, np.ndarray.flatten(t))
        T3 = self._form_transf(R1, np.ndarray.flatten(-t))
        T4 = self._form_transf(R2, np.ndarray.flatten(-t))
        transformations = [T1, T2, T3, T4]

        K = np.concatenate((self.K, np.zeros((3, 1))), axis=1)
        projections = [K @ T1, K @ T2, K @ T3, K @ T4]

        positives = []
        for P, T in zip(projections, transformations):
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            hom_Q2 = T @ hom_Q1
            Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            total_sum = sum(Q2[2, :] > 0) + sum(Q1[2, :] > 0)
            relative_scale = np.mean(np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1) /
                                     np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1))
            positives.append(total_sum + relative_scale)

        max = np.argmax(positives)
        if max == 2:
            return R1, np.ndarray.flatten(-t)
        elif max == 3:
            return R2, np.ndarray.flatten(-t)
        elif max == 0:
            return R1, np.ndarray.flatten(t)
        elif max == 1:
            return R2, np.ndarray.flatten(t)

    def accumulate_3d_points(self, q1, q2, cur_pose):
        """
        Triangulates and accumulates 3D points from the matched keypoints.

        Parameters
        ----------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image
        cur_pose (ndarray): The current camera pose

        Returns
        -------
        None
        """
        P1 = self.P
        P2 = self.K @ cur_pose[:3, :]
        points_4d_hom = cv2.triangulatePoints(P1, P2, q1.T, q2.T)
        points_3d = points_4d_hom[:3, :] / points_4d_hom[3, :]
        self.point_cloud.extend(points_3d.T.tolist())

    def save_point_cloud(self, filename):
        """
        Saves the accumulated 3D points as a .ply file.

        Parameters
        ----------
        filename (str): The file path to save the .ply file

        Returns
        -------
        None
        """
        points = np.array(self.point_cloud)
        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for point in points:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")

def main():
    data_dir = 'KITTI_sequence_1'  # Try KITTI_sequence_2 too
    vo = VisualOdometry(data_dir)
    play_trip(vo.images)  # Comment out to not play the trip

    gt_path = []
    estimated_path = []
    for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="pose")):
        if i == 0:
            cur_pose = gt_pose
        else:
            q1, q2 = vo.get_matches(i)
            transf = vo.get_pose(q1, q2)
            vo.accumulate_3d_points(q1, q2, cur_pose)
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
            print("\nGround truth pose:\n" + str(gt_pose))
            print("\nCurrent pose:\n" + str(cur_pose))
            print("The current pose used x,y: \n" + str(cur_pose[0, 3]) + "   " + str(cur_pose[2, 3]))
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))

    vo.save_point_cloud(os.path.basename(data_dir) + ".ply")
    plotting.visualize_paths(gt_path, estimated_path, "Visual Odometry", file_out=os.path.basename(data_dir) + ".html")

if __name__ == "__main__":
    main()
