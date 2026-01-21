from VO.loadData import load

import rospy
import os
import numpy as np
import cv2

import matplotlib.pyplot as plt

from tqdm import tqdm
# from lib.visualization import plotting
# from lib.visualization.video import play_trip

def form_transf(R, t):
    """
    Makes a transformation matrix from the given rotation matrix and translation vector
    Parameters
    ----------
    R (ndarray): The rotation matrix
    t (list): The translation vector
    Returns
    -------
    T (ndarray): The transformation matrix
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def get_matches(orb, flann, i1, i2):
    """
    This function detect and compute keypoints and descriptors from the i-1'th and i'th image using the class orb object
    Parameters
    ----------
    i (int): The current frame
    Returns
    -------
    q1 (ndarray): The good keypoints matches position in i-1'th image
    q2 (ndarray): The good keypoints matches position in i'th image
    """
    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(i1, None)
    kp2, des2 = orb.detectAndCompute(i2, None)
    # Find matches
    matches = flann.knnMatch(des1, des2, k=2)
    # Find the matches there do not have a to high distance
    good = []
    try:
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good.append(m)
    except ValueError:
        pass
    draw_params = dict(matchColor = -1, # draw matches in green color
             singlePointColor = None,
             matchesMask = None, # draw only inliers
             flags = 2)
    img3 = cv2.drawMatches(i2, kp1, i1,kp2, good ,None,**draw_params)
    cv2.imshow("image", img3)
    cv2.waitKey(200)
    # Get the image points form the good matches
    q1 = np.float32([kp1[m.queryIdx].pt for m in good])
    q2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return q1, q2

def get_pose(q1, q2, K):
    """
    Calculates the transformation matrix
    Parameters
    ----------
    q1 (ndarray): The good keypoints matches position in i-1'th image
    q2 (ndarray): The good keypoints matches position in i'th image
    Returns
    -------
    transformation_matrix (ndarray): The transformation matrix
    """
    # Essential matrix
#     print(K)
    # print(f"Number of points1: {q1.shape}")
    # print(f"Number of points2: {q2.shape}")
    # print(f"type q1: {type(q1)}")
    # print(f"type q2: {type(q2)}")
    E, _ = cv2.findEssentialMat(q1, q2, K, threshold=1)
    # Decompose the Essential matrix into R and t
    R, t = decomp_essential_mat(E, q1, q2, K, P)
    # Get transformation matrix
    transformation_matrix = form_transf(R, np.squeeze(t))
    return transformation_matrix, E

def decomp_essential_mat(E, q1, q2, K, P):
    """
    Decompose the Essential matrix
    Parameters
    ----------
    E (ndarray): Essential matrix
    q1 (ndarray): The good keypoints matches position in i-1'th image
    q2 (ndarray): The good keypoints matches position in i'th image
    Returns
    -------
    right_pair (list): Contains the rotation matrix and translation vector
    """
    def sum_z_cal_relative_scale(R, t, K, P):
        # Get the transformation matrix
        T = form_transf(R, t)
        # Make the projection matrix
        nP = np.matmul(np.concatenate((K, np.zeros((3, 1))), axis=1), T)
        # Triangulate the 3D points
        hom_Q1 = cv2.triangulatePoints(P, nP, q1.T, q2.T)
        # Also seen from cam 2
        hom_Q2 = np.matmul(T, hom_Q1)
        # Un-homogenize
        uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
        uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]
        # Find the number of points there has positive z coordinate in both cameras
        sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
        sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)
        # Form point pairs and calculate the relative scale
        relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)/
                                 np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
        return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale
    # Decompose the essential matrix
    
    R1, R2, t = cv2.decomposeEssentialMat(E)
    t = np.squeeze(t)
    # Make a list of the different possible pairs
    pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]
    # Check which solution there is the right one
    z_sums = []
    relative_scales = []
    for R, t in pairs:
        z_sum, scale = sum_z_cal_relative_scale(R, t, K, P)
        z_sums.append(z_sum)
        relative_scales.append(scale)
    # Select the pair there has the most points with positive z coordinate
    right_pair_idx = np.argmax(z_sums)
    right_pair = pairs[right_pair_idx]
    relative_scale = relative_scales[right_pair_idx]
    R1, t = right_pair
    t = t * relative_scale
    return [R1, t]


main_path = "/home/cardonixjr/cuscobot_ws/src/cuscobot_mononav/param"
config_file = "camera.yaml"

K, P = load(os.path.join(main_path, config_file))
        
        # self.gt_poses = self._load_poses(os.path.join(self.main_path,"poses.txt"))
        # self.images = self._load_images(os.path.join(self.main_path,"image_l"))
        


# from VO.HandcraftDetector import HandcraftDetector
# from VO.FrameByFrameMatcher import FrameByFrameMatcher

# detector = HandcraftDetector({"type": "SIFT"})
# matcher = FrameByFrameMatcher({"type": "FLANN"})

detector = cv2.SIFT_create(3000)
#FLANN_INDEX_LSH = 6
FLANN_INDEX_KDTREE = 1

index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
matcher = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

cap = cv2.VideoCapture(0)

frame = 1
prev_frame = None

# # Pose acumulada (R, t)
# R_f = np.eye(3)
# t_f = np.zeros((3, 1))

trajectory = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if prev_frame is not None:
        q1, q2 = get_matches(detector, matcher,prev_frame,frame)
        transf, E = get_pose(q1, q2, K)
        #cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
        _, R, t, mask_pose = cv2.recoverPose(E, q2, q1, K)

        print(R)
        print(t)
        print("---------------")
        
        # # Acumular pose
        # t_f = t_f + R_f @ t
        # R_f = R @ R_f

        # trajectory.append(t_f.copy())
        

    cv2.imshow("frame", frame)    

    prev_frame = frame    
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    

    
    if cv2.waitKey(1) & 0xFF == 27:  # Pressione ESC para sair
        break

# traj = np.array(trajectory).squeeze()  # vira N x 3

# plt.plot(traj[:,0], traj[:,2])  # x vs z (coordenadas horizontais no KITTI)
# plt.xlabel("X")
# plt.ylabel("Z")
# plt.title("Trajetória estimada")
# plt.axis('equal')
# plt.show()
