#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from VO.HandcraftDetector import HandcraftDetector
from VO.FrameByFrameMatcher import FrameByFrameMatcher
from VO.tools import plot_keypoints, plot_results, plot_results_3d, plot_pose, image_processing, save_csv

class VisualOdometry(object):

    def __init__(self):
        # Initialize rospy
        rospy.init_node("visual_odom")

        # Subscriber
        rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback)
        rospy.Subscriber("/usb_cam/camera_info", CameraInfo, self.camera_info_callback)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)

        # Opencv camera config
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None

        # Feature detector and keypoint matcher
        self.detector = HandcraftDetector({"type": "SIFT"})
        self.matcher = FrameByFrameMatcher({"type": "FLANN"})
        self.absscale = 0.05

        # Frame index counter
        self.index = 0

        # Keypoint and descriptors
        self.kptdescs = {}

        # Current pose and frame
        self.cur_R = None
        self.cur_t = None

        start_pose = np.ones((3,4))
        start_translation = np.zeros((3,1))
        start_rotation = np.identity(3)
        start_pose = np.concatenate((start_rotation, start_translation), axis=1)
        self.cur_pose = start_pose
        
        # # Ground truth
        # self.cur_gt_pose = None
        # self.last_gt_pose = None

        self.pose_list = []
        self.vo_odom = []
        self.wheel_odom = []

        self.last_image = None

        rospy.loginfo("VO node initialized")

    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.K).reshape(3,3)
            self.extrinsic = np.array(((1,0,0,0),(0,1,0,0),(0,0,1,0)))
            self.P = self.camera_matrix @ self.extrinsic
            self.dist_coeffs = np.array(msg.D)
            rospy.loginfo("Camera calibration received")

    def odom_callback(self, msg):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        pz = msg.pose.pose.position.z
        self.wheel_odom.append(np.array([px,py,pz]))

        # print(f"ODOM RECEIVED x: {px} y:{py} z:{pz} ")

    def image_callback(self, msg):
        if self.camera_matrix is None:
            rospy.loginfo("Camera calibration not found")
            return
        
        if not msg:
            rospy.loginfo("Fail to receive image")
            return
        
        # Aplica o processamento de imagens
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        input_img = image_processing(img)

        # Aplica a detecção de keypoints no frame atual
        kptdesc = self.detector(input_img)
        
        # First frame
        if self.index == 0:
            self.kptdescs["cur"] = kptdesc

            # Start point
            self.cur_R = np.identity(3)
            self.cur_t = np.zeros((3,1))
        
        else:
            # Atualiza os keypoints anteriores
            self.kptdescs["ref"] = self.kptdescs["cur"]

            # Atualiza os keypoints atuais
            self.kptdescs["cur"] = kptdesc

            # Match keypoints
            matches = self.matcher(self.kptdescs)

            # img_matches = np.empty((max(self.last_image.shape[0], input_img.shape[0]), self.last_image.shape[1] + input_img.shape[1], 3), dtype=np.uint8)
            # cv2.drawMatches(self.last_image, matches['ref_keypoints'], input_img, matches['cur_keypoints'], matches, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            # cv2.imshow('Good Matches', img_matches)

            # Get Absscale
            # self.absscale = self.get_absscale()
            self.absscale = 0.05

            # Get transf. matrix
            transf = self.get_pose(matches['cur_keypoints'], matches['ref_keypoints'])
            
            self.cur_pose = self.cur_pose @ transf
            # self.cur_pose *= self.absscale

            hom_array = np.array([[0,0,0,1]])
            hom_camera_pose = np.concatenate((self.cur_pose,hom_array), axis=0)
            self.pose_list.append(hom_camera_pose)

            p_vo = self.cur_pose[:3,3]
            self.vo_odom.append(p_vo.copy())

            
            kp_img = plot_keypoints(img, kptdesc["keypoints"], kptdesc["scores"]/kptdesc["scores"].max())
            cv2.imshow("Processed Image", input_img)
            cv2.imshow("Keypoints", kp_img)   

        self.kptdescs['ref'] = self.kptdescs['cur']
        self.last_image = input_img
        self.index += 1   

        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return
            # raise KeyboardInterrupt
    
    def get_absscale(self):
        # print(self.cur_gt_pose)
        # print(self.last_gt_pose)
        cur_gt = None
        last_gt = None

        if len(self.wheel_odom > 1):
            cur_gt = self.wheel_odom[-1]
            last_gt = self.wheel_odom[-2]

        if cur_gt is not None and last_gt is not None:
            scale = np.sqrt(
                (cur_gt[0] - last_gt[0]) * (cur_gt[0] - last_gt[0])
                + (cur_gt[1] - last_gt[1]) *(cur_gt[1] - last_gt[1])
                + (cur_gt[2] - last_gt[2]) *(cur_gt[2] - last_gt[2]))
        print(f"scale: {scale}")

        if scale <= 1e-5: scale = 1.0
        return scale

    def get_pose(self, q1, q2):
    
        # Essential matrix
        E, mask = cv2.findEssentialMat(q1, q2, self.camera_matrix)

        # Decompose the Essential matrix into R and t
        R, t = self.decomp_essential_mat_old(E, q1, q2)

        if(self.absscale > 0.1):
            self.cur_t = self.cur_t + self.absscale * self.cur_t.dot(t)
            self.cur_R = R.dot(self.cur_R)
        else:
            self.cur_t = t
            self.cur_R = R

        # Get transformation matrix
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
                
        return T
    
    def form_transf(self, R, t):
        
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        
        return T


    def decomp_essential_mat(self, E, q1, q2):

        R1, R2, t = cv2.decomposeEssentialMat(E)
        T1 = self.form_transf(R1,np.ndarray.flatten(t))
        T2 = self.form_transf(R2,np.ndarray.flatten(t))
        T3 = self.form_transf(R1,np.ndarray.flatten(-t))
        T4 = self.form_transf(R2,np.ndarray.flatten(-t))
        transformations = [T1, T2, T3, T4]
        
        # Homogenize K
        K = np.concatenate((self.camera_matrix, np.zeros((3,1)) ), axis = 1)

        # List of projections
        projections = [K @ T1, K @ T2, K @ T3, K @ T4]

        np.set_printoptions(suppress=True)

        # print ("\nTransform 1\n" +  str(T1))
        # print ("\nTransform 2\n" +  str(T2))
        # print ("\nTransform 3\n" +  str(T3))
        # print ("\nTransform 4\n" +  str(T4))

        positives = []
        for P, T in zip(projections, transformations):
            hom_Q1 = cv2.triangulatePoints(P, P, q1.T, q2.T)
            hom_Q2 = T @ hom_Q1
            # Un-homogenize
            Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            Q2 = hom_Q2[:3, :] / hom_Q2[3, :]
             
            total_sum = sum(Q2[2, :] > 0) + sum(Q1[2, :] > 0)
            
            dQ1 = np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1)
            dQ2 = np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1)
            mask = dQ2 > 1e-8  # threshold para evitar divisões por zero
            if np.any(mask):
                relative_scale = np.mean(dQ1[mask] / dQ2[mask])
            else:
                relative_scale = 0.1  # ou outro valor padrão

            positives.append(total_sum + relative_scale)
            

        # Decompose the Essential matrix using built in OpenCV function
        # Form the 4 possible transformation matrix T from R1, R2, and t
        # Create projection matrix using each T, and triangulate points hom_Q1
        # Transform hom_Q1 to second camera using T to create hom_Q2
        # Count how many points in hom_Q1 and hom_Q2 with positive z value
        # Return R and t pair which resulted in the most points with positive z

        max = np.argmax(positives)
        if (max == 2):
            # print(-t)
            return R1, np.ndarray.flatten(-t)
        elif (max == 3):
            # print(-t)
            return R2, np.ndarray.flatten(-t)
        elif (max == 0):
            # print(t)
            return R1, np.ndarray.flatten(t)
        elif (max == 1):
            # print(t)
            return R2, np.ndarray.flatten(t)
        
        
    def decomp_essential_mat_old(self, E, q1, q2):
        def sum_z_cal_relative_scale(R, t):
            # Get the transformation matrix
            T = self.form_transf(R, t)
            # Make the projection matrix
            P = np.matmul(np.concatenate((self.camera_matrix, np.zeros((3, 1))), axis=1), T)

            # Triangulate the 3D points
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            # Also seen from cam 2
            hom_Q2 = np.matmul(T, hom_Q1)

            # Un-homogenize
            Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            # Find the number of points there has positive z coordinate in both cameras
            sum_of_pos_z_Q1 = sum(Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(Q2[2, :] > 0)

            # Form point pairs and calculate the relative scale
            # Form point pairs and calculate the relative scale
            dQ1 = np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1)
            dQ2 = np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1)
            mask = dQ2 > 1e-8  # threshold para evitar divisões por zero
            if np.any(mask):
                relative_scale = np.mean(dQ1[mask] / dQ2[mask])
            else:
                relative_scale = 0.1  # ou outro valor padrão
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
            z_sum, scale = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        # Select the pair there has the most points with positive z coordinate
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        t = t * relative_scale
        
        T = self.form_transf(R1, t)
        # Make the projection matrix
        P = np.matmul(np.concatenate((self.camera_matrix, np.zeros((3, 1))), axis=1), T)

        # Triangulate the 3D points
        hom_Q1 = cv2.triangulatePoints(P, P, q1.T, q2.T)
        # Also seen from cam 2
        hom_Q2 = np.matmul(T, hom_Q1)

        # Un-homogenize
        Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
        Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

        return [R1, t]

    def spin(self):
        rospy.spin()
        save_csv(self.wheel_odom, self.vo_odom)
        plot_results(self.wheel_odom, self.vo_odom)
        plot_pose(self.pose_list, self.camera_matrix)
        plot_results_3d(self.wheel_odom, self.vo_odom)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    node = VisualOdometry()
    try:
        node.spin()
    except KeyboardInterrupt:
        # node.save_csv()
        pass