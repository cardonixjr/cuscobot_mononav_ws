#!/usr/bin/env python3

import time
import rospy
import cv2
import numpy as np
import psutil
import os
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from geometry_msgs.msg import Quaternion
from visual_core.msg import VOStatistics

from VO.tools import plot_results, plot_results_3d, image_processing, save_csv

class VisualOdometry(object):
    def __init__(self):
        # Initialize rospy
        rospy.init_node("visual_odom")

        # Subscriber
        rospy.Subscriber("/camera/image_raw", Image, self.image_callback)
        rospy.Subscriber("/camera/camera_info", CameraInfo, self.camera_info_callback)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)

        # Publisher
        self.visual_odom_pub = rospy.Publisher("visual_odom", Odometry, queue_size=10)
        self.statistics_pub = rospy.Publisher("vo_statistics", VOStatistics, queue_size=10)

        # Opencv camera config
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None

        # Detection
        self.distance_ratio = 0.65 #0.75

        # ------- ORB -------

        self.detector = cv2.ORB_create(nfeatures=1500,
            scaleFactor=1.5, #1.2
            nlevels=10, #8
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            patchSize=31,
            fastThreshold=7) #2
            
        # FLANN MATCHER 
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        # absscale
        self.absscale = 0.25


        # # ------- SIFT -------
        # self.detector = cv2.SIFT_create(nfeatures=1500, #1000
        #     nOctaveLayers=3,
        #     contrastThreshold=0.02, #0.04
        #     edgeThreshold=12, #10
        #     sigma=1.6
        #     )
        # 
        # FLANN_INDEX_KDTREE = 1
        # index_params = dict(algorithm=FLANN_INDEX_KDTREE, 
        #                     trees=10 #5
        #                     )
        # search_params = dict(checks=100) #50 # or pass empty dictionary
        # self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        # # absscale
        # self.absscale = 0.55

        # Frame index counter
        self.index = 0

        # Keypoint and descriptors
        self.kptdescs = {}

        # Current pose and frame
        self.cur_R = None
        self.cur_t = None

        self.last_kpts = []
        self.last_desc = []
        self.cur_kpts = []
        self.cur_desc = []

        self.cur_gt = []
        self.last_gt = []

        start_pose = np.ones((3,4))
        start_translation = np.zeros((3,1))

        # O eixo de rotação inicial da câmera possui "z" apontando para frente, "x" para a direita e "y" para baixo,
        # enquanto o eixo de rotação do robô possui "x" apontando para frente, "y" para a esquerda e "z" para cima. 
        # Portanto, é necessário realizar uma transformação entre os dois sistemas de coordenadas.
        base_to_camera_rotation = np.array([
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0]
        ])
        start_pose = np.concatenate((base_to_camera_rotation, start_translation), axis=1)
        self.cur_pose = start_pose

        # Acumula as poses
        self.pose_list = []
        self.scaled_pose_list = []

        # Acumula as odometrias
        self.vo_odom = []
        self.wheel_odom = []
        self.scaled_vo_odom = []

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
        
        t0 = time.perf_counter() # Tempo inicial do processo

        # Aplica o processamento de imagens
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        input_img = image_processing(img)
        t1 = time.perf_counter() # Tempo de processamento da imagem


        # rospy.loginfo("Image received")

        # Aplica a detecção de keypoints no frame atual
        kpts, desc = self.detector.detectAndCompute(input_img, None)
        t2 = time.perf_counter() # Tempo de detecção de keypoints

        scores = np.zeros((len(kpts)))
        for i, p in enumerate(kpts):
            scores[i] = p.response

        if self.index == 0:
            # Atualiza os keypoints anteriores
            self.cur_kpts = kpts
            self.cur_desc = desc
            self.last_image = input_img

            self.cur_gt = self.wheel_odom[-1] if len(self.wheel_odom) >0 else None
            self.last_gt = self.cur_gt

        else:
            # Atualiza os keypoints anteriores
            self.last_kpts = self.cur_kpts
            self.last_desc = self.cur_desc

            # Atualiza os keypoints atuais
            self.cur_desc = desc
            self.cur_kpts = kpts
               
            # Get Matches
            q1, q2, matches, good = self.get_matches(input_img, ratio = self.distance_ratio)
            t3 = time.perf_counter() # Tempo de inferência dos matches

            transformation, mask = self.get_pose(q1, q2)

            self.cur_pose = self.cur_pose @ transformation
            hom_array = np.array([[0,0,0,1]])
            hom_camera_pose = np.concatenate((self.cur_pose,hom_array), axis=0)

            t4 = time.perf_counter() # Tempo de cálculo da pose

            # Acumula os resultados
            self.pose_list.append(hom_camera_pose)
            self.vo_odom.append(self.cur_pose[:3, 3])
            
            # self.absscale = self.get_absscale()

            self.scaled_pose_list.append(hom_camera_pose * self.absscale)
            self.scaled_vo_odom.append(self.cur_pose[:3, 3] * self.absscale)


            # Publish visual odometry
            visual_odom = Odometry()
            visual_odom.header.stamp = msg.header.stamp
            visual_odom.header.frame_id = "odom"
            visual_odom.child_frame_id = "base_link"
            visual_odom.pose.pose.position.x = self.cur_pose[0, 3] * self.absscale
            visual_odom.pose.pose.position.y = self.cur_pose[1, 3] * self.absscale
            visual_odom.pose.pose.position.z = self.cur_pose[2, 3] * self.absscale
            visual_odom.pose.pose.orientation = Quaternion(0, 0, 0, 1)
            self.visual_odom_pub.publish(visual_odom)


            # Get environment statistics
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            temp = os.popen("vcgencmd measure_temp").read()
            fps = 1.0 / (t4 - t0)
            n_kpts = len(self.cur_kpts)
            n_matches = len(matches) if matches is not None else 0
            n_good_matches = len(good) if good is not None else 0
            inliners = np.count_nonzero(mask) if mask is not None else 0
            match_ratio = n_good_matches / n_matches if n_matches > 0 else 0
            inline_ratio = inliners / len(n_good_matches) if n_good_matches > 0 else 0
            processing_time = t1 - t0
            detection_time = t2 - t1
            matching_time = t3 - t2
            pose_estimation_time = t4 - t3
            

            # Publish statistics
            status = VOStatistics()
            status.header.stamp = time.time()
            status.preprocessing_time = processing_time
            status.detection_time = detection_time
            status.matching_time = matching_time
            status.pose_estimation_time = pose_estimation_time
            status.fps = fps
            status.detected_keypoints = n_kpts
            status.raw_matches = n_matches
            status.good_matches = n_good_matches
            status.inliners = inliners
            status.match_ratio = match_ratio
            status.inline_ratio = inline_ratio
            status.cpu_usage = cpu
            status.memory_usage = ram
            status.cpu_temperature = temp
            status.tracking_ok = len(n_good_matches) > 15
            self.statistics_pub.publish(status)


        self.cur_gt = self.wheel_odom[-1] if len(self.wheel_odom) >0 else None
        self.last_gt = self.cur_gt
        self.last_image = input_img
        self.index += 1
        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return
            # raise KeyboardInterrupt

    def get_matches(self, input_img, ratio = 0.75):
            # Match keypoints
            if len(self.last_kpts) > 6 and len(self.cur_kpts) > 6:
                matches = self.matcher.knnMatch(self.last_desc, self.cur_desc, k=2)

                # Find the matches there do not have a to high distance
                good_matches = []
                try:
                    for m, n in matches:
                        if m.distance < ratio * n.distance:
                            good_matches.append(m)
                except ValueError:
                    pass

                # Draw matches
                # img_matches = np.empty((max(self.last_image.shape[0], input_img.shape[0]), self.last_image.shape[1] + input_img.shape[1], 3), dtype=np.uint8)
                # cv2.drawMatches(self.last_image, self.last_kpts, input_img, self.cur_kpts, good_matches, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
                # cv2.imshow('Good Matches', img_matches)
                # cv2.waitKey(50)

                q1 = np.float32([self.last_kpts[m.queryIdx].pt for m in good_matches])
                q2 = np.float32([self.cur_kpts[m.trainIdx].pt for m in good_matches])

                return q1, q2, matches, good_matches
            else:
                return None, None, None

    def get_pose(self, q1, q2): 
        # # Essential matrix
        E, mask = cv2.findEssentialMat(q1, q2, self.camera_matrix)

        # Decompose the Essential matrix into R and t
        R, t = self.decomp_essential_mat_old(E, q1, q2)

        R_cam_to_base = np.array([
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0]
        ])

        R_base = R_cam_to_base @ R @ R_cam_to_base.T
        t_base = R_cam_to_base @ t

        T = self.form_transf(R_base, np.squeeze(t_base))

        return T, mask

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
            
            #self.world_points.append(Q1)

            # Find the number of points there has positive z coordinate in both cameras
            sum_of_pos_z_Q1 = sum(Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(Q2[2, :] > 0)

            # Form point pairs and calculate the relative scale
            relative_scale = np.mean(np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1)/
                                     np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1))
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
        # t = t * relative_scale
        t = t * 0.05
        
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

    def form_transf(self, R, t):
        
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        
        return T

    def get_absscale(self):
        scale = self.last_scale

        if self.cur_gt is not None and self.last_gt is not None:
            scale = np.sqrt(
                (self.cur_gt[0] - self.last_gt[0]) * (self.cur_gt[0] - self.last_gt[0])
                + (self.cur_gt[1] - self.last_gt[1]) * (self.cur_gt[1] - self.last_gt[1])
                + (self.cur_gt[2] - self.last_gt[2]) * (self.cur_gt[2] - self.last_gt[2]))
            self.last_scale = scale
        # print(f"scale: {scale}")
        return scale

    def spin(self):
        # name_mod = "ORB_0404_03"

        rospy.spin()
        # save_csv(self.wheel_odom, self.scaled_vo_odom,name_modifier=name_mod) 
        # plot_results(self.wheel_odom, self.scaled_vo_odom, name_modifier=name_mod)
        # plot_pose(self.pose_list, self.camera_matrix)
        # plot_results_3d(self.wheel_odom, self.scaled_vo_odom, name_modifier=name_mod)
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    node = VisualOdometry()
    try:
        node.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Keyboard interrupted, shutting down.")
        pass
