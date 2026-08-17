#!/usr/bin/env python3

import time
import rospy
import cv2
import numpy as np
import psutil
import os
import json
from scipy.spatial.transform import Rotation
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from geometry_msgs.msg import Quaternion

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
        self.statistics_pub = rospy.Publisher("vo_statistics", String, queue_size=10)

        # Opencv camera config
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None

        ############# CONFIGURAÇÕES DE TESTE #############
        self.detector_type = "ORB" # "ORB" "SIFT"
        self.matcher_type = "FLANN" # "FLANN" "BF_RATIO" "BF_CROSS"
        self.use_ransac = True
        ##################################################

        self.distance_ratio = 0.65

        if self.detector_type == "ORB":
            # ------- ORB -------
            rospy.loginfo("Using ORB detector")
            self.detector = cv2.ORB_create(nfeatures=2000,
                scaleFactor=1.2, #1.2
                nlevels=10, #8
                edgeThreshold=31,
                firstLevel=0,
                WTA_K=2,
                patchSize=35,
                fastThreshold=40) #2
            
            # ------- FLANN -------
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=12, key_size=12, multi_probe_level=1)
            self.norm = cv2.NORM_HAMMING


        elif self.detector_type == "SIFT":
            # ------- SIFT -------
            rospy.loginfo("Using SIFT detector")
            self.detector = cv2.SIFT_create(nfeatures=1500, #1000
                nOctaveLayers=3,
                contrastThreshold=0.02, #0.04
                edgeThreshold=12, #10
                sigma=1.6
                )
            
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
            self.norm = cv2.NORM_L2


        # ------- FLANN -------
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

        # ---------- BF Ratio ----------
        self.bf = cv2.BFMatcher(self.norm)

        # ---------- BF Cross ----------
        self.bf_cross = cv2.BFMatcher(
            self.norm,
            crossCheck=True
        )

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

        self.cur_pose = np.eye(4)

        self.alpha = 0.5

        self.tx_f = 0.0
        self.ty_f = 0.0
        self.yaw_f = 0.0

        self.filter_initialized = False
        # Acumula as odometrias
        self.vo_odom = []
        self.wheel_odom = []
        self.scaled_vo_odom = []

        self.last_image = None

        #AUX
        self.absscale_sum = 0
        self.absscale_count = 0


        if self.matcher_type == "FLANN": rospy.loginfo("Using FLANN matcher")
        elif self.matcher_type == "BF_RATIO": rospy.loginfo("Using BF matcher with ratio test")
        elif self.matcher_type == "BF_CROSS": rospy.loginfo("Using BF matcher with cross check")

        rospy.loginfo("VO node initialized")

    def match_features(self, des1, des2):
        if des1 is None or des2 is None:
            return []

        # ===================================
        # FLANN + Lowe Ratio
        # ===================================
        if self.matcher_type == "FLANN":
            matches = self.flann.knnMatch(des1, des2, k=2)
            good = []
            for pair in matches:
                if len(pair) != 2:
                    continue

                m, n = pair
                if m.distance < self.distance_ratio * n.distance:
                    good.append(m)
            return matches, good

        # ===================================
        # BF + Lowe Ratio
        # ===================================
        elif self.matcher_type == "BF_RATIO":
            matches = self.bf.knnMatch(des1, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < self.distance_ratio * n.distance:
                    good.append(m)
            return matches, good

        # ===================================
        # BF Cross Check
        # ===================================
        elif self.matcher_type == "BF_CROSS":
            matches = self.bf_cross.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            good = matches[:len(matches)//2]  # Keep only the best half of matches
            return matches, matches
    
    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.K).reshape(3,3)
            self.extrinsic = np.array(((1,0,0,0),(0,1,0,0),(0,0,1,0)))
            self.P = self.camera_matrix @ self.extrinsic
            self.dist_coeffs = np.array(msg.D)
            rospy.loginfo("Camera calibration received")

    def odom_callback(self, msg):
        self.wheel_odom.append(msg.pose.pose.position)
        # print(f"ODOM RECEIVED {msg.pose.pose.position} ")

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
        # input_img = img
        t1 = time.perf_counter() # Tempo de processamento da imagem

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

            self.cur_wheel = self.wheel_odom[-1] if len(self.wheel_odom) > 0 else 0
            

        else:
            # Atualiza os keypoints anteriores
            self.last_kpts = self.cur_kpts
            self.last_desc = self.cur_desc

            # Atualiza a odometria das rodas neste momento
            self.last_wheel = self.cur_wheel
            self.cur_wheel = self.wheel_odom[-1] if len(self.wheel_odom) >0 else 0

            if self.last_wheel == self.cur_wheel: 
                parado = True
                print("parado")
            else: parado = False
            
            # Atualiza os keypoints atuais
            self.cur_desc = desc
            self.cur_kpts = kpts
               
            # Get Matches
            q1, q2, matches, good = self.get_matches(input_img)
            t3 = time.perf_counter() # Tempo de inferência dos matches

            # Get Pose
            if not parado:
                T, mask = self.get_pose(q1, q2)
            else:
                T = np.eye(4)
                mask = None

            self.cur_pose = self.cur_pose @ T
            #self.cur_pose = self.cur_pose @ np.linalg.inv(T)
            hom_array = np.array([[0,0,0,1]])
            hom_camera_pose = np.concatenate((self.cur_pose,hom_array), axis=0)


            t4 = time.perf_counter() # Tempo de cálculo da pose

            # Acumula os resultados
            self.vo_odom.append(self.cur_pose[:3, 3])

            
            q = Rotation.from_matrix(self.cur_pose[:3,:3]).as_quat()

            # Publish visual odometry
            visual_odom = Odometry()
            visual_odom.header.stamp = msg.header.stamp
            visual_odom.header.frame_id = "odom"
            visual_odom.child_frame_id = "base_link"
            visual_odom.pose.pose.position.x = self.cur_pose[0, 3]
            visual_odom.pose.pose.position.y = self.cur_pose[1, 3]
            visual_odom.pose.pose.position.z = self.cur_pose[2, 3]
            visual_odom.pose.pose.orientation.x = q[0]
            visual_odom.pose.pose.orientation.y = q[1]
            visual_odom.pose.pose.orientation.z = q[2]
            visual_odom.pose.pose.orientation.w = q[3]
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
            inline_ratio = inliners / n_good_matches if n_good_matches > 0 else 0
            processing_time = t1 - t0
            detection_time = t2 - t1
            matching_time = t3 - t2
            pose_estimation_time = t4 - t3
            

            # Publish statistics
            statistics = {
                "timestamp": rospy.Time.now().to_sec(),
                "preprocessing_time": processing_time,
                "detection_time": detection_time,
                "matching_time": matching_time,
                "pose_estimation_time": pose_estimation_time,
                "fps": fps,
                "detected_keypoints": n_kpts,
                "raw_matches": n_matches,
                "good_matches": n_good_matches,
                "inliners": inliners,
                "match_ratio": match_ratio,
                "inliner_ratio": inline_ratio,
                "cpu_usage": cpu,
                "memory_usage": ram,
                "cpu_temperature": temp,
                "tracking_ok": (n_good_matches > 15)
            }
            status_msg = String()
            status_msg.data = json.dumps(statistics)
            self.statistics_pub.publish(status_msg)

        self.last_image = input_img
        self.index += 1
        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return
            # raise KeyboardInterrupt

    def get_matches(self, input_img):
            # Match keypoints
            if len(self.last_kpts) > 6 and len(self.cur_kpts) > 6:
                
                matches, good = self.match_features(self.last_desc, self.cur_desc,)

                if len(good) < 7:
                    rospy.logwarn("Not enough good matches found: %d", len(good))
                    return None, None, matches, good

                # Draw matches
                # img_matches = np.empty((max(self.last_image.shape[0], input_img.shape[0]), self.last_image.shape[1] + input_img.shape[1], 3), dtype=np.uint8)
                # cv2.drawMatches(self.last_image, self.last_kpts, input_img, self.cur_kpts, good_matches, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
                # cv2.imshow('Good Matches', img_matches)
                # cv2.waitKey(50)

                q1 = np.float32([self.last_kpts[m.queryIdx].pt for m in good])
                q2 = np.float32([self.cur_kpts[m.trainIdx].pt for m in good])

                return q1, q2, matches, good
            else:
                return None, None, None, None

    def get_pose(self, q1, q2): 
        # # Essential matrix

        if self.use_ransac:
            E, mask = cv2.findEssentialMat(q1, q2, self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        else:
            E, mask = cv2.findEssentialMat(q1, q2, self.camera_matrix, method=cv2.LMEDS)

        inliers1 = q1[mask.ravel() == 1]
        inliers2 = q2[mask.ravel() == 1]

        ## Decompose the Essential matrix into R and t
        R, t = self.decomp_essential_mat_old(E, inliers1, inliers2)
        # _, R, t, mask = cv2.recoverPose(E, inliers1, inliers2, self.camera_matrix)

        # Aplica a escala
        absscale = self.get_absscale()
        #absscale = 0.006732
        
        t = t * absscale
        # print(t)

        R_cam_to_base = np.array([
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0]
        ])
        
        R_base = R_cam_to_base @ R @ R_cam_to_base.T
        t_base = R_cam_to_base @ t

        # Remove Roll e Pitch, deixa só Yaw
        #roll, pitch, yaw = Rotation.from_matrix(R_base).as_euler('xyz')
        #R_base = Rotation.from_euler(
        #        'z',
        #        yaw
        #    ).as_matrix()
        
        # Aplica suavização para reduzir o erro proveniente de ruido:
        yaw = np.arctan2(R_base[1,0], R_base[0,0])
        
        tx = t_base[0]
        ty = t_base[1]
        
        if not self.filter_initialized:
            self.tx_f = tx
            self.ty_f = ty
            self.yaw_f = yaw
        
            self.filter_initialized = True
        else:
            a = self.alpha
        
            self.tx_f = a*tx + (1-a)*self.tx_f
            self.ty_f = a*ty + (1-a)*self.ty_f
            self.yaw_f = a*yaw + (1-a)*self.yaw_f
        
        cy = np.cos(self.yaw_f)
        sy = np.sin(self.yaw_f)
        
        R_base = np.array([
            [ cy, -sy, 0],
            [ sy,  cy, 0],
            [  0,   0, 1]
        ])
        
        t_base = np.array([
            self.tx_f,
            self.ty_f,
            0.0
        ])
        
        T = self.form_transf(R_base, np.squeeze(t_base))
        
        return T, mask

    def decomp_essential_mat_old(self, E, q1, q2):
        def find_z_sum(R, t):
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

            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2
        
        # Decompose the essential matrix
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)

        # Make a list of the different possible pairs
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        # Check which solution there is the right one
        z_sums = []
        for R, t in pairs:
            z_sum= find_z_sum(R, t)
            z_sums.append(z_sum)

        # Select the pair there has the most points with positive z coordinate
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        R1, t = right_pair

        return [R1, t]

    def form_transf(self, R, t):
        
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        
        return T

    def get_absscale(self):
        if self.absscale_count > 0:
            avg_scale = self.absscale_sum / self.absscale_count
            # rospy.loginfo(f"Average scale: {avg_scale:.6f}")
        else: self.avg_scale = 0

        if self.cur_wheel and self.last_wheel:
            dx = self.cur_wheel.x - self.last_wheel.x
            dy = self.cur_wheel.y - self.last_wheel.y

            scale = np.sqrt(dx*dx + dy*dy)

            scale = scale if scale > 1e-4 else 0

            # rospy.loginfo(f"Current scale: {scale:.6f}")
            if scale != 0:
                self.absscale_sum += scale
                self.absscale_count += 1

            return scale
        else:
            return avg_scale

    def spin(self):
        # name_mod = "ORB_0404_03"

        rospy.spin()


if __name__ == "__main__":
    node = VisualOdometry()
    try:
        node.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Keyboard interrupted, shutting down.")
        pass

