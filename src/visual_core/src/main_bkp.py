#!/usr/bin/env python3
import rospy
import cv2
import csv
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo
import matplotlib.pyplot as plt
from cv_bridge import CvBridge

from VO.imageProcessing import *
from VO.tools import plot_keypoints, image_processing
from VO.HandcraftDetector import HandcraftDetector
from VO.FrameByFrameMatcher import FrameByFrameMatcher

DEBUG = True
PLOT = True

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

#        self.detector = cv2.SIFT_create()

#        FLANN_INDEX_KDTREE = 1
#        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
#        search_params = dict(checks=50)
#        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

        self.detector = HandcraftDetector({"type": "SIFT"})
        self.matcher = FrameByFrameMatcher({"type": "FLANN"})

        self.kptdescs = {}

        # AUX variables
        self.last_image = None
        self.last_kp = None
        self.last_des = None
        self.current_pose = np.eye(4)
        self.vo_odom = []
        self.vo_scaled_odom = []
        self.wheel_odom = []

        self.absolute_scale = 1.0

        rospy.loginfo("VO node initialized")
    
    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.K).reshape(3,3)
            self.dist_coeffs = np.array(msg.D)
            rospy.loginfo("Camera calibration received")

    def odom_callback(self, msg):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        pz = msg.pose.pose.position.z
        self.wheel_odom.append(np.array([px,py,pz]))

        if DEBUG: print(f"ODOM RECEIVED x: {px} y:{py} z:{pz} ")

    def image_callback(self, msg):
        if self.camera_matrix is None:
            rospy.loginfo("Camera calibration not found")
            return
        
        if not msg:
            rospy.loginfo("Fail to receive image")
            return

        # Converte a imagem recebida do ROS em formato BGR para opencv
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Aplica o pré-processamento de imagem
        input_img = image_processing(img)

        # kp_dict = self.detector(input_img)

        # kp = kp_dict["keypoints"]
        # des = kp_dict["descriptors"]

        # Aplica a detecção de keypoints
        # kp, des = self.detector.detectAndCompute(input_img, None)
        kptdesc = self.detector(input_img)

        kp = kptdesc["keypoints"]
        des = kptdesc["descriptors"]

        if self.last_image is None:
            self.kptdescs["cur"] = kptdesc

        if self.last_image is not None: # and len(self.wheel_odom) > 1:    # Verifica se não é o primeiro loop  
            self.kptdescs["ref"] = self.kptdescs["cur"]
            self.kptdescs["cur"] = kptdesc 

            # Faz o matching entre os keypoints do ultimo frame com o atual
            matches = self.matcher(self.kptdescs)

            #good = matches["matches"]

            # Separa apenas os matches considerados bons
            #good = [m for m,n in matches if m.distance < 0.7*n.distance]

            # img_matches = np.empty((max(self.last_image.shape[0], input_img.shape[0]), self.last_image.shape[1] + input_img.shape[1], 3), dtype=np.uint8)
            # cv2.drawMatches(self.last_image, self.last_kp, input_img, kp, good, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
            # cv2.imshow('Good Matches', img_matches)
            # cv2.waitKey(50)


            if len(matches) > 8:
                # Coleta os pontos bons do frame atual e do ultimo frame
                # pts1 = np.float32([self.last_kp[m.queryIdx].pt for m in good])
                # pts2 = np.float32([kp[m.trainIdx].pt for m in good])

                # Encontra a matriz essencial
                E, mask = cv2.findEssentialMat(matches['cur_keypoints'], matches['ref_keypoints'], self.camera_matrix, cv2.RANSAC, 0.999, 1.0)
                
                # print(E)
                if E is not None:

                    # Recupera a pose do ultimo frame
                    _, R, t, _ = cv2.recoverPose(E, matches['cur_keypoints'], matches['ref_keypoints'], self.camera_matrix)

                    # Encontra a pose atual
                    T = np.eye(4)
                    T[:3,:3] = R
                    T[:3,3]  = t.flatten()
                    self.current_pose = self.current_pose @ np.linalg.inv(T)

                    p_vo = self.current_pose[:3,3]
                    self.vo_odom.append(p_vo.copy())

                    # Aplica a escala
                    # self.absolute_scale = self.get_scale()
                    self.absolute_scale = 0.1
                    p_vo_scaled = p_vo * self.absolute_scale

                    print(f"pvo: {p_vo}")
                    print(f"pvo_scaled: {p_vo_scaled}")
                        
                    self.vo_scaled_odom.append(p_vo_scaled.copy())


            if PLOT:
                kp_img = plot_keypoints(img, kptdesc["keypoints"], kptdesc["scores"] / kptdesc["scores"].max())
                cv2.imshow("Processed Image", input_img)
                cv2.imshow("Keypoints", kp_img)
                cv2.waitKey(1)

        # Reseta variaveis
        self.last_image = input_img
        self.last_kp = kp
        self.last_des = des

    def get_scale(self):
        scale = 1.0
        if len(self.vo_odom) >1:
            d_vo = np.linalg.norm(self.vo_odom[-1] - self.vo_odom[-2])
            d_odom = np.linalg.norm(self.wheel_odom[-1] - self.wheel_odom[-2])
            if d_vo > 1e-6:
                scale = d_odom / d_vo
            

        # if DEBUG: print(f"scale: {scale}")
        return scale
    
    def form_transf(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T



    def save_csv(self):
        # # salvar VO
        # with open("vo_trajectory.csv", "w", newline="") as f:
        #     w = csv.writer(f)
        #     w.writerow(["x","y","z"])
        #     w.writerows(self.vo_odom)

        # salvar VO escalada
        with open("vo_scaled_trajectory.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x","y","z"])
            w.writerows(self.vo_scaled_odom)

        # salvar odom
        with open("wheel_trajectory.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x","y","z"])
            w.writerows(self.wheel_odom)

    def plot_results(self):
        # vo = np.array(self.vo_odom)
        vos = np.array(self.vo_scaled_odom)
        od = np.array(self.wheel_odom)

        plt.figure()
        # if len(vo)>0:
        #     plt.plot(vo[:,0], vo[:,2], label="VO (sem escala)")
        if len(vos)>0:
            plt.plot(vos[:,0], vos[:,1], label="VO (escalada)")
        if len(od)>0:
            plt.plot(od[:,0], od[:,2], label="Wheel Odometry")

        plt.title("Trajetórias")
        plt.legend()
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.grid()
        plt.show()

        vos = np.array(self.vo_scaled_odom)
        od = np.array(self.wheel_odom)

        plt.figure()
        # if len(vo)>0:
        #     plt.plot(vo[:,0], vo[:,2], label="VO (sem escala)")
        if len(vos)>0:
            plt.plot(vos[:,0], -vos[:,1], label="VO XY (escalada) negativo")
        if len(od)>0:
            plt.plot(od[:,0], od[:,2], label="Wheel Odometry XZ")

        plt.title("Trajetórias")
        plt.legend()
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.grid()
        plt.show()

    def spin(self):
        rospy.spin()
        self.save_csv()
        self.plot_results()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    node = VisualOdometry()
    try:
        node.spin()
    except KeyboardInterrupt:
        node.save_csv()
        pass
