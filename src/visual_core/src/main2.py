#!/usr/bin/env python3
import rospy
import cv2
import csv
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo
import matplotlib.pyplot as plt
from cv_bridge import CvBridge


class VisualOdometryROS:
    def __init__(self):
        rospy.init_node("vo_sift_flann")

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None

        rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback)
        rospy.Subscriber("/usb_cam/camera_info", CameraInfo, self.camera_info_callback)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)

        self.sift = cv2.SIFT_create()
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        self.last_image = None
        self.last_kp = None
        self.last_des = None

        self.current_pose = np.eye(4)
        self.vo_traj = []
        self.vo_scaled_traj = []
        self.wheel_traj = []

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
        self.wheel_traj.append(np.array([px,py,pz]))

    def image_callback(self, msg):
        if self.camera_matrix is None:
            return

        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        input_img = img_gray
        kp, des = self.sift.detectAndCompute(input_img, None)

        if self.last_image is not None and len(self.wheel_traj) > 1:
            matches = self.flann.knnMatch(self.last_des, des, k=2)
            good = [m for m,n in matches if m.distance < 0.7*n.distance]

            if len(good) > 8:
                pts1 = np.float32([self.last_kp[m.queryIdx].pt for m in good])
                pts2 = np.float32([kp[m.trainIdx].pt for m in good])

                E, mask = cv2.findEssentialMat(pts2, pts1, self.camera_matrix, cv2.RANSAC, 0.999, 1.0)
                if E is not None:
                    _, R, t, _ = cv2.recoverPose(E, pts2, pts1, self.camera_matrix)

                    T = np.eye(4)
                    T[:3,:3] = R
                    T[:3,3]  = t.flatten()
                    self.current_pose = self.current_pose @ np.linalg.inv(T)

                    p_vo = self.current_pose[:3,3]
                    self.vo_traj.append(p_vo.copy())

                    # ---- ESCALA ----
                    if len(self.vo_traj) > 1:
                        d_vo = np.linalg.norm(self.vo_traj[-1] - self.vo_traj[-2])
                        d_odom = np.linalg.norm(self.wheel_traj[-1] - self.wheel_traj[-2])

                        if d_vo > 1e-6:
                            scale = d_odom / d_vo
                        else:
                            scale = 1.0

                        p_vo_scaled = p_vo * scale
                        self.vo_scaled_traj.append(p_vo_scaled.copy())

                    kp_img = cv2.drawKeypoints(img, kp, None, color=(0,255,0))

                
                    cv2.imshow("Image", img)
                    cv2.imshow("Processed Image", input_img)
                    cv2.imshow("Keypoints", kp_img)
                    cv2.waitKey(1)

        self.last_image = img_gray
        self.last_kp = kp
        self.last_des = des

    def save_csv(self):
        # # salvar VO
        # with open("vo_trajectory.csv", "w", newline="") as f:
        #     w = csv.writer(f)
        #     w.writerow(["x","y","z"])
        #     w.writerows(self.vo_traj)

        # salvar VO escalada
        with open("vo_scaled_trajectory.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x","y","z"])
            w.writerows(self.vo_scaled_traj)

        # salvar odom
        with open("wheel_trajectory.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x","y","z"])
            w.writerows(self.wheel_traj)

    def plot_results(self):
        # vo = np.array(self.vo_traj)
        vos = np.array(self.vo_scaled_traj)
        od = np.array(self.wheel_traj)

        plt.figure()
        # if len(vo)>0:
        #     plt.plot(vo[:,0], vo[:,2], label="VO (sem escala)")
        if len(vos)>0:
            plt.plot(vos[:,0], vos[:,2], label="VO (escalada)")
        if len(od)>0:
            plt.plot(od[:,0], od[:,2], label="Wheel Odometry")

        plt.title("Trajetórias")
        plt.legend()
        plt.xlabel("X (m)")
        plt.ylabel("Z (m)")
        plt.grid()
        plt.show()

    def spin(self):
        rospy.spin()
        self.save_csv()
        self.plot_results()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    node = VisualOdometryROS()
    try:
        node.spin()
    except KeyboardInterrupt:
        node.save_csv()
        pass
