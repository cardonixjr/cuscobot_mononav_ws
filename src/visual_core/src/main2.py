#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from VO.HandcraftDetector import HandcraftDetector
from VO.FrameByFrameMatcher import FrameByFrameMatcher
from VO.imageProcessing import *
from VO.tools import plot_keypoints

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
        self.absscale = 1.0

        # Camera parameters
        self.focal = None
        self.pp = ()

        # Frame index counter
        self.index = 0

        # Keypoint and descriptors
        self.kptdescs = {}

        # Current pose and frame
        self.cur_R = None
        self.cur_t = None
        
        # Ground truth
        self.cur_gt_pose = None
        self.last_gt_pose = None
        # self.vo_odom = []
        # self.vo_scaled_odom = []
        self.wheel_odom = []

        # Plotter
        self.traj_plotter = TrajPlotter()

        rospy.loginfo("VO node initialized")

    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.K).reshape(3,3)
            self.dist_coeffs = np.array(msg.D)

            # TODO: AJEITAR PARAMETROS DA CAMERA COM BASE NA LEITURA DO ROS
            self.focal = 490
            self.pp = (320, 240)
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
        
        if len(self.wheel_odom) == 0:
            return
        else:
            self.cur_gt_pose = self.wheel_odom[-1]
        
        # Converte a imagem recebida do ROS em formato BGR para opencv
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        input_img = self.image_processing(img)

        kptdesc = self.detector(input_img)

        # First frame
        if self.index == 0:
            self.kptdescs["cur"] = kptdesc

            # Start point
            self.cur_R = np.identity(3)
            self.cur_t = np.zeros((3,1))
        else:
            # Update keypoints and descriptors
            self.kptdescs["cur"] = kptdesc

            # Match keypoints
            matches = self.matcher(self.kptdescs)

            # Compute relative R, t between ref and cur frame
            E, mask = cv2.findEssentialMat(matches['cur_keypoints'], matches['ref_keypoints'],
                                           focal=self.focal, pp=self.pp,
                                           method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, mask = cv2.recoverPose(E, matches['cur_keypoints'], matches['ref_keypoints'],
                                            focal=self.focal, pp=self.pp)
            
            # get absolute pose based on absolute scale
            self.absscale = self.get_absscale()

            if(self.absscale > 0.1):
                self.cur_t = self.cur_t + self.absscale * self.cur_t.dot(t)
                self.cur_R = R.dot(self.cur_R)
            else:
                self.cur_t = t
                self.cur_R = R


            if self.cur_gt_pose is not None:
                kp_img = plot_keypoints(img, kptdesc["keypoints"], kptdesc["scores"]/kptdesc["scores"].max())
                cv2.imshow("Processed Image", input_img)
                cv2.imshow("Keypoints", kp_img)

                img2 = self.traj_plotter.update(t, self.cur_gt_pose)
                cv2.imshow("traj",img2)
                cv2.waitKey(1)

        self.last_gt_pose = self.cur_gt_pose
        self.kptdescs['ref'] = self.kptdescs['cur']
        self.index += 1        
        # return self.cur_R, self.cur_t

    def get_absscale(self):
        scale = 1.0
        print(self.cur_gt_pose)
        print(self.last_gt_pose)
        if self.cur_gt_pose is not None and self.last_gt_pose is not None:
            scale = np.sqrt(
                (self.cur_gt_pose[0] - self.last_gt_pose[0]) * (self.cur_gt_pose[0] - self.last_gt_pose[0])
                + (self.cur_gt_pose[1] - self.last_gt_pose[1]) * (self.cur_gt_pose[1] - self.last_gt_pose[1])
                + (self.cur_gt_pose[2] - self.last_gt_pose[2]) * (self.cur_gt_pose[2] - self.last_gt_pose[2]))
        print(f"scale: {scale}")
        return scale

    def image_processing(self, img):
        # TODO: desenvolver processo de melhoria de imagens
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray
    
    def spin(self):
        rospy.spin()
        cv2.destroyAllWindows()

class TrajPlotter(object):
    def __init__(self):
        self.errors = []
        self.traj = np.zeros((600, 600, 3), dtype=np.uint8)
        pass

    def update(self, est_xyz, gt_xyz):
        x, z = est_xyz[0], est_xyz[2]
        gt_x, gt_z = gt_xyz[0], gt_xyz[2]

        est = np.array([x, z]).reshape(2)
        gt = np.array([gt_x, gt_z]).reshape(2)

        error = np.linalg.norm(est - gt)

        self.errors.append(error)

        avg_error = np.mean(np.array(self.errors))

        # === drawer ==================================
        # each point
        draw_x, draw_y = int(x) + 290, int(z) + 90
        true_x, true_y = int(gt_x) + 290, int(gt_z) + 90

        # draw trajectory
        cv2.circle(self.traj, (draw_x, draw_y), 1, (0, 255, 0), 1)
        cv2.circle(self.traj, (true_x, true_y), 1, (0, 0, 255), 2)
        cv2.rectangle(self.traj, (10, 20), (600, 80), (0, 0, 0), -1)

        # draw text
        text = "[AvgError] %2.4fm" % (avg_error)
        cv2.putText(self.traj, text, (20, 40),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

        return self.traj
    
    
def keypoints_plot(img, vo):
    if img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return plot_keypoints(img, vo.kptdescs["cur"]["keypoints"], vo.kptdescs["cur"]["scores"])

if __name__ == "__main__":
    node = VisualOdometry()
    try:
        node.spin()
    except KeyboardInterrupt:
        node.save_csv()
        pass