"""
SUBSCRIBE IN ODOM, VISUAL ODOM AND STATISTICS ROS TOPICS AND WRITE THE DATA IN CSV FILES
"""

import rospy
from nav_msgs.msg import Odometry
import numpy as np
from threading import Lock
import time
import re

import matplotlib.pyplot as plt

from std_msgs.msg import String
import json

class PoseReader:
    def __init__(self):
        rospy.init_node('pose_reader')
        
        self.vo_statistics = []
        self.odom_positions = []
        self.visual_odom_positions = []
        self.lock = Lock()
        
        # Subscribe to topics
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/visual_odom', Odometry, self.visual_odom_callback)
        rospy.Subscriber('/vo_statistics', String, self.statistics_callback)
        
        rospy.on_shutdown(self.shutdown_callback)
        
        rospy.loginfo("Pose Reader initialized. Subscribing to /odom and /visual_odom")
    
    def shutdown_callback(self):
        rospy.loginfo("Saving trajectories...")
        self.save_trajectories()
        # self.plot_trajectories()

    def statistics_callback(self, msg):
        with self.lock:

            stats = json.loads(msg.data)

            self.vo_statistics.append([

                stats["timestamp"],

                stats["preprocessing_time"],
                stats["detection_time"],
                stats["matching_time"],
                stats["pose_estimation_time"],

                stats["fps"],

                stats["detected_keypoints"],
                stats["raw_matches"],
                stats["good_matches"],
                stats["inliners"],

                stats["match_ratio"],
                stats["inliner_ratio"],

                stats["cpu_usage"],
                stats["memory_usage"],
                float(re.sub(r'[^\d.]', '', stats["cpu_temperature"])),

                int(stats["tracking_ok"])

            ])


    def odom_callback(self, msg):
        with self.lock:
            t = msg.header.stamp.to_sec()
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            z = msg.pose.pose.position.z
            self.odom_positions.append([t, x, y, z])

    def visual_odom_callback(self, msg):
        with self.lock:
            t = msg.header.stamp.to_sec()
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            z = msg.pose.pose.position.z
            self.visual_odom_positions.append([t, x, y, z])
    
    def plot_trajectories(self):
        with self.lock:
            if len(self.odom_positions) == 0 and len(self.visual_odom_positions) == 0:
                rospy.logwarn("No trajectory data collected")
                return
            
            fig = plt.figure(figsize=(12, 5))
            
            # 3D plot
            ax1 = fig.add_subplot(121, projection='3d')
            
            if len(self.odom_positions) > 0:
                odom_array = np.array(self.odom_positions)
                ax1.plot(odom_array[:, 0], odom_array[:, 1], odom_array[:, 2], 'b-', label='Odometry')
            
            if len(self.visual_odom_positions) > 0:
                visual_array = np.array(self.visual_odom_positions)
                ax1.plot(visual_array[:, 0], visual_array[:, 1], visual_array[:, 2], 'r-', label='Visual Odometry')
            
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            ax1.legend()
            ax1.set_title('3D Trajectories')
            
            # 2D plot (XY)
            ax2 = fig.add_subplot(122)
            
            if len(self.odom_positions) > 0:
                odom_array = np.array(self.odom_positions)
                ax2.plot(odom_array[:, 0], odom_array[:, 1], 'b-', label='Odometry')
            
            if len(self.visual_odom_positions) > 0:
                visual_array = np.array(self.visual_odom_positions)
                ax2.plot(visual_array[:, 0], visual_array[:, 1], 'r-', label='Visual Odometry')
            
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.legend()
            ax2.set_title('2D Trajectories (XY)')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.show()
    
    def save_trajectories(self):
        # Save trajectories to CSV
        # timestamp = int(time.time())
        timestamp = "s_SIFT_RATIO_CLAHE_03"
        # Save trajectories to CSV
        odom_array = np.array(self.odom_positions)
        visual_array = np.array(self.visual_odom_positions)
        np.savetxt(f'trajetorias/odom_trajectory_{timestamp}.csv', odom_array, delimiter=',', header='timestamp,x,y,z', comments='')
        np.savetxt(f'trajetorias/visual_odom_trajectory_{timestamp}.csv', visual_array, delimiter=',', header='timestamp,x,y,z', comments='')
        
 

        stats_array = np.array(self.vo_statistics)
        np.savetxt(
            f"trajetorias/vo_statistics_{timestamp}.csv",
            stats_array,
            delimiter=",",
            header=(
                "timestamp,"
                "preprocessing_time,"
                "detection_time,"
                "matching_time,"
                "pose_estimation_time,"
                "fps,"
                "detected_keypoints,"
                "raw_matches,"
                "good_matches,"
                "inliners,"
                "match_ratio,"
                "inliner_ratio,"
                "cpu_usage,"
                "memory_usage,"
                "cpu_temperature,"
                "tracking_ok"
            ),
            comments=""
        )


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    reader = PoseReader()
    reader.run()
