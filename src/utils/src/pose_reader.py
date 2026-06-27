import rospy
from nav_msgs.msg import Odometry
import numpy as np
from threading import Lock
import time


import matplotlib.pyplot as plt

class PoseReader:
    def __init__(self):
        rospy.init_node('pose_reader')
        
        self.odom_positions = []
        self.visual_odom_positions = []
        self.lock = Lock()
        
        # Subscribe to topics
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/visual_odom', Odometry, self.visual_odom_callback)
        
        rospy.loginfo("Pose Reader initialized. Subscribing to /odom and /visual_odom")
    
    def odom_callback(self, msg):
        with self.lock:
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            z = msg.pose.pose.position.z
            self.odom_positions.append([x, y, z])
    
    def visual_odom_callback(self, msg):
        with self.lock:
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            z = msg.pose.pose.position.z
            self.visual_odom_positions.append([x, y, z])
    
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
        timestamp = int(time.time())
        # Save trajectories to CSV
        odom_array = np.array(self.odom_positions)
        visual_array = np.array(self.visual_odom_positions)
        np.savetxt(f'trajetorias/odom_trajectory_{timestamp}.csv', odom_array, delimiter=',', header='x,y,z', comments='')
        np.savetxt(f'trajetorias/visual_odom_trajectory_{timestamp}.csv', visual_array, delimiter=',', header='x,y,z', comments='')
    
    def run(self):
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Plotting trajectories...")
            #self.plot_trajectories()
            self.save_trajectories()

if __name__ == '__main__':
    reader = PoseReader()
    reader.run()