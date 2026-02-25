import rospy
from geometry_msgs.msg import Point, Pose, Twist, Vector3
from nav_msgs.msg import Odometry as Odom
import numpy as np
import time
import matplotlib.pyplot as plt
import goToGoal
import matplotlib.pyplot as plt

# Frequency in Hz
UPDATE_FREQUENCY = 1

# Robot info
WHEEL_RADIUS = 0.06
WHEEL_BASE = 0.37
TICKS_PER_REVOLUTION = 980
MAX_PWM = 40
MAX_PWM_STEP = 30           # Biggest PWM step made each loop.
MAX_SPEED_DISTANCE = 1      # Distance (meters) from goal before the robot start reducing its speed.


PATH = [[1,1],[0,0]]      # An array of x and y coordinates that the robot must follow
step = 0            # Index for the actual target
goal = PATH[step]   # Actual target coordinate


class lineFollower():
    def __init__(self):
        self.goal = goal
        self.path = PATH
        self.step = step

        # Update Frequency
        self.updateFrequencyPublish = UPDATE_FREQUENCY

        # Odometry
        self.x = 0
        self.y = 0
        self.theta = 0

        # PID Controller
        self.controller = goToGoal.GoToGoal()
        self.w = 0      # Angular speed
        self.last_w = 0

        # Time stamps
        self.start_time = time.time_ns()
        self.last_read = time.time()


        self.left_pwm = 128
        self.right_pwm = 128

        # aux
        self.is_first_loop = True


        ############################## ROS DEFINITION ##############################
        # ROS Node name to this class
        self.nodeName = "LineFollower"


        # ROS node
        rospy.init_node(self.nodeName, anonymous = True)
        self.nodeName = rospy.get_name()
        rospy.loginfo(f"The node - {self.nodeName} has started")

        # Subscribers for receiving encoder readings
        rospy.Subscriber("odom", Odom, self.callback_odom)

        # Publishers
        self.cmdVelPublisher = rospy.Publisher("cmd_vel", Twist, queue_size=10)
        self.ratePublisher = rospy.Rate(self.updateFrequencyPublish)


    def callback_odom(self, message1):
        ''' Callback function called when "odom" topic receive a message'''
        self.x = message1.pose.pose.position.x
        self.y = message1.pose.pose.position.y
        self.theta = message1.pose.pose.orientation.z

    def stop(self):
        self.cmdVelPublisher.publish(Twist(Vector3(0,0,0), Vector3(0,0,0)))

    def calculateUpdate(self):

        # Calculate how many ns passed since last read
        t = time.time_ns()
        dt = t - self.start_time
        self.start_time = t

        ############################## PID ##############################
        # Calculates the angular speed w
        self.w = self.controller.step(self.goal[0], self.goal[1], self.x, self.y, self.theta, dt, precision = 0.05)
        if self.w > 0.1: self.w = 0.1
        if self.w < -0.1: self.w = -0.1
        
        if self.w != None: 
            self.last_w = self.w

        twist_vel = Twist()
        twist_vel.linear.x = 0.08
        twist_vel.angular.z = self.w
        self.cmdVelPublisher.publish(twist_vel)

        # Check if reached the target
        # If reach the target, the controller will return None for angular speed
        # if this is the case, consider the last calculated speed
        if self.w == None: 
            self.w = self.last_w
            
            # Then, check if there is a next coordinate to go in the path
            # If the path continues, makes the next point the goal
            if self.step+1 < len(self.path):
                self.step += 1
                self.goal = self.path[self.step]
                self.w = self.last_w
                
            # If there is no more points in path, end the code
            else: return False

        # End the loop, keep running the code    
        return True

    def mainLoop(self):
        try:
            running = True

            while running and not rospy.is_shutdown():
                running = self.calculateUpdate()
                self.ratePublisher.sleep()

        except Exception as e:
            print(e)

        finally:
            # Stop the robot
            self.stop()


if __name__ == "__main__":
    """ main """
    follower = lineFollower()
    follower.mainLoop()

    # rospy.on_shutdown(function)
