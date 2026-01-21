import math

class odometry():
    '''
        Class that defines a odometry calculator. receives the general robot dimensions and MD49 encoders readings
        and calculates the actual coodinates of the robot (dead reckoning).
    '''

    def __init__(self, el, er, L):
        '''
            Initialize odometry
            
            el - left wheel encoder
            er - right wheel encoder
            L - distance  between the wheels in meters
            R - wheel radius
        '''
        self.el = el
        self.er = er
        self.L = L
        self.wl_last_counter = 0
        self.wr_last_counter = 0
        
        self.x = 0
        self.y = 0
        self.theta = 0

        self.v = 0
        self.w = 0
        
        # Calculate meters per tick
        self.meters_per_tick_left = (2 * math.pi * self.el.radius) / (el.ticks_p_revol)
        self.meters_per_tick_right = (2 * math.pi * self.er.radius) / (er.ticks_p_revol)

        # RPM speed
        self.wl_speed_rpm = (60/self.el.ticks_p_revol) * 10**9
        self.wr_speed_rpm = (60/self.er.ticks_p_revol) * 10**9


    def step(self, dt):
        '''
            Call this function periodically to update robot pose estimiation.
        '''
        # Calculate the delta for ticks since last read
        delta_ticks_left = (self.el.counter - self.wl_last_counter)
        delta_ticks_right = (self.er.counter - self.wr_last_counter)
        
        # Update counters to next read
        self.wl_last_counter = self.el.counter
        self.wr_last_counter = self.er.counter
        
        # Calculate new pose
        Dl = self.meters_per_tick_left * delta_ticks_left
        Dr = self.meters_per_tick_right * delta_ticks_right
        Dc = (Dr + Dl) / 2
        
        x_dt = Dc * math.cos(self.theta)
        y_dt = Dc * math.sin(self.theta)
        theta_dt = (Dr - Dl) / self.L

        self.x = self.x + x_dt
        self.y = self.y + y_dt
        self.theta = self.theta + theta_dt

#        # Calculate Angular Velocity
#        left_angular_speed = Dl/dt
#        right_angular_speed = Dr/dt

#        # Calculate linear velocity
#        left_linear_speed = self.el.radius*left_angular_speed
#        right_linear_speed = self.er.radius*right_angular_speed

        left_linear_speed = Dl/dt
        right_linear_speed = Dr/dt

        # Robot center velocity
        self.v = ((left_linear_speed + right_linear_speed)/2)*10**9

        #Robot angular velocity
        self.w = ((right_linear_speed - left_linear_speed)/self.L)*10**9
        
        #print(f"v: {self.v} , w: {self.w}")

        return self.x, self.y, self.theta
        
    def resetPose(self):
        self.x = 0
        self.y = 0
        self.theta = 0
    
    def getPose(self):
        return self.x, self.y, self.theta

    def getSpeed(self):
        return self.v, self.w
    
class speedEstimator():

    def __init__(self, el, er, R, L):
        '''
            Initialize speed estimator
            
            el - left wheel encoder
            er - right wheel encoder
            R - radious of the wheels in meters
            L - distance  between the wheels in meters
        '''
        self.el = el
        self.er = er
        self.R = R
        self.L = L
        self.wl_last_counter = 0
        self.wr_last_counter = 0
        
        # Speed converter for rpm
        self.wl_speed_rpm = (60/self.el.ticks_p_revol) * 10**9
        self.wr_speed_rpm = (60/self.er.ticks_p_revol) * 10**9

    def wheelSpeed(self, dt):
        '''
            Calculate wheel speeds.
            
            dt - time delta since last read (in ns)
        '''
        wl_delta_ticks = self.el.counter - self.wl_last_counter
        wr_delta_ticks = self.er.counter - self.wr_last_counter

        self.wl_last_counter = self.el.counter
        self.wr_last_counter = self.er.counter

        left_wheel_speed = wl_delta_ticks/dt * self.wl_speed_rpm
        right_wheel_speed = wr_delta_ticks/dt * self.wr_speed_rpm

        return (left_wheel_speed, right_wheel_speed)

    def robotSpeed(self, left_rpm, right_rpm):
        left_rad_s = math.pi/30 * left_rpm
        right_rad_s = math.pi/30 * right_rpm

        v = self.R/2 * (right_rad_s + left_rad_s)
        w = self.R/self.L * (right_rad_s - left_rad_s)

        return v, w

def uni_to_diff(v,w, el, er, L):

    vel_r = (2 * v + w * L)/(2 * er.radius)
    vel_l = (2 * v - w * L)/(2 * el.radius)
    
    return vel_l, vel_r
