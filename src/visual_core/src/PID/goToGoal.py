import math

class GoToGoal:
    def __init__(self):
        self.E_d = 0        # Error for the derivative term
        self.E_i = 0        # Error for the integrative term

        # PID gains
        self.Kp = 100
        self.Ki = 2
        self.Kd = 20

    def step(self, x_goal, y_goal, x, y, theta, dt, precision = 0.05):
        dt = dt/1000000000

        # Distance between goal and robot position in X
        u_x = x_goal - x

        # Distance between goal and robot position in Y
        u_y = y_goal - y

        if abs(u_x) < precision and abs(u_y) < precision:
            print("voce chegou ao seu destino")
            return None
        
        # Angle from robot to goal
        theta_goal = math.atan2(u_y, u_x)

        # Error between the goal angle and robot's angle
        e_k = theta_goal - theta
        e_k = math.atan2(math.sin(e_k), math.cos(e_k))

        # Error for the proportional term
        e_P = e_k

        # Error for the integral term
        e_I = self.E_i + e_k*dt

        # Error for the derivative term
        e_D = (e_k - self.E_d)/dt

        w = self.Kp * e_P + self.Ki * e_I + self.Kd * e_D
        # print("Distance from goal", '{:02.3f}'.format(u_x), '{:02.3f}'.format(u_y))
        # print("w: ", w)

        # Save errors for next time step
        self.E_i = e_I
        self.E_d = e_k

        return w
    
    def speed_limit_by_distance(self, max_distance, max_pwm, x_goal, y_goal, x, y):
        # Distance between goal and robot position in X
        u_x = x_goal - x

        # Distance between goal and robot position in Y
        u_y = y_goal - y

        euclidean_u = math.sqrt(u_y**2 + u_x**2)

        pwm = euclidean_u*100/max_distance
        
        return pwm if pwm <= max_pwm else max_pwm


def limit(value, downLimit, upLimit):
        return upLimit if value >= upLimit else downLimit if value <= downLimit else value

def rateLimit(value, ctrlVar, upLimit, downLimit):
        ctrlVar = limit(value, (downLimit + ctrlVar),(upLimit + ctrlVar))
        return ctrlVar