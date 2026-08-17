#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist

# Dimensões da trajetória
DIAMETER = 1          # m
RADIUS = DIAMETER / 2.0 # 0.5 m
ARC_ANGLE = 3.14159265  # 180 graus (pi)

# Tempo desejado para realizar a semicircunferência
ARC_TIME = 20.0         # segundos - ajuste conforme necessário

# Ajuste de tempo para corrigir erros de movimentação
offset = -2.3

# Velocidades
LINEAR_SPEED = (3.14159265 * RADIUS) / ARC_TIME
ANGULAR_SPEED = ARC_ANGLE / ARC_TIME

# Aplica o offset
ARC_TIME += offset


def stop(pub):
    msg = Twist()
    pub.publish(msg)


def semicircle_left(pub):
    msg = Twist()

    # Movimento circular:
    # v = omega * R
    msg.linear.x = LINEAR_SPEED
    msg.angular.z = ANGULAR_SPEED

    start = rospy.Time.now().to_sec()

    rate = rospy.Rate(20)

    while (rospy.Time.now().to_sec() - start) < ARC_TIME:
        pub.publish(msg)
        rate.sleep()

    stop(pub)


def semicircle_right(pub):
    msg = Twist()

    # Movimento circular para a direita
    msg.linear.x = LINEAR_SPEED
    msg.angular.z = -ANGULAR_SPEED

    start = rospy.Time.now().to_sec()

    rate = rospy.Rate(20)

    while (rospy.Time.now().to_sec() - start) < ARC_TIME:
        pub.publish(msg)
        rate.sleep()

    stop(pub)


if __name__ == "__main__":

    rospy.init_node("semicircle_trajectory")

    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

    try:

        rospy.sleep(1.0)

        # Faz uma semicircunferência de 1 m de diâmetro
        semicircle_right(pub)

        stop(pub)

    except KeyboardInterrupt:
        rospy.loginfo("Keyboard interrupted, shutting down.")
        stop(pub)

    rospy.loginfo("Trajetoria semicircular concluida.")