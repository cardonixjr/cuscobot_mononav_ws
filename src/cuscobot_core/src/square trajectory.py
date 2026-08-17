#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist

# Distancias percorridas
SIDE_LENGTH = 1.0        # m
TURN_ANGLE = 1.5708      # 90 graus (pi/2)

# Tempos necessários
MOVE_TIME = 15.0        # Tempo para mover 1 metro (ajuste conforme necessário)
TURN_TIME = 10.0         # Tempo para girar 90 graus (ajuste conforme necessário)
offset = -1.6         # AJUSTE DE TEMPO DE FORÇA BRUTA PAR ACORRIGIR ERROS NA MOVIMENTAÇÃO (ajuste conforme necessário)

# Cálculo das velocidades
LINEAR_SPEED = SIDE_LENGTH / MOVE_TIME
ANGULAR_SPEED = TURN_ANGLE / TURN_TIME

# RESETA TEMPO ADICIONANDO O OFFSET PARA CORRIGIR ERROS DE MOVIMENTAÇÃO
MOVE_TIME += offset
TURN_TIME += offset

def stop(pub):
    msg = Twist()
    pub.publish(msg)


def move_forward(pub):
    msg = Twist()
    msg.linear.x = LINEAR_SPEED

    start = rospy.Time.now().to_sec()

    rate = rospy.Rate(20)

    while (rospy.Time.now().to_sec() - start) < MOVE_TIME + 1.3:
        pub.publish(msg)
        rate.sleep()

    stop(pub)


def turn_left(pub):
    msg = Twist()
    msg.angular.z = ANGULAR_SPEED

    start = rospy.Time.now().to_sec()

    rate = rospy.Rate(20)

    while (rospy.Time.now().to_sec() - start) < TURN_TIME:
        pub.publish(msg)
        rate.sleep()

    stop(pub)

def turn_right(pub):
    msg = Twist()
    msg.angular.z = -ANGULAR_SPEED

    start = rospy.Time.now().to_sec()

    rate = rospy.Rate(20)

    while (rospy.Time.now().to_sec() - start) < TURN_TIME:
        pub.publish(msg)
        rate.sleep()

    stop(pub)


if __name__ == "__main__":

    rospy.init_node("square_trajectory")
    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

    try:
    

        rospy.sleep(1.0)

        move_forward(pub)

        stop(pub)
    except KeyboardInterrupt:
        rospy.loginfo("Keyboard interrupted, shutting down.")
        stop(pub)

    rospy.loginfo("Trajetoria concluida.")

