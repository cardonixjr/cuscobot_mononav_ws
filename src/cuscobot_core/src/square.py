#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist

## Distancias percorridas
#SIDE_LENGTH = 1.0        # m
#TURN_ANGLE = 1.5708      # 90 graus (pi/2)
#
## Tempos necessários
#MOVE_TIME = 10.0        # Tempo para mover 1 metro (ajuste conforme necessário)
#TURN_TIME = 5.0         # Tempo para girar 90 graus (ajuste conforme necessário)
#offset = -0.375         # AJUSTE DE TEMPO DE FORÇA BRUTA PAR ACORRIGIR ERROS NA MOVIMENTAÇÃO (ajuste conforme necessário)
#
## Cálculo das velocidades
#LINEAR_SPEED = SIDE_LENGTH / MOVE_TIME
#ANGULAR_SPEED = TURN_ANGLE / TURN_TIME
#
## RESETA TEMPO ADICIONANDO O OFFSET PARA CORRIGIR ERROS DE MOVIMENTAÇÃO
#MOVE_TIME += offset
#TURN_TIME += offset


OFFSET = -0.375         # AJUSTE DE TEMPO DE FORÇA BRUTA PARA CORRIGIR ERROS NA MOVIMENTAÇÃO (ajuste conforme necessário)
def stop(pub):
    cmd = Twist()
    for _ in range(10):
        pub.publish(cmd)
        rospy.sleep(0.02)


def square_trajectory(pub):

    rate = rospy.Rate(100)

    move_time = 10.0      # 1 m
    turn_time = 3.14      # ajustar experimentalmente

    for _ in range(4):

        cmd = Twist()
        cmd.linear.x = 0.10

        start = rospy.get_time()

        while rospy.get_time() - start < move_time-OFFSET:
            pub.publish(cmd)
            rate.sleep()

        stop(pub)
        rospy.sleep(1.0)

        cmd = Twist()
        cmd.angular.z = 0.50

        start = rospy.get_time()

        while rospy.get_time() - start < turn_time-OFFSET:
            pub.publish(cmd)
            rate.sleep()

        stop(pub)
        rospy.sleep(1.0)

def circle_trajectory(pub):

    rate = rospy.Rate(100)

    cmd = Twist()

    cmd.linear.x = 0.10
    cmd.angular.z = 0.20

    duration = 31.4

    start = rospy.get_time()

    while rospy.get_time() - start < duration-OFFSET:
        pub.publish(cmd)
        rate.sleep()

    stop(pub)


def s_trajectory(pub):

    rate = rospy.Rate(100)

    cmd = Twist()

    cmd.linear.x = 0.10
    cmd.angular.z = 0.25

    start = rospy.get_time()

    while rospy.get_time() - start < 6.0:
        pub.publish(cmd)
        rate.sleep()

    cmd.angular.z = -0.25

    start = rospy.get_time()

    while rospy.get_time() - start < 6.0:
        pub.publish(cmd)
        rate.sleep()

    stop(pub)

def figure8_trajectory(pub):

    rate = rospy.Rate(100)

    linear_speed = 0.10
    angular_speed = 0.25

    cmd = Twist()
    cmd.linear.x = linear_speed

    # meia volta para a esquerda
    cmd.angular.z = angular_speed

    start = rospy.get_time()

    while rospy.get_time() - start < 12.57 + OFFSET:
        pub.publish(cmd)
        rate.sleep()

    # volta completa para a direita
    cmd.angular.z = -angular_speed

    start = rospy.get_time()

    while rospy.get_time() - start < 25.13 + OFFSET:
        pub.publish(cmd)
        rate.sleep()

    # meia volta para a esquerda
    cmd.angular.z = angular_speed

    start = rospy.get_time()

    while rospy.get_time() - start < 12.57 + OFFSET:
        pub.publish(cmd)
        rate.sleep()

    stop(pub)


if __name__ == "__main__":

    rospy.init_node("trajectory_test")

    pub = rospy.Publisher(
        "/cmd_vel",
        Twist,
        queue_size=10
    )

    rospy.sleep(1.0)

    # Escolha UMA:

    # quare_trajectory(pub)
    # circle_trajectory(pub)
    # s_trajectory(pub)
    figure8_trajectory(pub)
    
    rospy.loginfo("Teste concluído.")