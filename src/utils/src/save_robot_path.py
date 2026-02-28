#!/usr/bin/env python3
import rospy
import csv
import os
import rospkg
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path

class PathRecorder:
    def __init__(self):
        rospy.init_node("save_robot_path")

        # Parâmetros configuráveis
        self.pose_topic = rospy.get_param("~pose_topic", "/amcl_pose")
        self.path_topic = rospy.get_param("~path_topic", "/robot_path")
        # Hardcode save directory to utils_package/trajetorias next to scripts
        self.save_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "trajetorias"))
        self.save_interval = rospy.get_param("~save_interval", 1.0)  # segundos

        # Cria pasta de saída se não existir
        os.makedirs(self.save_dir, exist_ok=True)
        timestamp = rospy.get_time()
        self.csv_path = os.path.join(self.save_dir, f"trajetoria_robo_{int(timestamp)}.csv")

        # Inicializa publicador do Path
        self.path_pub = rospy.Publisher(self.path_topic, Path, queue_size=10)
        self.path_msg = Path()
        self.path_msg.header.frame_id = "map"

        # Inicializa CSV
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["time", "x", "y", "z", "qx", "qy", "qz", "qw"])

        # Subscreve ao tópico de pose (com ou sem covariância)
        if self.pose_topic.endswith("amcl_pose"):
            rospy.Subscriber(self.pose_topic, PoseWithCovarianceStamped, self.pose_with_cov_callback)
        else:
            rospy.Subscriber(self.pose_topic, PoseStamped, self.pose_callback)


        self.last_save_time = rospy.get_time()
        rospy.loginfo(f"[PathRecorder] Gravando trajetória em {self.csv_path}")
        rospy.loginfo(f"[PathRecorder] Tópico de pose: {self.pose_topic}")
        rospy.loginfo(f"[PathRecorder] Publicando trajetória em: {self.path_topic}")

    def pose_callback(self, msg):
        now = rospy.get_time()

        # Armazena pose no Path
        pose_stamped = PoseStamped()
        pose_stamped.header = msg.header
        pose_stamped.pose = msg.pose
        self.path_msg.poses.append(pose_stamped)
        self.path_msg.header.stamp = rospy.Time.now()
        self.path_pub.publish(self.path_msg)

        # Salva no CSV de forma periódica
        if now - self.last_save_time >= self.save_interval:
            self.csv_writer.writerow([
                msg.header.stamp.to_sec(),
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w
            ])
            self.csv_file.flush()
            self.last_save_time = now

    def pose_with_cov_callback(self, msg):
        # Converte PoseWithCovarianceStamped → PoseStamped e reusa a lógica existente
        pose_stamped = PoseStamped()
        pose_stamped.header = msg.header
        pose_stamped.pose = msg.pose.pose
        self.pose_callback(pose_stamped)


    def shutdown_hook(self):
        rospy.loginfo(f"[PathRecorder] Salvando e fechando {self.csv_path}")
        self.csv_file.close()

if __name__ == "__main__":
    recorder = PathRecorder()
    rospy.on_shutdown(recorder.shutdown_hook)
    rospy.spin()
