import rospy
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from VO.loadData import load

main_path = "/home/cardonixjr/cuscobot_mononav_ws/src/visual_core/param"
config_file = "camera.yaml"

K, P = load(os.path.join(main_path, config_file))