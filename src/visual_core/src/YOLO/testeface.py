# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 18:56:12 2025

@author: PC Gamer
"""

from ultralytics import YOLO
import cv2
import numpy as np
import pyvista as pv
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def plot_path(pos):
    
    line = pv.lines_from_points(pos)
    
    # Create scalar array for coloring
    scalars = np.linspace(0, 1, len(pos))
    
    
    line['order'] = scalars
    
    tube = line.tube(radius=0.03)
    
    plotter = pv.Plotter()
    plotter.add_mesh(tube, scalars='order', cmap='bwr', show_edges=False)
    plotter.show_grid()
    plotter.add_axes()
    
    plotter.add_mesh(pv.Cube(x_length = 0.1, y_length = 0.1, z_length = 0.1), color = 'red')
    plotter.add_point_labels(points = np.array([[0,0,0.12]]), labels = ['Camera'], font_size=16, text_color='black', shape_color='yellow')
    
    plotter.show()
    # point_cloud = pv.PolyData(np.array(pos))
    
    # # Create a scalar array representing the order
    # scalars = np.linspace(0, 1, len(pos))
    
    # # Add the scalar array to the point cloud
    # point_cloud['order'] = scalars
    
    # # Create a custom colormap from blue to red
    # cmap = pv.LookupTable()
    # cmap.value_range = (0, 1)
    # cmap.hue_range = (0.667, 0.0)  # Blue (0.667) to Red (0.0) in HSV
    
    # # Plot with custom colormap
    # plotter = pv.Plotter()
    # plotter.add_mesh(pv.Cube(x_length = 0.1, y_length = 0.1, z_length = 0.1), color = 'red')
    # plotter.add_point_labels(points = np.array([[0,0,0.12]]), labels = ['Camera'], font_size=16, text_color='black', shape_color='yellow')
    # plotter.add_points(point_cloud, scalars='order', cmap=cmap, 
    #                    point_size=10, render_points_as_spheres=True)
    
    # plotter.add_axes()
    # plotter.show_grid()
    # plotter.show()

def calculate_3d_position(object_pixel_x, object_pixel_y, object_pixel_height, real_height):
    """Calculate full 3D position of an object"""
    
    # Camera parameters (Logitech C270)
    
    # f_pixels = (focal_length / sensor_height) * image_height
    # f_pixels = (3.67 / 3.60) * 720 ≈ 1150 pixels
    
    
    f_pixels = 490
    image_center_x = 320  # For 1280x720
    image_center_y = 240
    
    # 1. Calculate distance using object height
    distance = (real_height * f_pixels) / object_pixel_height
    
    # 2. Calculate horizontal position
    lateral_pixels = object_pixel_x - image_center_x
    lateral_position = (lateral_pixels * distance) / f_pixels
    
    # 3. Calculate vertical position  
    vertical_pixels = image_center_y - object_pixel_y  # Flip Y-axis (image coordinates vs real world)
    vertical_position = (vertical_pixels * distance) / f_pixels
    
    return distance, lateral_position, vertical_position

def main():
    rospy.init_node('yolo_subscriber')
    model = YOLO("yolov8n.pt")
    model.to('cpu')
    target_name = "cell phone"
    target_id = next(k for k, v in model.names.items() if v == target_name)
    bridge = CvBridge()
    pos_list = []
    
    class FrameProcessor:
        def __init__(self):
            self.model = model
            self.target_id = target_id
            self.pos_list = pos_list
            self.bridge = bridge
            self.class_names = self.model.names

        def callback(self, msg):
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            frame = cv2.flip(frame, 1)
            results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[self.target_id])
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                class_ids = results[0].boxes.cls.int().cpu().tolist()
                confidences = results[0].boxes.conf.float().cpu().tolist()
                for box, track_id, class_id, conf in zip(boxes, track_ids, class_ids, confidences):
                    if class_id != self.target_id:
                        continue
                    x_center, y_center, width, height = box
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)
                    size = abs(y2 - y1)
                    distance, lateral, vertical = calculate_3d_position(x_center, y_center, size, 0.18)
                    self.pos_list.append([distance, lateral, vertical])
                    class_name = self.class_names[class_id]
                    print(f"Track ID: {track_id}, Class: {class_name}, Confidence: {conf:.2f}")
                    print(f"Position: Center({x_center:.1f}, {y_center:.1f}), Box[{x1}, {y1}, {x2}, {y2}]")
                    print(f"Size: {width:.1f}x{height:.1f}")
                    print("-" * 50)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} {track_id} ({conf:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow('YOLOv8 Object Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                rospy.signal_shutdown('User requested shutdown.')
                
    processor = FrameProcessor()
    rospy.Subscriber("/camera/image_raw", Image, processor.callback, queue_size=1, buff_size=2**24)
    print("Pressione Ctrl+C para sair.")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    plot_path(pos_list)
    

if __name__ == "__main__":
    main()
