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
from scipy.spatial import distance as dist
from collections import defaultdict

class CentroidTracker:
    """Rastreador de objetos baseado em centroide - funciona com OpenVINO"""
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = {}
        self.disappeared = defaultdict(int)
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        """
        Atualiza o tracker com detecções atuais.
        rects: lista de [x_center, y_center] dos objetos detectados
        """
        if len(rects) == 0:
            # Se nenhuma detecção, marcar como desaparecido
            disappearedIDs = list(self.disappeared.keys())
            for objectID in disappearedIDs:
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        # Calcular centroide das detecções atuais
        inputCentroids = np.array(rects)
        
        if len(self.objects) == 0:
            # Primeiro frame com detecções
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            # Associar detecções com objetos rastreados
            objectIDs = list(self.objects.keys())
            objectCentroids = np.array(list(self.objects.values()))
            
            # Calcular distâncias
            D = dist.cdist(objectCentroids, inputCentroids)
            
            # Encontrar correspondências com menor distância
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            usedRows = set()
            usedCols = set()
            
            # Atualizar centros dos objetos rastreados
            for row, col in zip(rows, cols):
                if col in usedCols or D[row, col] > 50:  # Threshold de distância
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)
            
            # Processar objetos não utilizados
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])
        
        return self.objects

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
    # Usar ONNX para melhor performance com OpenVINO
    # ONNX oferece inferência rápida e é totalmente compatível
    model = YOLO("yolov8n.onnx")
    # Para modelos exportados (ONNX/OpenVINO), não usar model.to()
    # O device será passado diretamente no predict()
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
            # Inicializar tracker externo
            self.tracker = CentroidTracker(maxDisappeared=30)

        def callback(self, msg):
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # frame = cv2.flip(frame, 1)

            # Para modelos OpenVINO: usar predict() em vez de track()
            # e passar device='cpu' como argumento
            results = self.model.predict(source=frame,
                                        conf=0.6,
                                        imgsz=640,
                                        device='cpu',
                                        verbose=False)

            
            if results[0].boxes is not None:
                boxes = results[0].boxes.xywh
                class_ids = results[0].boxes.cls.int().tolist()
                confidences = results[0].boxes.conf.float().tolist()
                
                # Preparar lista de centroides para o tracker
                centroids = []
                detections_info = []
                
                for box, class_id, conf in zip(boxes, class_ids, confidences):
                    if class_id != self.target_id:
                        continue
                    x_center, y_center, width, height = box
                    centroids.append([x_center.item(), y_center.item()])
                    detections_info.append({
                        'x_center': x_center.item(),
                        'y_center': y_center.item(),
                        'width': width.item(),
                        'height': height.item(),
                        'conf': conf,
                        'class_id': class_id
                    })
                
                # Atualizar tracker com centroides
                tracked_objects = self.tracker.update(centroids)
                
                # Processar objetos rastreados
                for track_id, centroid in tracked_objects.items():
                    # Encontrar detecção correspondente ao tracker
                    for det in detections_info:
                        if np.sqrt((det['x_center'] - centroid[0])**2 + 
                                 (det['y_center'] - centroid[1])**2) < 30:
                            x_center = det['x_center']
                            y_center = det['y_center']
                            width = det['width']
                            height = det['height']
                            conf = det['conf']
                            
                            x1 = int(x_center - width / 2)
                            y1 = int(y_center - height / 2)
                            x2 = int(x_center + width / 2)
                            y2 = int(y_center + height / 2)
                            size = abs(y2 - y1)
                            
                            distance, lateral, vertical = calculate_3d_position(x_center, y_center, size, 0.18)
                            self.pos_list.append([distance, lateral, vertical])
                            class_name = self.class_names[det['class_id']]
                            
                            print(f"Track ID: {track_id}, Class: {class_name}, Confidence: {conf:.2f}")
                            print(f"Position: Center({x_center:.1f}, {y_center:.1f}), Box[{x1}, {y1}, {x2}, {y2}]")
                            print(f"Size: {width:.1f}x{height:.1f}")
                            print("-" * 50)
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{class_name} {track_id} ({conf:.2f})"
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            break
            
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
