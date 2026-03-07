#!/usr/bin/env python3
"""
RealSense + ArUco + HSV Purple Object Tracking (3D + World Projection)
"""

import cv2
import pyrealsense2 as rs
import numpy as np
import csv
from datetime import datetime
import time
import os

# ==============================
# CONFIG
# ==============================

CSV_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "trajetorias"))
ARUCO_MARKER_LENGTH = 0.15
ASSOC_DISTANCE_THRESHOLD = 1.5
MAX_MISSES = 10
DETECT_EVERY_N_FRAMES = 1

# Purple HSV range
LOWER_TRESHOLD = np.array([45,88,89])
UPPER_TRESHOLD = np.array([179,255,255])

MIN_AREA = 60


# ==============================
# TRACK CLASS
# ==============================

class Track:
    next_id = 0

    def __init__(self, position, frame_num):
        self.id = Track.next_id
        Track.next_id += 1
        self.position = position
        self.last_update = frame_num
        self.miss_count = 0

    def update(self, position, frame_num):
        self.position = position
        self.last_update = frame_num
        self.miss_count = 0

    def increment_miss(self):
        self.miss_count += 1

    def is_lost(self):
        return self.miss_count >= MAX_MISSES


# ==============================
# MAIN CLASS
# ==============================

class RealsenseArucoHSV:

    def __init__(self):

        os.makedirs(CSV_DIR, exist_ok=True)
        timestamp = int(time.time())
        self.csv_path = os.path.join(CSV_DIR, f"trajetoria_objeto_roxo_{timestamp}.csv")

        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["timestamp_iso", "track_id", "x_world", "y_world", "z_world"])

        # ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
        except:
            self.aruco_params = cv2.aruco.DetectorParameters_create()

        # RealSense
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.accel)

        self.profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)

        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()

        self.intrinsics = self.profile.get_stream(
            rs.stream.color).as_video_stream_profile().get_intrinsics()

        self.camera_matrix = np.array([
            [self.intrinsics.fx, 0, self.intrinsics.ppx],
            [0, self.intrinsics.fy, self.intrinsics.ppy],
            [0, 0, 1]
        ], dtype=np.float32)

        self.dist_coeffs = np.array(self.intrinsics.coeffs, dtype=np.float32)

        self.aruco_transform = None
        self.aruco_detected = False

        self.tracks = []
        self.frame_count = 0

        print("Sistema iniciado.")
        print("Salvando CSV em:", self.csv_path)


    def run(self):

        try:
            while True:

                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)

                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                self.frame_count += 1

                # ==============================
                # ARUCO
                # ==============================

                if self.frame_count % DETECT_EVERY_N_FRAMES == 0:

                    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

                    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
                    detectorParams = cv2.aruco.DetectorParameters()
                    detector = cv2.aruco.ArucoDetector(dictionary, detectorParams)
                    corners, ids, _ = detector.detectMarkers(gray)

                    if ids is not None and len(ids) > 0:

                        cv2.aruco.drawDetectedMarkers(color_image, corners, ids)

                        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                            corners, ARUCO_MARKER_LENGTH,
                            self.camera_matrix, self.dist_coeffs
                        )

                        rvec = rvecs[0][0]
                        tvec = tvecs[0][0]

                        R_marker, _ = cv2.Rodrigues(rvec)

                        T_cam_to_marker = np.eye(4)
                        T_cam_to_marker[:3, :3] = R_marker
                        T_cam_to_marker[:3, 3] = tvec

                        self.aruco_transform = np.linalg.inv(T_cam_to_marker)

                        if not self.aruco_detected:
                            self.aruco_detected = True
                            for track in self.tracks:
                                pos_cam = np.array(
                                    [track.position[0],
                                     track.position[1],
                                     track.position[2], 1.0]
                                )
                                pos_world = self.aruco_transform @ pos_cam
                                track.position = pos_world[:3].tolist()

                        cv2.drawFrameAxes(
                            color_image,
                            self.camera_matrix,
                            self.dist_coeffs,
                            rvec,
                            tvec,
                            ARUCO_MARKER_LENGTH * 0.5
                        )

                # ==============================
                # HSV DETECTION
                # ==============================

                hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, LOWER_TRESHOLD, UPPER_TRESHOLD)

                mask = cv2.erode(mask, None, iterations=2)
                mask = cv2.dilate(mask, None, iterations=2)

                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                detections = []
                detection_boxes = []

                if contours:
                    c = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(c)

                    if area > MIN_AREA:

                        M = cv2.moments(c)
                        if M["m00"] > 0:

                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])

                            depth_value = depth_frame.get_distance(cx, cy)

                            if depth_value > 0:

                                point_3d_cam = rs.rs2_deproject_pixel_to_point(
                                    self.intrinsics, [cx, cy], depth_value
                                )

                                if self.aruco_transform is not None:
                                    point_cam_homo = np.array(
                                        [point_3d_cam[0],
                                         point_3d_cam[1],
                                         point_3d_cam[2], 1.0]
                                    )
                                    point_world = (
                                        self.aruco_transform @ point_cam_homo
                                    )[:3]
                                else:
                                    point_world = point_3d_cam

                                detections.append(point_world)

                                x, y, w, h = cv2.boundingRect(c)
                                detection_boxes.append((x, y, x+w, y+h))

                                cv2.rectangle(
                                    color_image, (x, y),
                                    (x+w, y+h),
                                    (255, 0, 255), 2
                                )
                                cv2.circle(
                                    color_image, (cx, cy),
                                    5, (0, 255, 0), -1
                                )

                # ==============================
                # TRACKING
                # ==============================

                matched_tracks = set()

                for detection in detections:

                    best_track = None
                    best_distance = ASSOC_DISTANCE_THRESHOLD

                    for track in self.tracks:

                        dist = np.linalg.norm(
                            np.array(detection) -
                            np.array(track.position)
                        )

                        if dist < best_distance:
                            best_distance = dist
                            best_track = track

                    if best_track:
                        best_track.update(detection, self.frame_count)
                        matched_tracks.add(best_track)
                    else:
                        new_track = Track(detection, self.frame_count)
                        self.tracks.append(new_track)

                for track in self.tracks:
                    if track not in matched_tracks:
                        track.increment_miss()

                self.tracks = [t for t in self.tracks if not t.is_lost()]

                # ==============================
                # SAVE CSV + DRAW INFO
                # ==============================

                timestamp_iso = datetime.now().isoformat()

                for track in self.tracks:

                    self.csv_writer.writerow([
                        timestamp_iso,
                        track.id,
                        track.position[0],
                        track.position[1],
                        track.position[2]
                    ])

                    label = f"ID:{track.id}"
                    if self.aruco_transform is not None:
                        label += f" ({track.position[0]:.2f}, {track.position[1]:.2f})m"

                    cv2.putText(
                        color_image,
                        label,
                        (20, 30 + 30*track.id),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 255),
                        2
                    )

                # ==============================
                # DISPLAY
                # ==============================

                cv2.imshow("RealSense ArUco + HSV Purple Tracking", color_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            pass

        finally:
            self.shutdown()


    def shutdown(self):
        self.csv_file.close()
        self.pipeline.stop()
        cv2.destroyAllWindows()
        print("Finalizado.")
        print("CSV salvo em:", self.csv_path)


# ==============================

if __name__ == "__main__":
    node = RealsenseArucoHSV()
    node.run()