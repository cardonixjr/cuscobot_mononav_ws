import cv2
import numpy as np
import pyrealsense2 as rs

# Configuração da RealSense D435i
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

lower_color = np.array([43, 116, 110])
upper_color = np.array([78, 227, 255])

# Lista para guardar os pontos do trajeto
trajectory_points = []

# Frame "canvas" onde será desenhado o trajeto
canvas = None

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    frame = np.asanyarray(color_frame.get_data())
    frame = cv2.resize(frame, (640, 480))

    if canvas is None:
        canvas = np.zeros_like(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area > 60:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                trajectory_points.append((cx, cy))
                cv2.circle(canvas, (cx, cy), 2, (0, 255, 0), -1)

                if len(trajectory_points) > 1:
                    for i in range(1, len(trajectory_points)):
                        cv2.line(canvas, trajectory_points[i-1], trajectory_points[i], (0, 255, 0), 2)

    output = cv2.addWeighted(frame, 0.7, canvas, 1, 0)
    cv2.imshow("Tracking", output)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
        break

cv2.imwrite("trajeto.png", canvas)
print("Trajeto salvo como trajeto.png")

pipeline.stop()
cv2.destroyAllWindows()