import cv2
import numpy as np

# Capture Device

#device = "rtsp://admin:nupedee7@192.168.0.108:554/tcp"
device = "rtsp://admin:nupedee7@192.168.0.108:554/cam/realmonitor?channel=1&subtype=0&proto=Onvif"
# Open capture
cap = cv2.VideoCapture(device)

# Bola de Tênis 
#lower_color = np.array([22, 62, 67])
#upper_color = np.array([40, 162, 255])

# Yellow disc
#lower_color = np.array([24,76,145])
#upper_color = np.array([36,162,187])

# Purple disc
lower_color = np.array([122,87,121])
upper_color = np.array([134,228,227])


# Lista para guardar os pontos do trajeto
trajectory_points = []

# Frame "canvas" onde será desenhado o trajeto
canvas = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    if canvas is None:
        # Criar canvas do mesmo tamanho do frame
        canvas = np.zeros_like(frame)

    # Converter para HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Criar máscara
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Limpar ruído
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Encontrar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area > 60:
            # Centroide
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                trajectory_points.append((cx, cy))

                # Desenhar o ponto no canvas
                cv2.circle(canvas, (cx, cy), 2, (0, 255, 0), -1)

                # Desenhar linhas conectando os pontos
                if len(trajectory_points) > 1:
                    for i in range(1, len(trajectory_points)):
                        cv2.line(canvas, trajectory_points[i-1], trajectory_points[i], (0, 255, 0), 2)

    # Mostrar vídeo com máscara + canvas sobreposto
    output = cv2.addWeighted(frame, 0.7, canvas, 1, 0)
    cv2.imshow("Tracking", output)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
        break

# Ao final, salvar o canvas com o trajeto
cv2.imwrite("trajeto.png", canvas)
print("Trajeto salvo como trajeto.png")

cap.release()
cv2.destroyAllWindows()
