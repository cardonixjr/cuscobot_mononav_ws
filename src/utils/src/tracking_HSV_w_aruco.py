import cv2
import numpy as np
import time
import csv
import os
import json
import pickle
from matplotlib import pyplot as plt
from scipy.ndimage import uniform_filter1d

# Tamanho do marcador (14.4 cm)
MARKER_SIZE = 0.144

# Carregar calibração de câmera (se disponível)
def load_camera_calibration(calib_pkl="calibracao/calibração/calibration.pkl", calib_json="camera_calibration/camera_calibration.json"):
    """Carrega calibração de câmera de arquivo .pkl (preferencial) ou JSON."""
    # Tentar carregar do arquivo .pkl primeiro (formato da calibração do OpenCV)
    try:
        if os.path.exists(calib_pkl):
            with open(calib_pkl, 'rb') as f:
                camera_matrix, dist_coeffs = pickle.load(f)
            print(f"✅ Calibração de câmera carregada (PKL): {calib_pkl}")
            print(f"   Camera Matrix shape: {camera_matrix.shape}")
            print(f"   Distortion coeffs shape: {dist_coeffs.shape}")
            return camera_matrix, dist_coeffs
    except Exception as e:
        print(f"⚠️  Não foi possível carregar calibração PKL: {e}")
    
    # Fallback: tentar JSON
    try:
        if os.path.exists(calib_json):
            with open(calib_json, 'r') as f:
                data = json.load(f)
            camera_matrix = np.array(data['camera_matrix'])
            dist_coeffs = np.array(data['dist_coeffs']).reshape(-1, 1)
            print(f"✅ Calibração de câmera carregada (JSON): {calib_json}")
            return camera_matrix, dist_coeffs
    except Exception as e:
        print(f"⚠️  Calibração JSON não disponível: {e}")
    
    print("❌ Nenhuma calibração encontrada. Tracking sem correção de distorção.")
    return None, None

# Função para criar mapas de retificação (mais eficiente que undistort)
def create_undistort_maps(camera_matrix, dist_coeffs, frame_size):
    """Cria mapas de retificação para corrigir distorção de forma eficiente."""
    if camera_matrix is None or dist_coeffs is None:
        return None, None, None
    
    w, h = frame_size
    # Obter nova matriz de câmera otimizada
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    
    # Criar mapas de retificação
    mapx, mapy = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_32FC1
    )
    
    print(f"✅ Mapas de retificação criados para resolução {w}x{h}")
    print(f"   ROI (região de interesse): {roi}")
    
    return mapx, mapy, new_camera_matrix

# Função para calcular pose do marker e transformação
def get_marker_pose_and_transform(marker_corners, use_cache=True, cache=None):
    """
    Calcula a pose do marcador usando os cantos.
    Detecta automaticamente a orientação dos cantos.
    """
    # Se temos cache e use_cache=True, retorna direto
    if use_cache and cache is not None:
        return cache
    
    # Os 4 cantos (ordem pode variar)
    corners = marker_corners.astype(np.float32)
    
    # Encontrar os 4 cantos na ordem correta: TL, TR, BR, BL
    # Ordenar por Y (Top vs Bottom), depois por X (Left vs Right)
    sorted_by_y = corners[np.argsort(corners[:, 1])]
    top_two = sorted_by_y[:2]
    bottom_two = sorted_by_y[2:]
    
    # Top: sort by X (left to right)
    top_left = top_two[np.argsort(top_two[:, 0])][0]
    top_right = top_two[np.argsort(top_two[:, 0])][1]
    
    # Bottom: sort by X (left to right)
    bottom_left = bottom_two[np.argsort(bottom_two[:, 0])][0]
    bottom_right = bottom_two[np.argsort(bottom_two[:, 0])][1]
    
    p0 = top_left
    p1 = top_right
    p2 = bottom_right
    p3 = bottom_left
    
    # Centro do marker (origem)
    center = (p0 + p1 + p2 + p3) / 4
    
    # Vetores dos eixos
    # Eixo X: aponta para a direita
    axis_x = (p1 - p0 + p2 - p3) / 2
    # Eixo Y: aponta para baixo
    axis_y = (p2 - p1 + p3 - p0) / 2
    
    # Calcular escala em pixels
    marker_width_pixels = np.linalg.norm(axis_x)
    marker_height_pixels = np.linalg.norm(axis_y)
    
    # Escala: pixels por metro
    px_per_meter = (marker_width_pixels + marker_height_pixels) / 2 / MARKER_SIZE
    
    # Normalizar os vetores
    axis_x_norm = axis_x / (marker_width_pixels + 1e-6)
    axis_y_norm = axis_y / (marker_height_pixels + 1e-6)
    
    # Inverter eixo X para corrigir orientação visual
    axis_x_norm = -axis_x_norm
    
    result = (center, axis_x_norm, axis_y_norm, px_per_meter)
    
    return result

# Função para transformar coordenadas para o sistema local do marker (usando transformação corrigida)
def get_local_coords(point, marker_corners, cache=None):
    """
    Transforma um ponto para coordenadas locais do marker EM METROS.
    """
    center, axis_x_norm, axis_y_norm, px_per_meter = get_marker_pose_and_transform(marker_corners, use_cache=True, cache=cache)
    
    # Vetor do ponto relativo ao centro (em pixels)
    point_vec = np.array([point[0] - center[0], point[1] - center[1]], dtype=np.float32)
    
    # Projetar nos eixos locais (em pixels)
    local_x_px = np.dot(point_vec, axis_x_norm)
    local_y_px = np.dot(point_vec, axis_y_norm)
    
    # Converter para metros
    local_x_m = local_x_px / px_per_meter
    local_y_m = local_y_px / px_per_meter
    
    return (local_x_m, local_y_m)

# Função para desenhar eixos do marker na imagem
def draw_marker_axes(frame, marker_corners, scale=100, cache=None):
    """
    Desenha os eixos X (vermelho) e Y (verde) do marker na imagem.
    """
    try:
        center, axis_x_norm, axis_y_norm, px_per_meter = get_marker_pose_and_transform(marker_corners, use_cache=True, cache=cache)
        center_int = center.astype(int)
        
        # Desenhar a origem (círculo branco)
        cv2.circle(frame, tuple(center_int), 8, (255, 255, 255), -1)
        cv2.putText(frame, "O", tuple(center_int - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Desenhar eixo X (vermelho)
        end_x = center + axis_x_norm * scale
        pt1 = tuple(center_int)
        pt2 = tuple(end_x.astype(int))
        cv2.arrowedLine(frame, pt1, pt2, (0, 0, 255), 3, tipLength=0.2)
        cv2.putText(frame, 'X', tuple((end_x + 15).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Desenhar eixo Y (verde)
        end_y = center + axis_y_norm * scale
        pt1 = tuple(center_int)
        pt2 = tuple(end_y.astype(int))
        cv2.arrowedLine(frame, pt1, pt2, (0, 255, 0), 3, tipLength=0.2)
        cv2.putText(frame, 'Y', tuple((end_y + 15).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
    except Exception as e:
        print(f"[ERROR] Erro ao desenhar eixos: {e}")
    
    return frame

# Função para ajustar coordenadas retroativamente após detecção do marcador
def adjust_coordinates_retroactively(trajectory_points, trajectory_points_relative, marker_corners, detected_at_frame):
    """
    Ajusta todas as coordenadas anteriores à detecção do marcador,
    convertendo-as para o sistema de coordenadas relativas ao marker.
    """
    print(f"\n[AJUSTE RETROATIVO] Detectado marcador no frame {detected_at_frame}")
    print(f"  Ajustando {detected_at_frame} pontos anteriores...")
    
    # Recalcular coordenadas relativas para todos os pontos
    for i in range(detected_at_frame):
        if i < len(trajectory_points):
            point = trajectory_points[i]
            local_coords = get_local_coords(point, marker_corners, cache=None)
            trajectory_points_relative[i] = local_coords
    
    print(f"[AJUSTE RETROATIVO] Concluído! {detected_at_frame} pontos convertidos para coordenadas relativas.")

# Função para suavizar trajetória com filtro Savitzky-Golay
def smooth_trajectory(trajectory_points_relative, window_length=5, polyorder=2):
    """
    Suaviza a trajetória usando filtro de média móvel para reduzir distorção de paralaxe.
    window_length: tamanho da janela (deve ser ímpar)
    polyorder: ordem do polinômio
    """
    if len(trajectory_points_relative) < window_length:
        return trajectory_points_relative
    
    # Garantir que window_length seja ímpar
    window_length = window_length if window_length % 2 == 1 else window_length + 1
    
    try:
        from scipy.signal import savgol_filter
        
        trajectory_array = np.array(trajectory_points_relative)
        
        # Aplicar Savitzky-Golay filter em cada dimensão
        smoothed_x = savgol_filter(trajectory_array[:, 0], window_length, polyorder)
        smoothed_y = savgol_filter(trajectory_array[:, 1], window_length, polyorder)
        
        smoothed_trajectory = list(zip(smoothed_x, smoothed_y))
        
        print(f"[SUAVIZAÇÃO] Trajetória suavizada com Savitzky-Golay (window={window_length}, polyorder={polyorder})")
        return smoothed_trajectory
    
    except ImportError:
        print("[SUAVIZAÇÃO] scipy não disponível, usando média móvel simples")
        # Fallback: média móvel simples
        trajectory_array = np.array(trajectory_points_relative)
        smoothed_x = uniform_filter1d(trajectory_array[:, 0], size=window_length, mode='nearest')
        smoothed_y = uniform_filter1d(trajectory_array[:, 1], size=window_length, mode='nearest')
        
        smoothed_trajectory = list(zip(smoothed_x, smoothed_y))
        return smoothed_trajectory

# Capture Device

#device = "rtsp://admin:nupedee7@192.168.0.108:554/tcp"
device = "rtsp://admin:nupedee7@192.168.1.6:554/cam/realmonitor?channel=1&subtype=0&proto=Onvif"
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

# ArUco Marker parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
MARKER_ID = 0  # 4x4_50_0 (ID = 0)
MARKER_SIZE = 0.15  # 15 cm

# Carregar calibração de câmera
camera_matrix_calib, dist_coeffs_calib = load_camera_calibration()

# Variáveis para armazenar mapas de retificação (criados após primeiro frame)
mapx, mapy, new_camera_matrix = None, None, None
undistort_initialized = False

# Lista para guardar os pontos do trajeto (coordenadas da câmera)
trajectory_points = []
trajectory_points_relative = []  # Coordenadas relativas ao marker
trajectory_timestamps = []

# Frame "canvas" onde será desenhado o trajeto
canvas = None
last_frame = None  # Salva a última frame capturada
marker_corners_fixed = None  # Cantos do marker (detectado uma vez)
marker_origin_fixed = False  # Flag para fixar o marker na primeira detecção
marker_pose_cache = None  # Cache da pose para não recalcular todo frame
marker_detected_at_frame = None  # Frame onde o marker foi detectado (para ajuste retroativo)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    
    # Inicializar mapas de retificação (uma vez apenas)
    if not undistort_initialized and camera_matrix_calib is not None:
        h, w = frame.shape[:2]
        mapx, mapy, new_camera_matrix = create_undistort_maps(
            camera_matrix_calib, dist_coeffs_calib, (w, h)
        )
        undistort_initialized = True
    
    # Aplicar correção de distorção ao frame inteiro (se disponível)
    if mapx is not None and mapy is not None:
        frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    
    last_frame = frame.copy()  # Salvar a frame atual (já corrigida)

    if canvas is None:
        # Criar canvas do mesmo tamanho do frame
        canvas = np.zeros_like(frame)

    # Detectar marcador ArUco
    corners, ids, rejected = detector.detectMarkers(frame)
    
    if ids is not None and not marker_origin_fixed:
        # Desenhar markers detectados
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        # Procurar pelo marker com ID = 5 (apenas na primeira detecção)
        for i, marker_id in enumerate(ids):
            if marker_id[0] == MARKER_ID:
                # Salvar os cantos do marker
                marker_corners_fixed = corners[i][0]
                marker_origin_fixed = True  # Fixar a origem
                marker_detected_at_frame = len(trajectory_points)  # Frame atual
                
                marker_center = np.mean(marker_corners_fixed, axis=0).astype(int)
                
                # DEBUG: Imprimir cantos para entender a ordem
                print(f"\n[DEBUG CANTOS]")
                for idx, corner in enumerate(marker_corners_fixed):
                    print(f"  Canto {idx}: {corner}")
                print(f"  Centro: {marker_center}")
                
                # Calcular distâncias entre cantos
                side1 = np.linalg.norm(marker_corners_fixed[1] - marker_corners_fixed[0])
                side2 = np.linalg.norm(marker_corners_fixed[2] - marker_corners_fixed[1])
                side3 = np.linalg.norm(marker_corners_fixed[3] - marker_corners_fixed[2])
                side4 = np.linalg.norm(marker_corners_fixed[0] - marker_corners_fixed[3])
                
                print(f"  Lado 0->1: {side1:.1f}px, Lado 1->2: {side2:.1f}px")
                print(f"  Lado 2->3: {side3:.1f}px, Lado 3->0: {side4:.1f}px")
                print(f"  Tamanho esperado: {MARKER_SIZE*100:.1f}cm = {MARKER_SIZE:.4f}m")
                
                # Calcular pose para cache
                marker_pose_cache = get_marker_pose_and_transform(marker_corners_fixed, use_cache=False)
                center, axis_x_norm, axis_y_norm, px_per_meter = marker_pose_cache
                print(f"[CALIBRAÇÃO] Escala: {px_per_meter:.1f} px/m ({MARKER_SIZE:.4f}m = ~{MARKER_SIZE*px_per_meter:.1f}px)")
                print(f"[EIXOS] X normalizado: {axis_x_norm}, Y normalizado: {axis_y_norm}")
                
                # Se há pontos anteriores, ajustar retroativamente
                if marker_detected_at_frame > 0:
                    adjust_coordinates_retroactively(trajectory_points, trajectory_points_relative, marker_corners_fixed, marker_detected_at_frame)

    
    # Desenhar o marker na frame (se foi detectado)
    if marker_origin_fixed and ids is not None:
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        # Desenhar os eixos do marker
        frame = draw_marker_axes(frame, marker_corners_fixed, scale=100, cache=marker_pose_cache)
    elif marker_origin_fixed:
        # Desenhar os eixos do marker mesmo se não foi detectado novamente
        frame = draw_marker_axes(frame, marker_corners_fixed, scale=100, cache=marker_pose_cache)
        # Adicionar um círculo no centro para debug
        center = np.mean(marker_corners_fixed, axis=0).astype(int)
        cv2.circle(frame, tuple(center), 8, (255, 255, 255), -1)
        cv2.putText(frame, "Marker Fixed", tuple(center - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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
                
                # Nota: A correção de distorção já foi aplicada ao frame inteiro
                # então cx, cy já estão em coordenadas corrigidas

                trajectory_points.append((cx, cy))
                trajectory_timestamps.append(time.time())

                # Calcular coordenadas relativas ao marker (se foi detectado)
                if marker_origin_fixed:
                    # Transformar para coordenadas locais do marker
                    local_coords = get_local_coords((cx, cy), marker_corners_fixed, cache=marker_pose_cache)
                    trajectory_points_relative.append(local_coords)
                else:
                    trajectory_points_relative.append((cx, cy))

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
timestamp = int(time.time())

# 1. Salvar imagem com canvas do trajeto
cv2.imwrite(f"trajeto_camera_{timestamp}.png", canvas)
print(f"Trajeto salvo como trajeto_camera_{timestamp}.png")

# 2. Salvar CSV com coordenadas relativas ao marker ArUco
os.makedirs("trajetorias", exist_ok=True)
csv_filename = f"trajetorias/trajeto_aruco_markers_{timestamp}.csv"

# Aplicar suavização à trajetória antes de salvar
trajectory_points_relative_smoothed = smooth_trajectory(trajectory_points_relative, window_length=5, polyorder=2)

with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['timestamp', 'frame_x', 'frame_y', 'relative_x_m', 'relative_y_m', 'relative_x_smooth_m', 'relative_y_smooth_m', 'time_elapsed'])
    
    start_time = trajectory_timestamps[0] if trajectory_timestamps else 0
    
    for i, (point_cam, point_rel) in enumerate(zip(trajectory_points, trajectory_points_relative)):
        elapsed_time = trajectory_timestamps[i] - start_time if i < len(trajectory_timestamps) else 0
        point_smooth = trajectory_points_relative_smoothed[i] if i < len(trajectory_points_relative_smoothed) else point_rel
        writer.writerow([
            trajectory_timestamps[i] if i < len(trajectory_timestamps) else '',
            point_cam[0],
            point_cam[1],
            f"{point_rel[0]:.4f}",
            f"{point_rel[1]:.4f}",
            f"{point_smooth[0]:.4f}",
            f"{point_smooth[1]:.4f}",
            f"{elapsed_time:.3f}"
        ])

print(f"Coordenadas relativas ao marker salvas em: {csv_filename}")

# 3. Gerar gráficos do tracking em arquivos separados
if trajectory_points and last_frame is not None:
    # Figura 1: Trajeto em coordenadas da câmera (sobreposto na frame real)
    fig_cam, ax_cam = plt.subplots(1, 1, figsize=(7, 6))
    ax_cam.imshow(last_frame[:, :, ::-1])  # Converter BGR para RGB (frame real da câmera)
    xs = [p[0] for p in trajectory_points]
    ys = [p[1] for p in trajectory_points]
    ax_cam.plot(xs, ys, 'r-', linewidth=2, label='Trajectory')
    ax_cam.scatter(xs, ys, c='red', s=20, alpha=0.6)
    if marker_origin_fixed and marker_corners_fixed is not None:
        marker_center = np.mean(marker_corners_fixed, axis=0)
        ax_cam.scatter([marker_center[0]], [marker_center[1]], c='blue', s=100, marker='x', linewidth=3, label='ArUco Origin')
    
    ax_cam.set_title('Rastreamento em Coordenadas de Câmera')
    ax_cam.set_xlabel('X (pixels)')
    ax_cam.set_ylabel('Y (pixels)')
    ax_cam.legend()
    ax_cam.grid(True, alpha=0.3)
    
    cam_plot_filename = f"trajeto_tracking_camera_{timestamp}.png"
    fig_cam.savefig(cam_plot_filename, dpi=150, bbox_inches='tight')
    print(f"Gráfico do tracking (câmera) salvo como: {cam_plot_filename}")

    # Figura 2: Trajeto em coordenadas relativas ao marker
    fig_rel, ax_rel = plt.subplots(1, 1, figsize=(7, 6))
    rel_xs = [p[0] for p in trajectory_points_relative]
    rel_ys = [p[1] for p in trajectory_points_relative]
    
    ax_rel.plot(rel_xs, rel_ys, 'g-', linewidth=1.5, label='Trajectory (Raw)', alpha=0.5)
    ax_rel.scatter(rel_xs, rel_ys, c='green', s=15, alpha=0.4)
    
    smooth_xs = [p[0] for p in trajectory_points_relative_smoothed]
    smooth_ys = [p[1] for p in trajectory_points_relative_smoothed]
    ax_rel.plot(smooth_xs, smooth_ys, 'b-', linewidth=2.5, label='Trajectory (Smoothed)', alpha=0.9)
    ax_rel.scatter(smooth_xs, smooth_ys, c='blue', s=20, alpha=0.7)
    
    ax_rel.scatter([0], [0], c='blue', s=100, marker='x', linewidth=3, label='Origin (Marker)')
    ax_rel.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax_rel.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    ax_rel.set_title('Rastreamento em Coordenadas Relativas ao Marker (Raw vs Smoothed)')
    ax_rel.set_xlabel('X relativo (metros)')
    ax_rel.set_ylabel('Y relativo (metros)')
    ax_rel.legend()
    ax_rel.grid(True, alpha=0.3)
    ax_rel.set_aspect('equal')
    
    rel_plot_filename = f"trajeto_tracking_relative_{timestamp}.png"
    fig_rel.savefig(rel_plot_filename, dpi=150, bbox_inches='tight')
    print(f"Gráfico do tracking (relativo) salvo como: {rel_plot_filename}")

    plt.show()

cap.release()
cv2.destroyAllWindows()
