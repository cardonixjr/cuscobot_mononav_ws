import cv2
import numpy as np
import time
import csv
import os
import json
import pickle
import scipy.io
from matplotlib import pyplot as plt
from scipy.ndimage import uniform_filter1d

# Tamanho do marcador (14.4 cm)
MARKER_SIZE = 0.144

# Carregar calibração de câmera (se disponível)
def load_camera_calibration(
    calib_mat="/home/luciano/cuscobot_mononav_ws/src/utils/src/intelbras_rael/calibration.mat",
    calib_pkl="/home/luciano/cuscobot_mononav_ws/src/utils/src/intelbras_rael/calibration.pkl",
    calib_json="/home/luciano/cuscobot_mononav_ws/src/utils/src/intelbras_rael/camera_calibration.json"
):
    """Carrega calibração de câmera.
    Prioridade: MATLAB .mat > OpenCV .pkl > JSON.

    O MATLAB (Computer Vision Toolbox) armazena a matriz intrínseca em formato
    coluna-maior (transposta em relação ao OpenCV), por isso é necessário transpor K.
    Os coeficientes de distorção do MATLAB são apenas radiais [k1, k2]; os
    coeficientes tangenciais [p1, p2] são preenchidos com zero para o OpenCV.
    """
    # 1. Tentar carregar do arquivo MATLAB .mat
    try:
        if os.path.exists(calib_mat):
            print("find calib.mat")
            data = scipy.io.loadmat(calib_mat)
            # MATLAB armazena K transposta: K_matlab = K_opencv.T
            camera_matrix = data['K'].T.astype(np.float64)
            # Distorção radial do MATLAB: [k1, k2] → OpenCV: [k1, k2, p1, p2]
            k1, k2 = float(data['dist'][0, 0]), float(data['dist'][0, 1])
            dist_coeffs = np.array([[k1, k2, 0.0, 0.0]], dtype=np.float64)
            fx = float(data['fx'])
            fy = float(data['fy'])
            cx = float(data['cx'])
            cy = float(data['cy'])
            print(f"✅ Calibração MATLAB carregada: {calib_mat}")
            print(f"   fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
            print(f"   Distorção radial: k1={k1:.6f}, k2={k2:.6f}")
            print(f"   Camera Matrix:\n{camera_matrix}")
            return camera_matrix, dist_coeffs
    except Exception as e:
        print(f"⚠️  Não foi possível carregar calibração MATLAB .mat: {e}")

    # 2. Fallback: arquivo .pkl (OpenCV)
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

    # 3. Fallback: JSON
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

def marker_center_to_corners_world(center_xyz, marker_size):
    """Retorna os 4 cantos 3D (TL, TR, BR, BL) de um ArUco no plano Z=0."""
    cx, cy, cz = center_xyz
    h = marker_size / 2.0
    return np.array([
        [cx - h, cy + h, cz],  # TL
        [cx + h, cy + h, cz],  # TR
        [cx + h, cy - h, cz],  # BR
        [cx - h, cy - h, cz],  # BL
    ], dtype=np.float32)

def build_aruco_correspondences(corners, ids, marker_positions, marker_size):
    """Monta correspondências 3D-2D para solvePnP usando múltiplos ArUcos."""
    if ids is None:
        return None, None, []

    object_points = []
    image_points = []
    used_ids = []

    for i, marker_id in enumerate(ids.flatten()):
        if marker_id not in marker_positions:
            continue
        obj_corners = marker_center_to_corners_world(marker_positions[marker_id], marker_size)
        img_corners = corners[i][0].astype(np.float32)
        object_points.extend(obj_corners.tolist())
        image_points.extend(img_corners.tolist())
        used_ids.append(int(marker_id))

    if len(object_points) < 4:
        return None, None, used_ids

    return np.array(object_points, dtype=np.float32), np.array(image_points, dtype=np.float32), used_ids

def estimate_camera_pose_from_markers(corners, ids, camera_matrix, dist_coeffs, marker_positions, marker_size):
    """Estima pose da câmera no mundo dos marcadores via solvePnPRansac."""
    obj_pts, img_pts, used_ids = build_aruco_correspondences(
        corners, ids, marker_positions, marker_size
    )
    if obj_pts is None:
        return None, None, used_ids, None

    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts,
            img_pts,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=3.0,
            confidence=0.99,
            iterationsCount=200,
        )
        if not ok:
            return None, None, used_ids, None

        # Erro médio de reprojeção (diagnóstico)
        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
        proj = proj.reshape(-1, 2)
        err = float(np.mean(np.linalg.norm(proj - img_pts, axis=1)))
        return rvec, tvec, used_ids, err
    except Exception as e:
        print(f"[POSE] Falha ao estimar pose: {e}")
        return None, None, used_ids, None

def pixel_to_world_on_plane(point_xy, camera_matrix, dist_coeffs, rvec, tvec, plane_z=0.0):
    """Projeta pixel para ponto 3D no plano Z=plane_z do mundo."""
    pts = np.array([[point_xy]], dtype=np.float32)
    und = cv2.undistortPoints(pts, camera_matrix, dist_coeffs)
    x_n, y_n = und[0, 0, 0], und[0, 0, 1]

    # Raio na câmera (normalizado)
    ray_cam = np.array([x_n, y_n, 1.0], dtype=np.float64)

    R, _ = cv2.Rodrigues(rvec)
    R_inv = R.T
    t = tvec.reshape(3)

    # Centro da câmera no mundo: C = -R^T t
    cam_center_world = -R_inv @ t
    # Direção do raio no mundo
    ray_world = R_inv @ ray_cam

    if abs(ray_world[2]) < 1e-8:
        return None

    s = (plane_z - cam_center_world[2]) / ray_world[2]
    if s <= 0:
        return None

    p_world = cam_center_world + s * ray_world
    return float(p_world[0]), float(p_world[1]), float(p_world[2])

def fill_missing_points(points_xy):
    """Preenche lacunas (NaN) por forward/backward fill para permitir suavização."""
    if not points_xy:
        return points_xy

    arr = np.array(points_xy, dtype=np.float64)
    valid = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
    if not np.any(valid):
        return points_xy

    first_valid = int(np.argmax(valid))
    arr[:first_valid] = arr[first_valid]

    for i in range(first_valid + 1, len(arr)):
        if not (np.isfinite(arr[i, 0]) and np.isfinite(arr[i, 1])):
            arr[i] = arr[i - 1]

    return [tuple(v) for v in arr]

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
device = "rtsp://admin:nupedee7@192.168.1.4:554/cam/realmonitor?channel=1&subtype=0&proto=Onvif"
# Open capture
cap = cv2.VideoCapture(device)


lower_color = np.array([63, 75, 0])
upper_color = np.array([83, 236, 255])

# ArUco Marker parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
MARKER_ID = 0  # Mantido por compatibilidade (não é mais obrigatório)
MARKER_SIZE = 0.15  # 15 cm

# Mapa global dos ArUcos (referência no marker 0)
MARKER_POSITIONS = {
    0: (0.0, 0.0, 0.0),
    1: (0.0, 1.9, 0.0),
    2: (1.2, 1.9, 0.0),
    3: (1.2, 0.0, 0.0),
}

# Carregar calibração de câmera
camera_matrix_calib, dist_coeffs_calib = load_camera_calibration()

# Variáveis para armazenar mapas de retificação (criados após primeiro frame)
mapx, mapy, new_camera_matrix = None, None, None
undistort_initialized = False

# Lista para guardar os pontos do trajeto (coordenadas da câmera)
trajectory_points = []
trajectory_points_relative = []  # Coordenadas relativas ao marker
trajectory_points_world = []     # Coordenadas globais (m) no referencial do marker 0
trajectory_timestamps = []

# Frame "canvas" onde será desenhado o trajeto
canvas = None
last_frame = None  # Salva a última frame capturada
marker_corners_fixed = None  # Cantos do marker (detectado uma vez)
marker_origin_fixed = False  # Flag para fixar o marker na primeira detecção
marker_pose_cache = None  # Cache da pose para não recalcular todo frame
marker_detected_at_frame = None  # Frame onde o marker foi detectado (para ajuste retroativo)

# Pose da câmera no mundo dos ArUcos (atualizada por frame)
camera_rvec = None
camera_tvec = None
last_pose_error_px = None
last_used_marker_ids = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
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

    # Pose da câmera via múltiplos marcadores conhecidos
    if ids is not None:
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    pnp_camera_matrix = new_camera_matrix if new_camera_matrix is not None else camera_matrix_calib
    pnp_dist_coeffs = np.zeros((1, 4), dtype=np.float64) if new_camera_matrix is not None else dist_coeffs_calib

    if pnp_camera_matrix is not None:
        rvec, tvec, used_ids, reproj_err = estimate_camera_pose_from_markers(
            corners,
            ids,
            pnp_camera_matrix,
            pnp_dist_coeffs,
            MARKER_POSITIONS,
            MARKER_SIZE,
        )
        if rvec is not None and tvec is not None:
            camera_rvec, camera_tvec = rvec, tvec
            last_pose_error_px = reproj_err
            last_used_marker_ids = used_ids

    # Overlay de diagnóstico da pose
    if camera_rvec is not None:
        info = f"Pose OK | IDs: {last_used_marker_ids} | err={last_pose_error_px:.2f}px" if last_pose_error_px is not None else "Pose OK"
        cv2.putText(frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Pose indisponivel (mostre >=1 ArUco conhecido)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

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

                # Coordenadas globais no plano Z=0 (referencial do marker 0)
                world_point = None
                if camera_rvec is not None and camera_tvec is not None and pnp_camera_matrix is not None:
                    world_point = pixel_to_world_on_plane(
                        (cx, cy),
                        pnp_camera_matrix,
                        pnp_dist_coeffs,
                        camera_rvec,
                        camera_tvec,
                        plane_z=0.0,
                    )

                if world_point is not None:
                    trajectory_points_world.append((world_point[0], world_point[1]))
                    trajectory_points_relative.append((world_point[0], world_point[1]))
                else:
                    trajectory_points_world.append((np.nan, np.nan))
                    trajectory_points_relative.append((np.nan, np.nan))

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
name_mod = "0407_ORB_04"

# 1. Salvar imagem com canvas do trajeto
cv2.imwrite(f"/home/luciano/cuscobot_mononav_ws/src/utils/src/trajetorias/ground_truth_{name_mod}.png", canvas)
abs_path = os.path.abspath(f"ground_truth_{name_mod}.png")
print(f"Trajeto salvo como ground_truth_{name_mod}.png\nCaminho absoluto: {abs_path}")

# 2. Salvar CSV com coordenadas relativas ao marker ArUco
os.makedirs("trajetorias", exist_ok=True)
csv_filename = f"/home/luciano/cuscobot_mononav_ws/src/utils/src/trajetorias/trajeto_aruco_markers_{name_mod}.csv"

# Aplicar suavização à trajetória antes de salvar
trajectory_points_world_filled = fill_missing_points(trajectory_points_world)
trajectory_points_relative_smoothed = smooth_trajectory(trajectory_points_world_filled, window_length=5, polyorder=2)

with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        'timestamp',
        'frame_x',
        'frame_y',
        'world_x_m',
        'world_y_m',
        'world_x_smooth_m',
        'world_y_smooth_m',
        'time_elapsed',
    ])
    
    start_time = trajectory_timestamps[0] if trajectory_timestamps else 0
    
    for i, (point_cam, point_rel) in enumerate(zip(trajectory_points, trajectory_points_world)):
        elapsed_time = trajectory_timestamps[i] - start_time if i < len(trajectory_timestamps) else 0
        point_smooth = trajectory_points_relative_smoothed[i] if i < len(trajectory_points_relative_smoothed) else point_rel

        rel_x = f"{point_rel[0]:.4f}" if np.isfinite(point_rel[0]) else ''
        rel_y = f"{point_rel[1]:.4f}" if np.isfinite(point_rel[1]) else ''
        sm_x = f"{point_smooth[0]:.4f}" if np.isfinite(point_smooth[0]) else ''
        sm_y = f"{point_smooth[1]:.4f}" if np.isfinite(point_smooth[1]) else ''

        writer.writerow([
            trajectory_timestamps[i] if i < len(trajectory_timestamps) else '',
            point_cam[0],
            point_cam[1],
            rel_x,
            rel_y,
            sm_x,
            sm_y,
            f"{elapsed_time:.3f}"
        ])
abs_path = os.path.abspath(f"trajeto_aruco_markers_{name_mod}.csv")
print(f"Coordenadas relativas ao marker salvas em: {csv_filename}\nCaminho absoluto: {abs_path}")

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
    
    cam_plot_filename = f"/home/luciano/cuscobot_mononav_ws/src/utils/src/trajetorias/trajeto_tracking_camera_{name_mod}.png"
    fig_cam.savefig(cam_plot_filename, dpi=150, bbox_inches='tight')
    print(f"Gráfico do tracking (câmera) salvo como: {cam_plot_filename}")

    # Figura 2: Trajeto em coordenadas globais (m) pelo mapa dos ArUcos
    fig_rel, ax_rel = plt.subplots(1, 1, figsize=(7, 6))
    rel_xs = [p[0] for p in trajectory_points_world_filled]
    rel_ys = [p[1] for p in trajectory_points_world_filled]
    
    ax_rel.plot(rel_xs, rel_ys, 'g-', linewidth=1.5, label='Trajectory (Raw)', alpha=0.5)
    ax_rel.scatter(rel_xs, rel_ys, c='green', s=15, alpha=0.4)
    
    smooth_xs = [p[0] for p in trajectory_points_relative_smoothed]
    smooth_ys = [p[1] for p in trajectory_points_relative_smoothed]
    ax_rel.plot(smooth_xs, smooth_ys, 'b-', linewidth=2.5, label='Trajectory (Smoothed)', alpha=0.9)
    ax_rel.scatter(smooth_xs, smooth_ys, c='blue', s=20, alpha=0.7)
    
    ax_rel.scatter([0], [0], c='blue', s=100, marker='x', linewidth=3, label='Origin (Marker 0)')
    ax_rel.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax_rel.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    ax_rel.set_title('Rastreamento em Coordenadas Globais dos ArUcos (Raw vs Smoothed)')
    ax_rel.set_xlabel('X global (metros)')
    ax_rel.set_ylabel('Y global (metros)')
    ax_rel.legend()
    ax_rel.grid(True, alpha=0.3)
    ax_rel.set_aspect('equal')

    rel_plot_filename = f"/home/luciano/cuscobot_mononav_ws/src/utils/src/trajetorias/trajeto_tracking_relative_{name_mod}.png"
    fig_rel.savefig(rel_plot_filename, dpi=150, bbox_inches='tight')
    print(f"Gráfico do tracking (relativo) salvo como: {rel_plot_filename}")

    plt.show()

cap.release()
cv2.destroyAllWindows()
