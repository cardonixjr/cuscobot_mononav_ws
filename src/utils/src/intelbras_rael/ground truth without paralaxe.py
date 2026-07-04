import cv2
import numpy as np
import threading
import os
import csv
import time
import scipy.io
import matplotlib.pyplot as plt

# ==========================================
# CLASSE PARA ZERAR DELAY
# ==========================================

class VideoStream:

    def __init__(self, src=0):

        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
        )

        self.stream = cv2.VideoCapture(src)

        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.grabbed, self.frame = self.stream.read()

        self.stopped = False

        self.thread = threading.Thread(
            target=self.update,
            args=()
        )

        self.thread.daemon = True
        self.thread.start()

    def update(self):

        while not self.stopped:

            if not self.stream.isOpened():
                continue

            self.grabbed, self.frame = self.stream.read()

    def read(self):

        if not self.grabbed:
            return False, None

        return True, self.frame.copy()

    def stop(self):

        self.stopped = True

        self.thread.join()

        self.stream.release()


# ==========================================
# CARREGAR CALIBRAÇÃO
# ==========================================

def load_calibration(filename="calibration.mat"):

    mat = scipy.io.loadmat(filename)

    K = mat["K"].T.astype(np.float64)

    D = np.array(
        [
            float(mat["dist"][0,0]),
            float(mat["dist"][0,1]),
            0.0,
            0.0,
            0.0
        ],
        dtype=np.float64
    )

    return K, D


# ==========================================
# PARÂMETROS
# ==========================================

K_raw, D = load_calibration("calibration.mat")
D = D.reshape(1,5)

Hc = 2.803
Hr = 0.35

lower_green = np.array([40, 50, 50])
upper_green = np.array([75, 255, 255])

rtsp_url = (
    "rtsp://admin:nupedee7@192.168.1.4:554/"
    "cam/realmonitor?channel=1&subtype=0&proto=Onvif"
)

trajectory_world = []
trajectory_pixels = []
trajectory_timestamps = []

print("Conectando à câmera...")

cap = VideoStream(rtsp_url)

time.sleep(1.0)

# ==========================================
# CAPTURA FRAME INICIAL
# ==========================================

ret, frame0 = cap.read()

if not ret:
    raise RuntimeError("Não foi possível obter frame da câmera")

h, w = frame0.shape[:2]

new_K, roi = cv2.getOptimalNewCameraMatrix(
    K_raw,
    D,
    (w, h),
    0,
    (w, h)
)


last_frame = None

print("Rastreamento iniciado.")
print("Pressione Q para finalizar.")

# ==========================================
# LOOP PRINCIPAL
# ==========================================

while True:

    ret, frame_bruto = cap.read()

    if not ret or frame_bruto is None:
        continue

    frame = cv2.undistort(
        frame_bruto,
        K_raw,
        D,
        None,
        new_K
    )
    last_frame = frame.copy()

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    mask = cv2.erode(
        mask,
        None,
        iterations=2
    )

    mask = cv2.dilate(
        mask,
        None,
        iterations=2
    )

    contornos, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contornos) > 0:

        maior_contorno = max(
            contornos,
            key=cv2.contourArea
        )

        M = cv2.moments(maior_contorno)

        if M["m00"] > 0:

            u_corr = M["m10"] / M["m00"]
            v_corr = M["m01"] / M["m00"]

            c_x = new_K[0, 2]
            c_y = new_K[1, 2]

            f_x = new_K[0, 0]
            f_y = new_K[1, 1]

            dx = u_corr - c_x
            dy = v_corr - c_y

            fator_paralaxe = (
                (Hc - Hr) / Hc
            )

            u_chao = (
                c_x +
                dx * fator_paralaxe
            )

            v_chao = (
                c_y +
                dy * fator_paralaxe
            )

            X_metros = (
                (u_chao - c_x)
                * Hc
                / f_x
            )

            Y_metros = (
                (v_chao - c_y)
                * Hc
                / f_y
            )

            trajectory_world.append(
                (
                    float(X_metros),
                    float(Y_metros)
                )
            )

            trajectory_pixels.append(
	        (
		    int(u_chao),
		    int(v_chao)
	        )
	    )

            trajectory_timestamps.append(
                time.time()
            )

    cv2.imshow(
        "Tracking",
        frame
    )

    key = cv2.waitKey(1)

    if key & 0xFF == ord("q"):
        break

# ==========================================
# FINALIZAÇÃO
# ==========================================

cap.stop()
cv2.destroyAllWindows()

# ==========================================
# CSV
# ==========================================

timestamp = time.strftime(
    "%Y%m%d_%H%M%S"
)

os.makedirs(
    "trajetorias",
    exist_ok=True
)

FILENAME_BASE = "0407_ORB_04"
csv_filename = (
    f"trajetorias/ground_truth_{FILENAME_BASE}.csv"
)

with open(
    csv_filename,
    "w",
    newline=""
) as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow([
        "timestamp",
        "pixel_x",
        "pixel_y",
        "world_x_m",
        "world_y_m",
        "elapsed_time"
    ])

    t0 = (
        trajectory_timestamps[0]
        if len(trajectory_timestamps)
        else 0
    )

    for i in range(
        len(trajectory_world)
    ):

        writer.writerow([

            trajectory_timestamps[i],

            trajectory_pixels[i][0],
            trajectory_pixels[i][1],

            trajectory_world[i][0],
            trajectory_world[i][1],

            trajectory_timestamps[i] - t0
        ])

print(
    f"CSV salvo em: {csv_filename}"
)

# ==========================================
# PLOT FINAL
# ==========================================

if len(trajectory_world) > 1:

    xs = [
        p[0]
        for p in trajectory_world
    ]

    ys = [
        p[1]
        for p in trajectory_world
    ]

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        xs,
        ys,
        linewidth=2,
        label="Trajetória"
    )

    plt.scatter(
        xs,
        ys,
        s=10
    )

    plt.xlabel(
        "X (m)"
    )

    plt.ylabel(
        "Y (m)"
    )

    plt.title(
        "Ground Truth Odometry"
    )

    plt.axis("equal")

    plt.grid(True)

    plt.legend()

    plot_filename = (
        f"trajetorias/ground_truth_{FILENAME_BASE}.png"
    )

    plt.savefig(
        plot_filename,
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"Gráfico salvo em: {plot_filename}"
    )
    
    
    if last_frame is not None and len(trajectory_pixels) > 1:
        
        overlay = last_frame.copy()
        pts = np.array(
            trajectory_pixels,
            dtype=np.int32
        )

        for i in range(1, len(pts)):

            cv2.line(
                overlay,
                tuple(pts[i - 1]),
                tuple(pts[i]),
                (0, 255, 255),
                3
            )

        for p in pts:
    
            cv2.circle(
                overlay,
                tuple(p),
                3,
                (0, 0, 255),
                -1
            )

        overlay_filename = (
            f"trajetorias/"
            f"ground_truth_overlay_{FILENAME_BASE}.png"
        )

        cv2.imwrite(
            overlay_filename,
            overlay
        )

        print(
            f"Overlay salvo em: "
            f"{overlay_filename}"
        )

print("Finalizado.")
