import cv2
import numpy as np
import torch
import collections
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import csv

import pytransform3d.camera as pc

from cycler import cycle


def image2tensor(frame, device):
    return torch.from_numpy(frame / 255.).float()[None, None].to(device)


# --- VISUALIZATION ---
# based on: https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/utils.py
def plot_keypoints(image, kpts, scores=None):
    kpts = np.round(kpts).astype(int)

    if scores is not None:
        # get color
        smin, smax = scores.min(), scores.max()
        assert (0 <= smin <= 1 and 0 <= smax <= 1)

        color = cm.gist_rainbow(scores * 0.4)
        color = (np.array(color[:, :3]) * 255).astype(int)[:, ::-1]
        # text = f"min score: {smin}, max score: {smax}"

        for (x, y), c in zip(kpts, color):
            c = (int(c[0]), int(c[1]), int(c[2]))
            cv2.drawMarker(image, (x, y), tuple(c), cv2.MARKER_CROSS, 6)

    else:
        for x, y in kpts:
            cv2.drawMarker(image, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 6)

    return image


# based on: https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/utils.py
def plot_matches(image0, image1, kpts0, kpts1, scores=None, layout="lr"):
    """
    plot matches between two images. If score is nor None, then red: bad match, green: good match
    :param image0: reference image
    :param image1: current image
    :param kpts0: keypoints in reference image
    :param kpts1: keypoints in current image
    :param scores: matching score for each keypoint pair, range [0~1], 0: worst match, 1: best match
    :param layout: 'lr': left right; 'ud': up down
    :return:
    """
    H0, W0 = image0.shape[0], image0.shape[1]
    H1, W1 = image1.shape[0], image1.shape[1]

    if layout == "lr":
        H, W = max(H0, H1), W0 + W1
        out = 255 * np.ones((H, W, 3), np.uint8)
        out[:H0, :W0, :] = image0
        out[:H1, W0:, :] = image1
    elif layout == "ud":
        H, W = H0 + H1, max(W0, W1)
        out = 255 * np.ones((H, W, 3), np.uint8)
        out[:H0, :W0, :] = image0
        out[H0:, :W1, :] = image1
    else:
        raise ValueError("The layout must be 'lr' or 'ud'!")

    kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)

    # get color
    if scores is not None:
        smin, smax = scores.min(), scores.max()
        assert (0 <= smin <= 1 and 0 <= smax <= 1)

        color = cm.gist_rainbow(scores * 0.4)
        color = (np.array(color[:, :3]) * 255).astype(int)[:, ::-1]
    else:
        color = np.zeros((kpts0.shape[0], 3), dtype=int)
        color[:, 1] = 255

    for (x0, y0), (x1, y1), c in zip(kpts0, kpts1, color):
        c = c.tolist()
        if layout == "lr":
            cv2.line(out, (x0, y0), (x1 + W0, y1), color=c, thickness=1, lineType=cv2.LINE_AA)
            # display line end-points as circles
            cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x1 + W0, y1), 2, c, -1, lineType=cv2.LINE_AA)
        elif layout == "ud":
            cv2.line(out, (x0, y0), (x1, y1 + H0), color=c, thickness=1, lineType=cv2.LINE_AA)
            # display line end-points as circles
            cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x1, y1 + H0), 2, c, -1, lineType=cv2.LINE_AA)

    return out

def image_processing(img):
    # TODO: desenvolver processo de melhoria de imagens
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out_img = gray

    #bilat = cv2.bilateralFilter(gray, d=9, sigmaColor=50, sigmaSpace=50)

    #blur = cv2.GaussianBlur(bilat, (5,5), 0)
    #out_img = cv2.addWeighted(bilat, 1.2, blur, -0.2, 0)

    #outline = np.array([[-1, -1, -1],
    #                    [-1,  8, -1],
    #                    [-1, -1, -1]])
    #out_img = cv2.filter2D(gray, -1, outline)

    #sharpen = np.array([[0, -1, 0],
    #                    [-1, 5,-1],
    #                    [0, -1, 0]])
    #out_img = cv2.filter2D(out_img, -1, sharpen)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    g = clahe.apply(gray)

    blur = cv2.GaussianBlur(g, (7,7), 0)
    out_img = cv2.addWeighted(g, 1.15, blur, -0.15, 0)


    return out_img


def plot_results(wheel_odom, vo_odom):
    vos = np.array(vo_odom)
    od = np.array(wheel_odom)
    plt.figure()

    if len(vos)>0:
        plt.plot(-vos[:,2], -vos[:,1], label="VO (escalada)")

    if len(od)>0:
        plt.plot(od[:,0], od[:,1], label="Wheel Odometry")

    plt.title("Trajetórias")
    plt.legend()
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid()
    plt.show()

def plot_results_3d(wheel_odom, vo_odom):
    vos = np.array(vo_odom)
    od = np.array(wheel_odom)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plota linhas conectando os pontos e marcadores pequenos
    # ax.plot(-vos[:, 0], vos[:, 1], vos[:, 2], label='vo_scaled_trajectory', color='C0', marker='o', markersize=3, linewidth=1)
    # ax.plot(od[:, 0], od[:, 1], od[:, 2], label='wheel_trajectory', color='C1', marker='^', markersize=3, linewidth=1)

    ax.plot(vos[:, 2], vos[:, 1], vos[:, 0], label='vo_scaled_trajectory', color='C0', marker='o', markersize=3, linewidth=1)
    ax.plot(od[:, 0], od[:, 2], od[:, 1], label='wheel_trajectory', color='C1', marker='^', markersize=3, linewidth=1)


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Trajetórias 3D')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

def plot_pose(poses, K):
    number_of_frames = 20
    image_size = np.array([640, 480])
    #image_size = np.array([1920, 1080])

    plt.figure()
    #ax = pt.plot_transform()
    ax = plt.axes(projection='3d')

    camera_pose_poses = np.array(poses)


    key_frames_indices = np.linspace(0, len(camera_pose_poses) - 1, number_of_frames, dtype=int)
    colors = cycle("rgb")

    for i, c in zip(key_frames_indices, colors):
        pc.plot_camera(ax, K, camera_pose_poses[i],
                    sensor_size=(248,192), virtual_image_distance=0.6, c=c)


    plt.show()
    

def save_csv(wheel_odom, vo_odom):
    # salvar VO escalada
    with open("vo_scaled_trajectory.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x","y","z"])
        w.writerows(vo_odom)
    
    # salvar odom
    with open("wheel_trajectory.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x","y","z"])
        w.writerows(wheel_odom)

if __name__ == "__main__":
    cap = cv2.VideoCapture(2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame (e.g., image_processing)
        processed_frame = image_processing(frame)

        # Display the processed frame
        cv2.imshow("Processed Frame", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()