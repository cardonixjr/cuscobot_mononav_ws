import cv2
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

# --- Configuração da câmera ---
cap = cv2.VideoCapture(0)  # use 0 para webcam, ou substitua por um vídeo/câmera IP

sift = cv2.SIFT_create()

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)


K =  np.array([[673.84481 ,  0,         267.19239],
              [  0,         689.86342, 238.83533],
              [  0,         0,         1]],         dtype=np.float32)



prev_gray = None
prev_kp = None
prev_des = None

# Pose acumulada (R, t)
R_f = np.eye(3)
t_f = np.zeros((3, 1))

sum_kp = 0
count_kp = 0

sum_good = 0
count_good = 0

draw_params = dict(matchColor = -1, # draw matches in green color
                singlePointColor = None,
                matchesMask = None, # draw only inliers
                flags = 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # input_img = cv2.equalizeHist(gray)  # Histogram Equalization

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))    # CLAHE Equalization
    # input_img = clahe.apply(gray)

    # lap = cv2.Laplacian(gray, cv2.CV_64F)                     # Laplacian Edge augmentation
    # sharp = gray - 0.5 * lap
    # input_img = np.clip(sharp, 0, 255).astype(np.uint8)

    # input_img = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)    # Normalize

    # edges = cv2.Canny(gray, 50, 150)                            # Sobel multi-scale detection
    # input_img = cv2.addWeighted(gray, 0.7, edges, 0.3, 0)

    input_img = gray
    
    # --- Extração de features ---
    kp, des = sift.detectAndCompute(input_img, None)

    sum_kp += len(kp)
    count_kp+=1

    
    # TODO: Lê o numero de features encontradas
    #       Compara com o numero de features boas

    good = []

    if prev_gray is not None:
        if len(kp)>1:
            # Matching
            matches = flann.knnMatch(prev_des, des, k=2)
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)
            
            if len(good) > 8:
                pts1 = np.float32([prev_kp[m.queryIdx].pt for m in good])
                pts2 = np.float32([kp[m.trainIdx].pt for m in good])
    
            sum_good += len(good)
            count_good += 1
        
    
        # Desenhar Matches
        matches_frame = cv2.drawMatches(prev_gray, prev_kp, gray, kp, good ,None,**draw_params)
        cv2.imshow("Matches", matches_frame)
    
    # Desenhar features
    kp_frame = cv2.drawKeypoints(frame, kp, None, color=(0,255,0))

    cv2.imshow("Original Image", frame)
    cv2.imshow("Processed Image", input_img)
    cv2.imshow("Keypoints", kp_frame)
    

    prev_gray = gray
    prev_kp = kp
    prev_des = des
    
    if cv2.waitKey(1) & 0xFF == 27:  # Pressione ESC para sair
        break

avg_kp = sum_kp/count_kp
print(f"media de kp: {avg_kp}")

avg_good = sum_good/count_good
print(f"media de good: {avg_good}")

print(f"Percentual de bons: {avg_good*100/avg_kp}")

cap.release()
cv2.destroyAllWindows()

