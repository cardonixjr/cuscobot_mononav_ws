from tools import image_processing, plot_keypoints
import cv2
import numpy as np

if __name__ == "__main__":
    cap = cv2.VideoCapture(2)

    # Create SIFT detector
    detector = cv2.SIFT_create()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame (e.g., image_processing)
        processed_frame = image_processing(frame)

        # Detect SIFT keypoints and descriptors on the processed (grayscale) image
        kp, des = detector.detectAndCompute(processed_frame, None)

        # Prepare visualization image (convert processed gray to BGR so markers are colored)
        if processed_frame.ndim == 2:
            vis = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
        else:
            vis = processed_frame.copy()

        # If keypoints found, build arrays and plot only the 'good' ones
        if kp is not None and len(kp) > 0:
            kpts = np.zeros((len(kp), 2))
            scores = np.zeros((len(kp),))
            for i, p in enumerate(kp):
                kpts[i, 0] = p.pt[0]
                kpts[i, 1] = p.pt[1]
                scores[i] = p.response

            # Definir threshold para 'bons' keypoints (ex: 70% do score máximo)
            if scores.max() > 0:
                threshold = 0.6 * scores.max()
                good_idx = scores >= threshold
                good_kpts = kpts[good_idx]
                good_scores = scores[good_idx]
                if len(good_kpts) > 0:
                    kp_img = plot_keypoints(vis, good_kpts, good_scores / good_scores.max())
                else:
                    kp_img = vis
            else:
                kp_img = vis
        else:
            kp_img = vis

        # Display the processed frame with keypoints
        cv2.imshow("Processed Frame", kp_img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()