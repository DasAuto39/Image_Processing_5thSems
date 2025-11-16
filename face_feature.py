face_feature.py

"""
Miscellaneous facial features detection implementation
"""

import cv2
import numpy as np
from enum import Enum

class Eyes(Enum):
    LEFT = 1
    RIGHT = 2

class FacialFeatures:
    
    # ... (eye_key_indicies is unchanged)
    eye_key_indicies=[
        [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173 ],
        [ 263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398 ]
    ]


    @staticmethod
    def resize_img(img, scale_percent):
        # ... (unchanged)
        pass

    @staticmethod
    def eye_aspect_ratio(image_points, side):

        p1, p2, p3, p4, p5, p6 = 0, 0, 0, 0, 0, 0
        tip_of_eyebrow = 0

        if side == Eyes.LEFT:
            eye_key_left = FacialFeatures.eye_key_indicies[0]

            # OPTIMIZATION: Use np.mean for clarity and potential speedup
            p2 = np.mean(image_points[[eye_key_left[10], eye_key_left[11]]], axis=0)
            p3 = np.mean(image_points[[eye_key_left[13], eye_key_left[14]]], axis=0)
            p6 = np.mean(image_points[[eye_key_left[2],  eye_key_left[3]]], axis=0)
            p5 = np.mean(image_points[[eye_key_left[5],  eye_key_left[6]]], axis=0)
            
            p1 = image_points[eye_key_left[0]]
            p4 = image_points[eye_key_left[8]]
            tip_of_eyebrow = image_points[105]

        elif side == Eyes.RIGHT:
            eye_key_right = FacialFeatures.eye_key_indicies[1]

            # OPTIMIZATION: Use np.mean
            p3 = np.mean(image_points[[eye_key_right[10], eye_key_right[11]]], axis=0)
            p2 = np.mean(image_points[[eye_key_right[13], eye_key_right[14]]], axis=0)
            p5 = np.mean(image_points[[eye_key_right[2],  eye_key_right[3]]], axis=0)
            p6 = np.mean(image_points[[eye_key_right[5],  eye_key_right[6]]], axis=0)
            
            p1 = image_points[eye_key_right[8]]
            p4 = image_points[eye_key_right[0]]
            tip_of_eyebrow = image_points[334]

        ear = np.linalg.norm(p2-p6) + np.linalg.norm(p3-p5)
        ear /= (2 * np.linalg.norm(p1-p4) + 1e-6)
        
        # This normalization is complex, but the math is already NumPy
        ear_norm = np.linalg.norm(tip_of_eyebrow - image_points[2]) / (np.linalg.norm(image_points[6] - image_points[2]) + 1e-6)
        ear *= ear_norm
        return ear

    @staticmethod
    def mouth_aspect_ratio(image_points):
        # This function is already using efficient NumPy operations
        p1 = image_points[78]
        p2 = image_points[81]
        p3 = image_points[13]
        p4 = image_points[311]
        p5 = image_points[308]
        p6 = image_points[402]
        p7 = image_points[14]
        p8 = image_points[178]

        mar = np.linalg.norm(p2-p8) + np.linalg.norm(p3-p7) + np.linalg.norm(p4-p6)
        mar /= (2 * np.linalg.norm(p1-p5) + 1e-6)
        return mar

    @staticmethod
    def mouth_distance(image_points):
        # Already efficient
        p1 = image_points[78]
        p5 = image_points[308]
        return np.linalg.norm(p1-p5)

    @staticmethod
    def detect_iris(image_points, iris_image_points, side):
        # ... (variable setup is the same)
        iris_img_point = -1
        p1, p4 = 0, 0
        eye_y_high, eye_y_low = 0, 0
        
        if side == Eyes.LEFT:
            iris_img_point = 468
            eye_key_left = FacialFeatures.eye_key_indicies[0]
            p1 = image_points[eye_key_left[0]]
            p4 = image_points[eye_key_left[8]]
            eye_y_high = image_points[eye_key_left[12]]
            eye_y_low = image_points[eye_key_left[4]]

        elif side == Eyes.RIGHT:
            iris_img_point = 473
            eye_key_right = FacialFeatures.eye_key_indicies[1]
            p1 = image_points[eye_key_right[8]]
            p4 = image_points[eye_key_right[0]]
            eye_y_high = image_points[eye_key_right[12]]
            eye_y_low = image_points[eye_key_right[4]]

        p_iris = iris_image_points[iris_img_point - 468]

        # --- OPTIMIZATION ---
        # Replaced manual vector creation with faster NumPy subtraction.
        # Replaced 4x linalg.norm() calls (slow, uses sqrt)
        # with 4x np.dot() calls (fast, no sqrt) to get norm^2.
        
        # Calculate X-rate
        vec_p1_p4 = p4 - p1
        vec_p1_iris = p_iris - p1
        # norm_p1_p4_sq is norm(vec_p1_p4)^2
        norm_p1_p4_sq = np.dot(vec_p1_p4, vec_p1_p4) + 1e-6
        x_rate = np.dot(vec_p1_iris, vec_p1_p4) / norm_p1_p4_sq

        # Calculate Y-rate
        vec_eye_h_eye_l = eye_y_low - eye_y_high
        vec_eye_h_iris = p_iris - eye_y_high
        # norm_h_l_sq is norm(vec_eye_h_eye_l)^2
        norm_h_l_sq = np.dot(vec_eye_h_eye_l, vec_eye_h_eye_l) + 1e-6
        y_rate = np.dot(vec_eye_h_eye_l, vec_eye_h_iris) / norm_h_l_sq
        # --- END OF OPTIMIZATION ---

        return np.clip(x_rate, 0.0, 1.0), np.clip(y_rate, 0.0, 1.0) # Added clip for safety
def main():
    """
    Test function to run feature extraction from a webcam.
    """
    # Check if the required detector class was imported
    if 'FaceMeshDetector' not in globals():
        print("Error: FaceMeshDetector class not found.")
        print("Please make sure facial_landmark.py is in the same directory.")
        return

    detector = FaceMeshDetector()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Find landmarks
        # We need the optimized facial_landmark.py that returns NumPy arrays
        img_mesh, faces = detector.findFaceMesh(img)

        if faces:
            # Get the NumPy array of points
            points = faces[0]
            
            # Separate the 468 face points from the 10 iris points
            image_points = points[:468]
            iris_points = points[468:]

            # --- 1. Test Eye Aspect Ratio ---
            ear_left = FacialFeatures.eye_aspect_ratio(image_points, Eyes.LEFT)
            ear_right = FacialFeatures.eye_aspect_ratio(image_points, Eyes.RIGHT)
            
            cv2.putText(img_mesh, f"EAR Left:  {ear_left:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img_mesh, f"EAR Right: {ear_right:.2f}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # --- 2. Test Mouth Aspect Ratio ---
            mar = FacialFeatures.mouth_aspect_ratio(image_points)
            cv2.putText(img_mesh, f"MAR: {mar:.2f}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # --- 3. Test Iris Detection ---
            x_l, y_l = FacialFeatures.detect_iris(image_points, iris_points, Eyes.LEFT)
            x_r, y_r = FacialFeatures.detect_iris(image_points, iris_points, Eyes.RIGHT)
            
            cv2.putText(img_mesh, f"Iris L (X,Y): {x_l:.2f}, {y_l:.2f}", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img_mesh, f"Iris R (X,Y): {x_r:.2f}, {y_r:.2f}", (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Facial Features Test', img_mesh)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()