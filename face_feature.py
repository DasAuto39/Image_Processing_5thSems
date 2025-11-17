import cv2
import numpy as np
from enum import Enum

try:
    from face_landmark import FaceMeshDetector
except ImportError:
    print("Warning: facial_landmark.py not found. The main() test function will not run.")


class Eyes(Enum):
    LEFT = 1
    RIGHT = 2

class FacialFeatures:

    # Hard-coded indices for MediaPipe face mesh landmarks
    eye_key_indicies=[
        [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173 ],
        [ 263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398 ]
    ]

    # --- MODIFICATION ---
    # '@staticmethod' is removed and 'self' is added as the first argument.
    # This is now an 'instance method'.
    def eye_aspect_ratio(self, image_points, side):
        # Eye aspect ratio (EAR) is calculated to detect blinks.
        
        p1, p2, p3, p4, p5, p6 = 0, 0, 0, 0, 0, 0
        tip_of_eyebrow = 0

        if side == Eyes.LEFT:
            eye_key_left = self.eye_key_indicies[0] # Note: 'self' is used here now

            p2 = np.mean(image_points[[eye_key_left[10], eye_key_left[11]]], axis=0)
            p3 = np.mean(image_points[[eye_key_left[13], eye_key_left[14]]], axis=0)
            p6 = np.mean(image_points[[eye_key_left[2],  eye_key_left[3]]], axis=0)
            p5 = np.mean(image_points[[eye_key_left[5],  eye_key_left[6]]], axis=0)
            
            p1 = image_points[eye_key_left[0]]
            p4 = image_points[eye_key_left[8]]
            tip_of_eyebrow = image_points[105]

        elif side == Eyes.RIGHT:
            eye_key_right = self.eye_key_indicies[1] # Note: 'self' is used here now

            p3 = np.mean(image_points[[eye_key_right[10], eye_key_right[11]]], axis=0)
            p2 = np.mean(image_points[[eye_key_right[13], eye_key_right[14]]], axis=0)
            p5 = np.mean(image_points[[eye_key_right[2],  eye_key_right[3]]], axis=0)
            p6 = np.mean(image_points[[eye_key_right[5],  eye_key_right[6]]], axis=0)
            
            p1 = image_points[eye_key_right[8]]
            p4 = image_points[eye_key_right[0]]
            tip_of_eyebrow = image_points[334]

        ear = np.linalg.norm(p2-p6) + np.linalg.norm(p3-p5)
        ear /= (2 * np.linalg.norm(p1-p4) + 1e-6)
        
        ear_norm = np.linalg.norm(tip_of_eyebrow - image_points[2]) / (np.linalg.norm(image_points[6] - image_points[2]) + 1e-6)
        ear *= ear_norm
        return ear


    def mouth_aspect_ratio(self, image_points):
        # Mouth aspect ratio (MAR) is calculated for mouth openness.
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

    def mouth_distance(self, image_points):
        # Horizontal mouth distance is calculated.
        p1 = image_points[78]
        p5 = image_points[308]
        return np.linalg.norm(p1-p5)


    def detect_iris(self, image_points, iris_image_points, side):
        # Iris position is detected using landmarks 468-478.
        
        iris_img_point = -1
        p1, p4 = 0, 0
        eye_y_high, eye_y_low = 0, 0
        
        if side == Eyes.LEFT:
            iris_img_point = 468
            eye_key_left = self.eye_key_indicies[0] # Note: 'self' is used here now
            p1 = image_points[eye_key_left[0]]
            p4 = image_points[eye_key_left[8]]
            eye_y_high = image_points[eye_key_left[12]]
            eye_y_low = image_points[eye_key_left[4]]

        elif side == Eyes.RIGHT:
            iris_img_point = 473
            eye_key_right = self.eye_key_indicies[1] # Note: 'self' is used here now
            p1 = image_points[eye_key_right[8]]
            p4 = image_points[eye_key_right[0]]
            eye_y_high = image_points[eye_key_right[12]]
            eye_y_low = image_points[eye_key_right[4]]

        p_iris = iris_image_points[iris_img_point - 468]

        vec_p1_p4 = p4 - p1
        vec_p1_iris = p_iris - p1
        norm_p1_p4_sq = np.dot(vec_p1_p4, vec_p1_p4) + 1e-6
        x_rate = np.dot(vec_p1_iris, vec_p1_p4) / norm_p1_p4_sq

        vec_eye_h_eye_l = eye_y_low - eye_y_high
        vec_eye_h_iris = p_iris - eye_y_high
        norm_h_l_sq = np.dot(vec_eye_h_eye_l, vec_eye_h_eye_l) + 1e-6
        y_rate = np.dot(vec_eye_h_eye_l, vec_eye_h_iris) / norm_h_l_sq

        return np.clip(x_rate, 0.0, 1.0), np.clip(y_rate, 0.0, 1.0)


def main():

    if 'FaceMeshDetector' not in globals():
        print("Error: FaceMeshDetector class not found.")
        return

    detector = FaceMeshDetector()
    cap = cv2.VideoCapture(0)
    
    # An object must be created to call the instance methods
    features_calc = FacialFeatures()

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            continue

        img_mesh, faces = detector.findFaceMesh(img)

        if faces:
            points = faces[0]
            image_points = points[:468]
            iris_points = points[468:]

            # Methods are called on the 'features_calc' object
            ear_left = features_calc.eye_aspect_ratio(image_points, Eyes.LEFT)
            ear_right = features_calc.eye_aspect_ratio(image_points, Eyes.RIGHT)
            mar = features_calc.mouth_aspect_ratio(image_points)
            x_l, y_l = features_calc.detect_iris(image_points, iris_points, Eyes.LEFT)
            x_r, y_r = features_calc.detect_iris(image_points, iris_points, Eyes.RIGHT)
            
            cv2.putText(img_mesh, f"EAR Left:  {ear_left:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img_mesh, f"MAR: {mar:.2f}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img_mesh, f"Iris L (X,Y): {x_l:.2f}, {y_l:.2f}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Facial Features Test', img_mesh)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()