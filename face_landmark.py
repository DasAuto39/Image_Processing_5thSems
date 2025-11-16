face_landmark.py

"""
For finding the face and face landmarks for further manipulication
"""

import cv2
import mediapipe as mp
import numpy as np

class FaceMeshDetector:
    def __init__(self,
                 static_image_mode=False,
                 max_num_faces=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            self.static_image_mode,
            self.max_num_faces,
            True,  # refine_landmarks=True (for iris)
            self.min_detection_confidence,
            self.min_tracking_confidence
        )

        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw=True):
        # convert the img from BRG to RGB
        # Flip is done here so coordinates are correct
        img_rgb = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)

        img_rgb.flags.writeable = False
        self.results = self.face_mesh.process(img_rgb)
        img_rgb.flags.writeable = True

        # Create the BGR image to draw on and return
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        self.imgH, self.imgW, self.imgC = img_bgr.shape
        self.faces = []

        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(
                        image=img_bgr,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=self.drawing_spec,
                        connection_drawing_spec=self.drawing_spec)

                # --- OPTIMIZATION ---
                # Replaced slow Python list.append() in a loop
                # with a pre-allocated NumPy array.
                num_landmarks = len(face_landmarks.landmark)
                face = np.zeros((num_landmarks, 2), dtype=np.int32)
                
                for i, lmk in enumerate(face_landmarks.landmark):
                    face[i, 0] = int(lmk.x * self.imgW)
                    face[i, 1] = int(lmk.y * self.imgH)
                
                # This appends a NumPy array, not a list of lists
                self.faces.append(face)
                # --- END OF OPTIMIZATION ---

        return img_bgr, self.faces


# sample run of the module
def main():
    # ... (demo code is fine as-is)
    #pass
    detector = FaceMeshDetector()

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, img = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue

        img, faces = detector.findFaceMesh(img)

        # if faces:
        #     print(faces[0])

        cv2.imshow('MediaPipe FaceMesh', img)

        # press "q" to leave
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
if __name__ == "__main__":
    # demo code
    # main() # Commented out to prevent accidental runs
    main()