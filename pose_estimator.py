import cv2
import numpy as np
import sys

try:
    # This test code needs facial_landmark.py
    from face_landmark import FaceMeshDetector
except ImportError:
    print("Warning: facial_landmark.py not found. The main() test function will not run.")


class PoseEstimator:

    def __init__(self, img_size=(480, 640)):
        self.size = img_size
        self.model_points_full = self.get_full_model_points()

        # Camera internals are defined
        self.focal_length = self.size[1]
        self.camera_center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0, 0, 1]], dtype="double")

        # No lens distortion is assumed
        self.dist_coeefs = np.zeros((4, 1))

        self.r_vec = None
        self.t_vec = None

    def get_full_model_points(self, filename='model.txt'):
        """All 468 3D model points are gotten from the file."""
        try:
            raw_value = np.loadtxt(filename, dtype=np.float32)
            model_points = raw_value.reshape(-1, 3)
            return model_points
        except IOError as e:
            print(f"Error loading model file {filename}: {e}")
            print("Please make sure 'model.txt' is in the correct directory.")
            sys.exit()


    def solve_pose_by_all_points(self, image_points):
        """
        Solving a pose from all the 468 image points.
        (rotation_vector, translation_vector) are returned as pose.
        """
        image_points_f64 = image_points.astype(np.float64)

        if self.r_vec is None:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(
                self.model_points_full, image_points_f64, self.camera_matrix, self.dist_coeefs)
            self.r_vec = rotation_vector
            self.t_vec = translation_vector
        else:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(
                self.model_points_full,
                image_points_f64,
                self.camera_matrix,
                self.dist_coeefs,
                rvec=self.r_vec,
                tvec=self.t_vec,
                useExtrinsicGuess=True) 
            
            self.r_vec = rotation_vector
            self.t_vec = translation_vector

        return (rotation_vector, translation_vector)

    def draw_annotation_box(self, image, rotation_vector, translation_vector, color=(255, 255, 255), line_width=2):
        """Drawing a 3D box as an annotation of pose."""
        point_3d = np.float32([
            (-75, -75, 0), (-75, 75, 0), (75, 75, 0), (75, -75, 0), (-75, -75, 0),
            (-40, -40, 400), (-40, 40, 400), (40, 40, 400), (40, -40, 400), (-40, -40, 400)
        ]).reshape(-1, 3)
        
        (point_2d, _) = cv2.projectPoints(point_3d,
                                          rotation_vector,
                                          translation_vector,
                                          self.camera_matrix,
                                          self.dist_coeefs)
        point_2d = np.int32(point_2d.reshape(-1, 2))
        
        cv2.polylines(image, [point_2d[:5]], True, color, line_width, cv2.LINE_AA)
        cv2.polylines(image, [point_2d[5:]], True, color, line_width, cv2.LINE_AA)
        for i in range(4):
            cv2.line(image, tuple(point_2d[i]), tuple(point_2d[i+5]), color, line_width, cv2.LINE_AA)


    def draw_axes(self, img, R, t):
         cv2.drawFrameAxes(img, self.camera_matrix, self.dist_coeefs, R, t, 20)

    def reset_r_vec_t_vec(self):
        self.r_vec = None
        self.t_vec = None


# Test code
def main():
    """
    Test function for pose_estimator.py
    This will estimate pose and draw the 3D axes on the face.
    """
    if 'FaceMeshDetector' not in globals():
        print("Error: FaceMeshDetector class not found.")
        print("Please make sure facial_landmark.py is in the same directory.")
        return

    detector = FaceMeshDetector()
    cap = cv2.VideoCapture(0)
    
    # Get a sample frame for size
    success, img = cap.read()
    if not success:
        print("Failed to read from camera.")
        cap.release()
        return
        
    pose_estimator = PoseEstimator((img.shape[0], img.shape[1]))

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            continue

        img_mesh, faces = detector.findFaceMesh(img)

        if faces:
            points = faces[0]
            image_points = points[:468] # Use all 468 points

            # 1. Solve the pose
            pose = pose_estimator.solve_pose_by_all_points(image_points)
            
            # 2. Draw the axes
            pose_estimator.draw_axes(img_mesh, pose[0], pose[1])
            
            # 3. (Optional) Draw the box
            # pose_estimator.draw_annotation_box(img_mesh, pose[0], pose[1])

        else:
            # Reset if face is lost
            pose_estimator.reset_r_vec_t_vec()

        cv2.imshow('Pose Estimator Test', img_mesh)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()