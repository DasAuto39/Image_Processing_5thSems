import cv2
import mediapipe as mp
import numpy as np
import socket
import sys

from face_landmark import FaceMeshDetector
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer
from face_feature import FacialFeatures, Eyes
from hand_tracker import HandTracker

def init_TCP(port):
    address = ('127.0.0.1', port)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(address)
        print("Connected to address:", socket.gethostbyname(socket.gethostname()) + ":" + str(port))
        return s
    except OSError as e:
        print("Error while connecting :: %s" % e)
        sys.exit()

def send_info_to_unity(s, args_tuple):
    msg = " ".join([f"{x:.4f}" for x in args_tuple]) + " "
    try:
        s.send(bytes(msg, "utf-8"))
    except socket.error as e:
        print("error while sending :: " + str(e))
        sys.exit()

def print_debug_msg(args_tuple):
    msg = " ".join([f"{x:.4f}" for x in args_tuple])
    print(msg)

def main():
    target_port = 5066
    camera_source = 0
    show_preview = True
    debug_console = False

    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_source}")
        sys.exit()

    detector = FaceMeshDetector()
    hand_tracker = HandTracker(detection_conf=0.6, tracking_conf=0.5)

    success, img = cap.read()
    if not success:
        print("Error: Could not read frame from camera.")
        cap.release()
        sys.exit()

    pose_estimator = PoseEstimator((img.shape[0], img.shape[1]))
    facial_features_calc = FacialFeatures()

    pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    eyes_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    mouth_dist_stabilizer = Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1
    )

    client_sock = init_TCP(target_port)

    while cap.isOpened():
        success, img = cap.read() 

        if not success:
            print("Ignoring empty camera frame.")
            continue

        img = cv2.flip(img, 1)

        lh_x, lh_y, rh_x, rh_y = hand_tracker.process(img, draw=True)

        img_facemesh, faces = detector.findFaceMesh(img)

        if faces:
            face_landmarks_np = faces[0]
            image_points = face_landmarks_np[:468]
            iris_image_points = face_landmarks_np[468:]

            pose = pose_estimator.solve_pose_by_all_points(image_points)

            x_ratio_left, y_ratio_left = facial_features_calc.detect_iris(image_points, iris_image_points, Eyes.LEFT)
            x_ratio_right, y_ratio_right = facial_features_calc.detect_iris(image_points, iris_image_points, Eyes.RIGHT)

            ear_left = facial_features_calc.eye_aspect_ratio(image_points, Eyes.LEFT)
            ear_right = facial_features_calc.eye_aspect_ratio(image_points, Eyes.RIGHT)

            pose_eye = [ear_left, ear_right, x_ratio_left, y_ratio_left, x_ratio_right, y_ratio_right]

            mar = facial_features_calc.mouth_aspect_ratio(image_points)
            mouth_distance = facial_features_calc.mouth_distance(image_points)

            steady_pose = []
            pose_np = np.array(pose).flatten()
            for value, ps_stb in zip(pose_np, pose_stabilizers):
                ps_stb.update([value])
                steady_pose.append(ps_stb.state[0][0])
            steady_pose = np.reshape(steady_pose, (-1, 3))
            
            steady_pose_eye = []
            for value, ps_stb in zip(pose_eye, eyes_stabilizers):
                ps_stb.update([value])
                steady_pose_eye.append(ps_stb.state[0][0])
        
            mouth_dist_stabilizer.update([mouth_distance])
            steady_mouth_dist = mouth_dist_stabilizer.state[0][0]

            roll = np.clip(np.degrees(steady_pose[0][1]), -90, 90)
            pitch = np.clip(-(180 + np.degrees(steady_pose[0][0])), -90, 90)
            yaw =  np.clip(np.degrees(steady_pose[0][2]), -90, 90)

            data_tuple = (roll, pitch, yaw,
                          steady_pose_eye[0], steady_pose_eye[1],
                          steady_pose_eye[2], steady_pose_eye[3],
                          steady_pose_eye[4], steady_pose_eye[5],
                          mar, steady_mouth_dist,
                          lh_x, lh_y, rh_x, rh_y) 
            
            print("-" * 50)
            print(f"[HEAD]  Roll: {roll:.1f}, Pitch: {pitch:.1f}, Yaw: {yaw:.1f}")
            print(f"[EYES]  L_Open: {steady_pose_eye[0]:.2f}, R_Open: {steady_pose_eye[1]:.2f}")
            print(f"[IRIS]  L_Pos: ({steady_pose_eye[2]:.2f}, {steady_pose_eye[3]:.2f}) | R_Pos: ({steady_pose_eye[4]:.2f}, {steady_pose_eye[5]:.2f})")
            print(f"[MOUTH] MAR: {mar:.2f}, Width: {steady_mouth_dist:.2f}")
            print(f"[HANDS] LH: ({lh_x:.2f}, {lh_y:.2f}) | RH: ({rh_x:.2f}, {rh_y:.2f})")
            print("-" * 50)
            
            if lh_y > -0.9:
                h, w, _ = img_facemesh.shape
                px_l = int((lh_x / 2 + 0.5) * w)
                py_l = int((-lh_y / 2 + 0.5) * h)
                cv2.putText(img_facemesh, "L Hand", (px_l, py_l - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            
            if rh_y > -0.9:
                h, w, _ = img_facemesh.shape
                px_r = int((rh_x / 2 + 0.5) * w)
                py_r = int((-rh_y / 2 + 0.5) * h)
                cv2.putText(img_facemesh, "R Hand", (px_r, py_r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)

            if client_sock:
                send_info_to_unity(client_sock, data_tuple)

            if debug_console:
                print_debug_msg(data_tuple)
            
            if show_preview:
                pose_estimator.draw_axes(img_facemesh, steady_pose[0], steady_pose[1])

        else:
            pose_estimator.reset_r_vec_t_vec()

        if show_preview:
            cv2.imshow('Facial landmark', img_facemesh)
   
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()