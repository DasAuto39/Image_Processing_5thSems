import mediapipe as mp
import cv2
import numpy as np

class HandTracker:
    def __init__(self, max_hands=2, detection_conf=0.7, tracking_conf=0.5):
        # 1. Initialize HANDS model (for fingers/wrist precision)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )
        
        # 2. Initialize POSE model (for Arm/Body lines: 11-16)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )

        self.mp_draw = mp.solutions.drawing_utils
        
    def process(self, img, draw=True):
        """
        Returns: left_x, left_y, right_x, right_y
        """
        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        

        results_pose = self.pose.process(img_rgb)
        
        if draw and results_pose.pose_landmarks:
            self.draw_body_skeleton(img, results_pose.pose_landmarks)


        results_hands = self.hands.process(img_rgb)

        lh_x, lh_y = 0.0, -1.0 
        rh_x, rh_y = 0.0, -1.0

        if results_hands.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
                
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                wrist = hand_landmarks.landmark[0]
                label = handedness.classification[0].label
                
                # Normalize coordinates
                norm_x = (wrist.x - 0.5) * 2
                norm_y = -(wrist.y - 0.5) * 2 

                # Mirror Logic Fix (Preserved from your previous request)
                if label == "Left": 
                    lh_x, lh_y = norm_x, norm_y
                else:
                    rh_x, rh_y = norm_x, norm_y

        return lh_x, lh_y, rh_x, rh_y

    def draw_body_skeleton(self, img, landmarks):

        h, w, _ = img.shape
        
        # pixel coordinates
        def get_coord(idx):
            lm = landmarks.landmark[idx]
            if lm.visibility < 0.5: return None
            return (int(lm.x * w), int(lm.y * h))


        
        # Draw Shoulders Connection (11-12)
        p11 = get_coord(11)
        p12 = get_coord(12)
        if p11 and p12:
            cv2.line(img, p11, p12, (255, 255, 255), 2) 

        # Draw Left Arm (11 -> 13 -> 15)
        p13 = get_coord(13)
        p15 = get_coord(15)
        if p11 and p13: cv2.line(img, p11, p13, (255, 255, 0), 4) # Cyan
        if p13 and p15: cv2.line(img, p13, p15, (255, 255, 0), 4)

        # Draw Right Arm (12 -> 14 -> 16)
        p14 = get_coord(14)
        p16 = get_coord(16)
        if p12 and p14: cv2.line(img, p12, p14, (255, 0, 255), 4) # Magenta
        if p14 and p16: cv2.line(img, p14, p16, (255, 0, 255), 4)


# --- VISUALIZATION TEST CODE ---
if __name__ == "__main__":
    print("Initializing Full Body & Hand Tracker...")
    cap = cv2.VideoCapture(0)
    
    # turn off internal drawing inside process so we can control the mirror flow better here
    tracker = HandTracker()

    while cap.isOpened():
        success, img = cap.read()
        if not success: continue

        # Mirroring
        img = cv2.flip(img, 1)

        # Process
        lh_x, lh_y, rh_x, rh_y = tracker.process(img, draw=True)

        # Add extra debug text
        if lh_y > -0.9:
            h, w, _ = img.shape
            px_l = int((lh_x / 2 + 0.5) * w)
            py_l = int((-lh_y / 2 + 0.5) * h)
            cv2.putText(img, "L Hand", (px_l, py_l - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            
        if rh_y > -0.9:
            h, w, _ = img.shape
            px_r = int((rh_x / 2 + 0.5) * w)
            py_r = int((-rh_y / 2 + 0.5) * h)
            cv2.putText(img, "R Hand", (px_r, py_r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)

        cv2.imshow('Arm & Hand Tracker Test', img)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()