import numpy as np
import cv2

class Stabilizer:
    """A Kalman filter is used as a point stabilizer."""

    def __init__(self,
                 state_num=4,
                 measure_num=2,
                 cov_process=0.0001,
                 cov_measure=0.1):
        assert state_num == 4 or state_num == 2, "Only scalar and point supported, Check state_num please."

        self.state_num = state_num
        self.measure_num = measure_num
        self.filter = cv2.KalmanFilter(state_num, measure_num, 0)
        
        self.state = np.zeros((state_num, 1), dtype=np.float32)
        self.prediction = np.zeros((state_num, 1), np.float32)
        self.measurement = np.zeros((measure_num, 1), np.float32)

        if self.measure_num == 1:
            self.filter.transitionMatrix = np.array([[1, 1],
                                                     [0, 1]], np.float32)
            self.filter.measurementMatrix = np.array([[1, 1]], np.float32)
            self.filter.processNoiseCov = np.array([[1, 0],
                                                    [0, 1]], np.float32) * cov_process
            self.filter.measurementNoiseCov = np.array([[1]], np.float32) * cov_measure
        else:
            # Kalman parameters are set up for a 2D point.
            self.filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                                     [0, 1, 0, 1],
                                                     [0, 0, 1, 0],
                                                     [0, 0, 0, 1]], np.float32)
            self.filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                                      [0, 1, 0, 0]], np.float32)
            self.filter.processNoiseCov = np.array([[1, 0, 0, 0],
                                                    [0, 1, 0, 0],
                                                    [0, 0, 1, 0],
                                                    [0, 0, 0, 1]], np.float32) * cov_process
            self.filter.measurementNoiseCov = np.array([[1, 0],
                                                        [0, 1]], np.float32) * cov_measure

    def update(self, measurement):
        self.prediction = self.filter.predict()

        if self.measure_num == 1:
            self.measurement[0, 0] = measurement[0]
        else:
            self.measurement[0, 0] = measurement[0]
            self.measurement[1, 0] = measurement[1]

        self.filter.correct(self.measurement)
        self.state = self.filter.statePost



# --- TEST CODE ---
mp = np.array((2, 1), np.float32)  # measurement

def onmouse(k, x, y, s, p):
    global mp
    mp = np.array([[np.float32(x)], [np.float32(y)]])

def main():
    cv2.namedWindow("Kalman Stabilizer Test")
    cv2.setMouseCallback("Kalman Stabilizer Test", onmouse)
    
    # testing with a 2D point (state_num=4, measure_num=2)
    kalman = Stabilizer(4, 2, cov_process=0.01, cov_measure=0.1)
    
    frame = np.zeros((480, 640, 3), np.uint8)  # drawing canvas

    while True:
        # Update the filter with the mouse position
        kalman.update(mp)
        
        # Get the 'predicted' point (before correction)
        point = kalman.prediction
        
        # Get the 'corrected' state (after blending)
        state = kalman.state
        
        # Draw the points
        # A faint trail is left by not clearing the frame
        # cv2.circle(frame, (int(mp[0]), int(mp[1])), 2, (0, 0, 255), -1) # Raw mouse (blue)
        cv2.circle(frame, (int(state[0]), int(state[1])), 2, (0, 0, 255), -1) # Corrected (red)
        cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1) # Predicted (green)
        
        cv2.imshow("Kalman Stabilizer Test", frame)
        
        k = cv2.waitKey(30) & 0xFF
        if k == 27 or k == ord('q'): # Press 'q' or 'ESC' to quit
            break
            
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()