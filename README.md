# Detail About the Assignment

There is 2 component for the codebase

1. Extracting the feature of our body information which consist of
   - face_feature.py
   - face_landmark.py
   - hand_tracker.py
   - main.py
   - pose_estimator.py
   - stabilizer.py
  The system detects facial landmarks and hand movements. Before transmission, the data goes through a stabilization layer to smoothing the value for the movement. The    processed data is then sent to Unity via a TCP socket connection on localhost.


2. Controlling the character in unity
   - Controller.cs
    
    The Controller.cs responsible the TCP client. It listens for incoming tracking data, parses the coordinates, and maps them to the Live2D Cubism parameters. It also handles the calibration logic to ensure the raw input values correspond proportionally to the avatar's movement range.



