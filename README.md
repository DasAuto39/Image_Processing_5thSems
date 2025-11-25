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


3. Controlling the character in unity
   - Controller.cs

The Controller.cs responsible the TCP client. It listens for incoming tracking data, parses the coordinates, and maps them to the Live2D Cubism parameters. It also handles the calibration logic to ensure the raw input values correspond proportionally to the avatar's movement range.

## Steps to Run

1. **Setup Unity Project**
   - Open Unity Hub and create a new project.
   - Select the **3D (Universal Render Pipeline)** or Standard 3D template.

2. **Import Assets**
   - Import the **Live2D Cubism SDK** into your project.
   - Import your Live2D model files (`.moc3`, `.model3.json`, etc.) into the Assets folder.

3. **Scene Setup**
   - Drag your Live2D model prefab into the scene hierarchy.
   - Position the model so it is visible in front of the Main Camera.

4. **Script Configuration**
   - Create a new folder named `Scripts` inside your model's directory.
   - Place the `Controller.cs` file into this folder.
   - Attach the `Controller.cs` script to your Live2D model GameObject.

5. **Optimization**
   - In the Inspector view of your model, locate the **Cubism Pose Controller** component and uncheck/disable it to prevent conflicts with our custom controller.

6. **Running the Application**
   - Press the **Play** button in Unity to start the server listener.

7. **Start Tracking**
   - Open your terminal or command prompt.
   - Navigate to the directory where the Python scripts are saved.
   - Run the following command:
     ```bash
     python main.py --connect
     ```

8. **Result**
   - The connection will be established, and the character should now move in sync with your real-world movements.

