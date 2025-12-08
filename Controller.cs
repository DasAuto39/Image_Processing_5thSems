using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Live2D.Cubism.Core;
using Live2D.Cubism.Framework;
using System;
using System.Threading;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Globalization;

public class VTuberController : MonoBehaviour
{
    private CubismModel model;

    public bool showDebugLogs = true;

    private Thread receiveThread;
    private TcpListener listener;
    private TcpClient client;
    public int port = 5066;
    private bool isRunning = false;

// Model parameters
    private CubismParameter p_AngleX, p_AngleY, p_AngleZ;
    private CubismParameter p_EyeL, p_EyeR, p_BallX, p_BallY;
    private CubismParameter p_Mouth, p_MouthForm, p_Cheek, p_Breath;
    private CubismParameter p_ParamA, p_ParamI, p_ParamU, p_ParamE, p_ParamO;

    private CubismParameter p_BodyRotationZ;

    private CubismParameter p_ArmLB_Shoulder, p_ArmLB_Elbow;
    private CubismParameter p_ArmRB_Shoulder, p_ArmRB_Elbow, p_ArmRB_Y;

    private CubismPart part_ArmLA, part_ArmRA;
    private CubismPart part_ArmLB, part_ArmRB;

// Parameter IDs
    public string ID_AngleX = "ParamAngleX";
    public string ID_AngleY = "ParamAngleY";
    public string ID_AngleZ = "ParamAngleZ";
    public string ID_BodyRotationZ = "Body_Rotation_Z"; 

    public string ID_EyeLOpen = "ParamEyeLOpen";
    public string ID_EyeROpen = "ParamEyeROpen";
    public string ID_BallX = "ParamEyeBallX";
    public string ID_BallY = "ParamEyeBallY";
    public string ID_MouthOpen = "ParamMouthOpenY";
    public string ID_MouthForm = "ParamMouthForm";
    public string ID_Cheek = "ParamCheek";
    public string ID_Breath = "ParamBreath";

    public string ID_ParamA = "ParamA";
    public string ID_ParamI = "ParamI";
    public string ID_ParamU = "ParamU";
    public string ID_ParamE = "ParamE";
    public string ID_ParamO = "ParamO";

    public string ID_ArmLB_Shoulder = "ParamArmLB01";
    public string ID_ArmLB_Elbow = "ParamArmLB02";
    public string ID_ArmRB_Shoulder = "ParamArmRB01";
    public string ID_ArmRB_Elbow = "ParamArmRB02";
    public string ID_ArmRB_Y = "ParamArmRB02Y";

    public string ID_Part_ArmLA = "PartArmLA";
    public string ID_Part_ArmLB = "PartArmLB";
    public string ID_Part_ArmRA = "PartArmRA";
    public string ID_Part_ArmRB = "PartArmRB";

    // Configuration parameters
    public Vector2 shoulder_offset_L = new Vector2(-0.3f, 0.2f);
    public Vector2 shoulder_offset_R = new Vector2(0.3f, 0.2f);
    public float arm_length_scale = 0.8f;

    public float ear_max_threshold = 0.5f;
    public float ear_min_threshold = 0.1f;
    public float iris_left_ceiling = 0.0f;
    public float iris_right_ceiling = 1.0f;
    public float iris_up_ceiling = 1.0f;
    public float iris_down_ceiling = 0.0f;
    public float mar_max_threshold = 1.0f;
    public float mar_min_threshold = 0.0f;
    public float mouth_width_max = 1.0f;
    public float mouth_width_min = 0.0f;

    public float hand_y_min = -1.0f;
    public float hand_y_max = 0.2f;
    public float elbow_bend_max = 30.0f;
    public float arm_switch_threshold = -0.5f;

    private volatile float roll, pitch, yaw;
    private volatile float ear_left, ear_right;
    private volatile float x_ratio_left, y_ratio_left, x_ratio_right, y_ratio_right;
    private volatile float mar, mouth_dist;
    private volatile float hand_l_x, hand_l_y, hand_r_x, hand_r_y;
    private float t1;

    private bool isLeftBent = false;
    private bool isRightBent = false;
    public float hysteresis_gap = 0.1f;

    void Start()
    {
        model = this.FindCubismModel();
        if (model == null) return;
        CacheParameters();
        InitTCP();
    }
// Finding model parameters and parts
    private void CacheParameters()
    {
        CubismParameter FindParam(string id) => model.Parameters.FindById(id);
        p_AngleX = FindParam(ID_AngleX);
        p_AngleY = FindParam(ID_AngleY);
        p_AngleZ = FindParam(ID_AngleZ);
        p_BodyRotationZ = FindParam(ID_BodyRotationZ); 

        p_EyeL = FindParam(ID_EyeLOpen);
        p_EyeR = FindParam(ID_EyeROpen);
        p_BallX = FindParam(ID_BallX);
        p_BallY = FindParam(ID_BallY);
        p_Mouth = FindParam(ID_MouthOpen);
        p_MouthForm = FindParam(ID_MouthForm);
        p_Cheek = FindParam(ID_Cheek);
        p_Breath = FindParam(ID_Breath);

        p_ParamA = FindParam(ID_ParamA);
        p_ParamI = FindParam(ID_ParamI);
        p_ParamU = FindParam(ID_ParamU);
        p_ParamE = FindParam(ID_ParamE);
        p_ParamO = FindParam(ID_ParamO);

        p_ArmLB_Shoulder = FindParam(ID_ArmLB_Shoulder);
        p_ArmLB_Elbow = FindParam(ID_ArmLB_Elbow);
        p_ArmRB_Shoulder = FindParam(ID_ArmRB_Shoulder);
        p_ArmRB_Elbow = FindParam(ID_ArmRB_Elbow);
        p_ArmRB_Y = FindParam(ID_ArmRB_Y);

        part_ArmLA = model.Parts.FindById(ID_Part_ArmLA);
        part_ArmLB = model.Parts.FindById(ID_Part_ArmLB);
        part_ArmRA = model.Parts.FindById(ID_Part_ArmRA);
        part_ArmRB = model.Parts.FindById(ID_Part_ArmRB);
    }
// Initialize TCP connection and start receiving thread
    private void InitTCP()
    {
        isRunning = true;
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();
    }
// TCP data receiving thread
    private void ReceiveData()
    {
        try
        {
            listener = new TcpListener(IPAddress.Parse("127.0.0.1"), port);
            listener.Start();
            Byte[] bytes = new Byte[2048];

            while (isRunning)
            {
                if (!listener.Pending()) { Thread.Sleep(5); continue; }

                using (client = listener.AcceptTcpClient())
                using (NetworkStream stream = client.GetStream())
                {
                    int length;
                    while ((length = stream.Read(bytes, 0, bytes.Length)) != 0)
                    {
                        byte[] data = new byte[length];
                        Array.Copy(bytes, 0, data, 0, length);
                        string msg = Encoding.ASCII.GetString(data);
                        string[] res = msg.Split(' ');
                        try
                        {
                            if (res.Length > 10)
                            {
                                float.TryParse(res[0], NumberStyles.Any, CultureInfo.InvariantCulture, out roll);
                                float.TryParse(res[1], NumberStyles.Any, CultureInfo.InvariantCulture, out pitch);
                                float.TryParse(res[2], NumberStyles.Any, CultureInfo.InvariantCulture, out yaw);
                                float.TryParse(res[3], NumberStyles.Any, CultureInfo.InvariantCulture, out ear_left);
                                float.TryParse(res[4], NumberStyles.Any, CultureInfo.InvariantCulture, out ear_right);
                                float.TryParse(res[5], NumberStyles.Any, CultureInfo.InvariantCulture, out x_ratio_left);
                                float.TryParse(res[6], NumberStyles.Any, CultureInfo.InvariantCulture, out y_ratio_left);
                                float.TryParse(res[7], NumberStyles.Any, CultureInfo.InvariantCulture, out x_ratio_right);
                                float.TryParse(res[8], NumberStyles.Any, CultureInfo.InvariantCulture, out y_ratio_right);
                                float.TryParse(res[9], NumberStyles.Any, CultureInfo.InvariantCulture, out mar);
                                float.TryParse(res[10], NumberStyles.Any, CultureInfo.InvariantCulture, out mouth_dist);
                            }
                            if (res.Length > 14)
                            {
                                float.TryParse(res[11], NumberStyles.Any, CultureInfo.InvariantCulture, out hand_l_x);
                                float.TryParse(res[12], NumberStyles.Any, CultureInfo.InvariantCulture, out hand_l_y);
                                float.TryParse(res[13], NumberStyles.Any, CultureInfo.InvariantCulture, out hand_r_x);
                                float.TryParse(res[14], NumberStyles.Any, CultureInfo.InvariantCulture, out hand_r_y);
                            }
                        }
                        catch { }
                    }
                }
            }
        }
        catch (Exception) { }
    }
//  Update function for applying parameters to the model
    void LateUpdate()
    {
        if (model == null) return;

        if (p_AngleX != null) p_AngleX.Value = -Mathf.Clamp(yaw, -30, 30);
        if (p_AngleY != null) p_AngleY.Value = Mathf.Clamp(pitch, -30, 30);
        if (p_AngleZ != null) p_AngleZ.Value = -Mathf.Clamp(roll, -30, 30);

        if (p_BodyRotationZ != null) p_BodyRotationZ.Value = -Mathf.Clamp(roll, -10, 10);

        t1 += Time.deltaTime;
        if (p_Breath != null) p_Breath.Value = (Mathf.Sin(t1 * 3f) + 1) * 0.5f;

        IrisMovement();
        UpdateMouthVowels();
        EyeBlinking();
        UpdateArms();     
        SwitchArmParts(); 

        if (showDebugLogs && Time.frameCount % 60 == 0)
        {
            Debug.Log($"[HAND L] Y: {hand_l_y:F2} | [HAND R] Y: {hand_r_y:F2}");
        }
    }
// Update eye blinking parameters based on EAR
    void EyeBlinking()
    {
        float valL = Mathf.Clamp(ear_left, ear_min_threshold, ear_max_threshold);
        float eye_L_value = (valL - ear_min_threshold) / (ear_max_threshold - ear_min_threshold);
        if (p_EyeL != null) p_EyeL.Value = Mathf.Clamp01(eye_L_value);

        float valR = Mathf.Clamp(ear_right, ear_min_threshold, ear_max_threshold);
        float eye_R_value = (valR - ear_min_threshold) / (ear_max_threshold - ear_min_threshold);
        if (p_EyeR != null) p_EyeR.Value = Mathf.Clamp01(eye_R_value);
    }
// Update mouth vowel parameters based on MAR and mouth width
    void UpdateMouthVowels()
    {
        float open_raw = Mathf.Clamp(mar, mar_min_threshold, mar_max_threshold);
        float open_factor = (open_raw - mar_min_threshold) / (mar_max_threshold - mar_min_threshold);
        float width_raw = Mathf.Clamp(mouth_dist, mouth_width_min, mouth_width_max);
        float width_factor = (width_raw - mouth_width_min) / (mouth_width_max - mouth_width_min);

        if (open_factor < 0.1f)
        {
            if (p_ParamA != null) p_ParamA.Value = 0;
            if (p_ParamI != null) p_ParamI.Value = 0;
            if (p_ParamU != null) p_ParamU.Value = 0;
            if (p_ParamE != null) p_ParamE.Value = 0;
            if (p_ParamO != null) p_ParamO.Value = 0;
            return;
        }

        if (p_ParamA != null) p_ParamA.Value = open_factor;
        if (p_ParamI != null) p_ParamI.Value = (Mathf.InverseLerp(0.6f, 1.0f, width_factor)) * open_factor;
        if (p_ParamU != null) p_ParamU.Value = (Mathf.InverseLerp(0.4f, 0.0f, width_factor)) * open_factor;
        if (p_ParamE != null) p_ParamE.Value = 0;
        if (p_ParamO != null) p_ParamO.Value = 0;
    }
// Update arm positions using Inverse Kinematics
    void UpdateArms()
    {
        ApplyIK(
            new Vector2(hand_l_x, hand_l_y),    
            shoulder_offset_L,
            p_ArmLB_Shoulder,
            p_ArmLB_Elbow,
            true                                
                );

        ApplyIK(
            new Vector2(-hand_r_x, hand_r_y),   
            shoulder_offset_R,
            p_ArmRB_Shoulder,
            p_ArmRB_Elbow,
            true                                
        );
    }
// Inverse Kinematics for arms
    void ApplyIK(Vector2 handPos, Vector2 shoulderPos, CubismParameter p_Shoulder, CubismParameter p_Elbow, bool isLeft)
    {
        Vector2 direction = handPos - shoulderPos;
        float angle = Mathf.Atan2(direction.y, direction.x) * Mathf.Rad2Deg;

        float baseRotationOffset = -90f;
        float finalShoulderAngle = angle - baseRotationOffset;

        if (!isLeft) finalShoulderAngle = 180 - finalShoulderAngle;

        finalShoulderAngle = Mathf.Clamp(finalShoulderAngle, -30, 30);
        if (p_Shoulder != null) p_Shoulder.Value = finalShoulderAngle;

        float dist = direction.magnitude;
        float bendFactor = 1.0f - Mathf.Clamp01(dist / arm_length_scale);
        if (p_Elbow != null) p_Elbow.Value = bendFactor * elbow_bend_max;
    }
    // For switching arm parts based on hand Y position with hysteresis
    void SwitchArmParts()
    {
        float switchPointLeft = isLeftBent ? (arm_switch_threshold - hysteresis_gap) : (arm_switch_threshold + hysteresis_gap);
        if (isLeftBent && hand_r_y < switchPointLeft) isLeftBent = false;
        else if (!isLeftBent && hand_r_y > switchPointLeft) isLeftBent = true;

        float switchPointRight = isRightBent ? (arm_switch_threshold - hysteresis_gap) : (arm_switch_threshold + hysteresis_gap);
        if (isRightBent && hand_l_y < switchPointRight) isRightBent = false;
        else if (!isRightBent && hand_l_y > switchPointRight) isRightBent = true;

        if (part_ArmLA != null) part_ArmLA.Opacity = 0;
        if (part_ArmLB != null) part_ArmLB.Opacity = 1;

        if (part_ArmRA != null) part_ArmRA.Opacity = 0;
        if (part_ArmRB != null) part_ArmRB.Opacity = 1;
    }

    void IrisMovement()
    {
        float eyeball_x = (x_ratio_left + x_ratio_right) / 2;
        float eyeball_y = (y_ratio_left + y_ratio_right) / 2;

        if (p_BallX != null)
        {
            float x = Mathf.Clamp(eyeball_x, iris_left_ceiling, iris_right_ceiling);
            x = (x - iris_left_ceiling) / (iris_right_ceiling - iris_left_ceiling) * 2 - 1;
            p_BallX.Value = Mathf.Pow(x, 3);
        }
        if (p_BallY != null)
        {
            float y = Mathf.Clamp(eyeball_y, iris_down_ceiling, iris_up_ceiling);
            y = (y - iris_down_ceiling) / (iris_up_ceiling - iris_down_ceiling) * 2 - 1;
            p_BallY.Value = Mathf.Pow(y, 3);
        }
    }

    void OnApplicationQuit()
    {
        isRunning = false;
        if (listener != null) listener.Stop();
        if (receiveThread != null) receiveThread.Abort();
    }
}