using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Video;

public enum PositionIndex : int
{
    rShldrBend = 0,
    rForearmBend,
    rHand,
    rThumb2,
    rMid1,

    lShldrBend,
    lForearmBend,
    lHand,
    lThumb2,
    lMid1,

    lEar,
    lEye,
    rEar,
    rEye,
    Nose,

    rThighBend,
    rShin,
    rFoot,
    rToe,

    lThighBend,
    lShin,
    lFoot,
    lToe,

    abdomenUpper,

    //Calculated coordinates
    hip,
    head,
    neck,
    spine,

    Count,
    None,
}

public static partial class EnumExtend
{
    public static int Int(this PositionIndex i)
    {
        return (int)i;
    }
}

public class ThreeDPoseScript : MonoBehaviour
{
    public class JointPoint
    {
        public Vector2 Pos2D = new Vector2();
        public float score2D;

        public Vector3 Pos3D = new Vector3();
        public Vector3 Now3D = new Vector3();
        public Vector3 PrevPos3D = new Vector3();
        public float score3D;

        // Bones
        public Transform Transform = null;
        public Quaternion InitRotation;
        public Quaternion Inverse;

        public JointPoint Child = null;
    }

    // Joint position and bone
    private JointPoint[] jointPoints;

    private Vector3 initPosition; // Initial center position

    private Quaternion InitGazeRotation;
    private Quaternion gazeInverse;

    // UnityChan
    public GameObject UnityChan;
    public GameObject Nose;
    private Animator anim;

    // For camera play
    public bool UseWebCam = true;
    private WebCamTexture webCamTexture;

    // For video play
    private RenderTexture videoTexture;

    private Texture2D texture;
    private int videoScreenWidth = 2560;
    private float videoWidth, videoHeight;
    private UnityEngine.Rect clipRect;
    public float clipScale;

    // Properties for onnx and estimation
    private Net Onnx;
    private Mat[] outputs = new Mat[4];

    public GameObject TextureObject;

    private const int inputImageSize = 224;
    private const int JointNum = 24;
    private const int HeatMapCol = 14;
    private const int HeatMapCol_Squared = 14 * 14;
    private const int HeatMapCol_Cube = 14 * 14 * 14;

    char[] heatMap2Dbuf = new char[JointNum * HeatMapCol_Squared * 4];
    float[] heatMap2D = new float[JointNum * HeatMapCol_Squared];
    char[] offset2Dbuf = new char[JointNum * HeatMapCol_Squared * 2 * 4];
    float[] offset2D = new float[JointNum * HeatMapCol_Squared * 2];

    char[] heatMap3Dbuf = new char[JointNum * HeatMapCol_Cube * 4];
    float[] heatMap3D = new float[JointNum * HeatMapCol_Cube];
    char[] offset3Dbuf = new char[JointNum * HeatMapCol_Cube * 3 * 4];
    float[] offset3D = new float[JointNum * HeatMapCol_Cube * 3];

    void Start()
    {
        jointPoints = new JointPoint[PositionIndex.Count.Int()];
        for (var i = 0; i < PositionIndex.Count.Int(); i++) jointPoints[i] = new JointPoint();

        anim = UnityChan.GetComponent<Animator>();

        if (UseWebCam)
        {
            CameraPlayStart();
        }
        else
        {
            VideoPlayStart();
        }

        // Clip size
        videoWidth = texture.width;
        videoHeight = texture.height;
        float padWidth = (videoWidth < videoHeight) ? 0 : (videoHeight - videoWidth) / 2;
        float padHeight = (videoWidth < videoHeight) ? (videoWidth - videoHeight) / 2 : 0;
        if (clipScale == 0f) clipScale = 0.001f;
        var w = (videoWidth + padWidth * 2f) * clipScale;
        padWidth += w;
        padHeight += w;
        clipRect = new UnityEngine.Rect(-padWidth, -padHeight, videoWidth + padWidth * 2, videoHeight + padHeight * 2);

        InitONNX();
        Init();
    }

    void Init()
    {
        // Right Arm
        jointPoints[PositionIndex.rShldrBend.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.RightUpperArm);
        jointPoints[PositionIndex.rForearmBend.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.RightLowerArm);
        jointPoints[PositionIndex.rHand.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.RightHand);
        jointPoints[PositionIndex.rThumb2.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.RightThumbIntermediate);
        jointPoints[PositionIndex.rMid1.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.RightMiddleProximal);
        // Left Arm
        jointPoints[PositionIndex.lShldrBend.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.LeftUpperArm);
        jointPoints[PositionIndex.lForearmBend.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.LeftLowerArm);
        jointPoints[PositionIndex.lHand.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.LeftHand);
        jointPoints[PositionIndex.lThumb2.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.LeftThumbIntermediate);
        jointPoints[PositionIndex.lMid1.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.LeftMiddleProximal);

        // Face
        jointPoints[PositionIndex.lEar.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.Head);
        jointPoints[PositionIndex.lEye.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.LeftEye);
        jointPoints[PositionIndex.rEar.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.Head);
        jointPoints[PositionIndex.rEye.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.RightEye);
        jointPoints[PositionIndex.Nose.Int()].Transform = Nose.transform;

        // Right Leg
        jointPoints[PositionIndex.rThighBend.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.RightUpperLeg);
        jointPoints[PositionIndex.rShin.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.RightLowerLeg);
        jointPoints[PositionIndex.rFoot.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.RightFoot);
        jointPoints[PositionIndex.rToe.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.RightToes);

        // Left Leg
        jointPoints[PositionIndex.lThighBend.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.LeftUpperLeg);
        jointPoints[PositionIndex.lShin.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.LeftLowerLeg);
        jointPoints[PositionIndex.lFoot.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.LeftFoot);
        jointPoints[PositionIndex.lToe.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.LeftToes);

        // etc
        jointPoints[PositionIndex.abdomenUpper.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.Spine);
        jointPoints[PositionIndex.hip.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.Hips);
        jointPoints[PositionIndex.head.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.Head);
        jointPoints[PositionIndex.neck.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.Neck);
        jointPoints[PositionIndex.spine.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.Spine);

        // Child Settings
        // Right Arm
        jointPoints[PositionIndex.rShldrBend.Int()].Child = jointPoints[PositionIndex.rForearmBend.Int()];
        jointPoints[PositionIndex.rForearmBend.Int()].Child = jointPoints[PositionIndex.rHand.Int()];

        // Left Arm
        jointPoints[PositionIndex.lShldrBend.Int()].Child = jointPoints[PositionIndex.lForearmBend.Int()];
        jointPoints[PositionIndex.lForearmBend.Int()].Child = jointPoints[PositionIndex.lHand.Int()];

        // Fase

        // Right Leg
        jointPoints[PositionIndex.rThighBend.Int()].Child = jointPoints[PositionIndex.rShin.Int()];
        jointPoints[PositionIndex.rShin.Int()].Child = jointPoints[PositionIndex.rFoot.Int()];
        jointPoints[PositionIndex.rFoot.Int()].Child = jointPoints[PositionIndex.rToe.Int()];

        // Left Leg
        jointPoints[PositionIndex.lThighBend.Int()].Child = jointPoints[PositionIndex.lShin.Int()];
        jointPoints[PositionIndex.lShin.Int()].Child = jointPoints[PositionIndex.lFoot.Int()];
        jointPoints[PositionIndex.lFoot.Int()].Child = jointPoints[PositionIndex.lToe.Int()];

        // etc
        jointPoints[PositionIndex.spine.Int()].Child = jointPoints[PositionIndex.neck.Int()];
        jointPoints[PositionIndex.neck.Int()].Child = jointPoints[PositionIndex.head.Int()];
        //jointPoints[PositionIndex.head.Int()].Child = jointPoints[PositionIndex.Nose.Int()];

        // Set Inverse
        foreach (var jointPoint in jointPoints)
        {
            if (jointPoint.Transform != null)
            {
                jointPoint.InitRotation = jointPoint.Transform.rotation;
            }

            if (jointPoint.Child != null)
            {
                jointPoint.Inverse = GetInverse(jointPoint, jointPoint.Child);
            }
        }
        initPosition = jointPoints[PositionIndex.hip.Int()].Transform.position;
        var forward = TriangleNormal(jointPoints[PositionIndex.hip.Int()].Transform.position, jointPoints[PositionIndex.lThighBend.Int()].Transform.position, jointPoints[PositionIndex.rThighBend.Int()].Transform.position);
        jointPoints[PositionIndex.hip.Int()].Inverse = Quaternion.Inverse(Quaternion.LookRotation(forward));

        // For Head Rotation
        jointPoints[PositionIndex.head.Int()].InitRotation = jointPoints[PositionIndex.head.Int()].Transform.rotation;
        var gaze = jointPoints[PositionIndex.Nose.Int()].Transform.position - jointPoints[PositionIndex.head.Int()].Transform.position;
        jointPoints[PositionIndex.head.Int()].Inverse = Quaternion.Inverse(Quaternion.LookRotation(gaze));

        jointPoints[PositionIndex.lHand.Int()].InitRotation = jointPoints[PositionIndex.lHand.Int()].Transform.rotation;
        jointPoints[PositionIndex.lHand.Int()].Inverse = Quaternion.Inverse(Quaternion.LookRotation(jointPoints[PositionIndex.lThumb2.Int()].Transform.position - jointPoints[PositionIndex.lMid1.Int()].Transform.position));

        jointPoints[PositionIndex.rHand.Int()].InitRotation = jointPoints[PositionIndex.rHand.Int()].Transform.rotation;
        jointPoints[PositionIndex.rHand.Int()].Inverse = Quaternion.Inverse(Quaternion.LookRotation(jointPoints[PositionIndex.rThumb2.Int()].Transform.position - jointPoints[PositionIndex.rMid1.Int()].Transform.position));
    }

    Vector3 TriangleNormal(Vector3 a, Vector3 b, Vector3 c)
    {
        Vector3 d1 = a - b;
        Vector3 d2 = a - c;

        Vector3 dd = Vector3.Cross(d1, d2);
        dd.Normalize();

        return dd;
    }

    private Quaternion GetInverse(JointPoint p1, JointPoint p2)
    {
        return Quaternion.Inverse(Quaternion.LookRotation(p1.Transform.position - p2.Transform.position));
    }


    private void CameraPlayStart()
    {
        WebCamDevice[] devices = WebCamTexture.devices;
        webCamTexture = new WebCamTexture(devices[0].name);

        GameObject videoScreen = GameObject.Find("VideoScreen");
        RawImage screen = videoScreen.GetComponent<RawImage>();
        var sd = screen.GetComponent<RectTransform>();
        screen.texture = webCamTexture;

        webCamTexture.Play();

        sd.sizeDelta = new Vector2(videoScreenWidth, (int)(videoScreenWidth * webCamTexture.height / webCamTexture.width));

        texture = new Texture2D(webCamTexture.width, webCamTexture.height);
    }

    private void VideoPlayStart()
    {
        var obj = GameObject.Find("Video Player");
        VideoPlayer videoPlayer = obj.GetComponent<VideoPlayer>();

        videoTexture = new RenderTexture((int)videoPlayer.clip.width, (int)videoPlayer.clip.height, 24);

        videoPlayer.renderMode = VideoRenderMode.RenderTexture;
        videoPlayer.targetTexture = videoTexture;

        GameObject videoScreen = GameObject.Find("VideoScreen");
        RawImage screen = videoScreen.GetComponent<RawImage>();
        var sd = screen.GetComponent<RectTransform>();
        sd.sizeDelta = new Vector2(videoScreenWidth, (int)(videoScreenWidth * videoPlayer.clip.height / videoPlayer.clip.width));
        screen.texture = videoTexture;

        videoPlayer.Play();

        texture = new Texture2D(videoTexture.width, videoTexture.height);
    }

    void Update()
    {
        if (UseWebCam)
        {
            if (webCamTexture != null)
            {
                Color32[] color32 = webCamTexture.GetPixels32();
                texture.SetPixels32(color32);
                texture.Apply();
            }
        }
        else
        {
            if (videoTexture != null)
            {
                Graphics.SetRenderTarget(videoTexture);
                texture.ReadPixels(new UnityEngine.Rect(0, 0, videoTexture.width, videoTexture.height), 0, 0);
                texture.Apply();
                Graphics.SetRenderTarget(null);
            }
        }

        StartCoroutine("PoseUpdate", texture);

    }

    IEnumerator PoseUpdate(Texture2D texture)
    {
        var img = ResizeTexture(texture);

        Predict(img);

        // センターの移動と回転
        var forward = TriangleNormal(jointPoints[PositionIndex.hip.Int()].Pos3D, jointPoints[PositionIndex.lThighBend.Int()].Pos3D, jointPoints[PositionIndex.rThighBend.Int()].Pos3D);
        jointPoints[PositionIndex.hip.Int()].Transform.position = jointPoints[PositionIndex.hip.Int()].Pos3D * 0.01f + new Vector3(initPosition.x, 0f, initPosition.z);
        jointPoints[PositionIndex.hip.Int()].Transform.rotation = Quaternion.LookRotation(forward) * jointPoints[PositionIndex.hip.Int()].Inverse * jointPoints[PositionIndex.hip.Int()].InitRotation;

        // 各ボーンの回転
        foreach (var jointPoint in jointPoints)
        {
            if (jointPoint.Child != null)
            {

                jointPoint.Transform.rotation = Quaternion.LookRotation(jointPoint.Pos3D - jointPoint.Child.Pos3D, forward) * jointPoint.Inverse * jointPoint.InitRotation;
            }
        }
        
        // Head Rotation
        var gaze = jointPoints[PositionIndex.Nose.Int()].Pos3D - jointPoints[PositionIndex.head.Int()].Pos3D;
        var f = TriangleNormal(jointPoints[PositionIndex.Nose.Int()].Pos3D, jointPoints[PositionIndex.rEar.Int()].Pos3D, jointPoints[PositionIndex.lEar.Int()].Pos3D);
        var head = jointPoints[PositionIndex.head.Int()];
        head.Transform.rotation = Quaternion.LookRotation(gaze, f) * head.Inverse * head.InitRotation;

        // Wrist rotation (Test code)
        var lf = TriangleNormal(jointPoints[PositionIndex.lHand.Int()].Pos3D, jointPoints[PositionIndex.lMid1.Int()].Pos3D, jointPoints[PositionIndex.lThumb2.Int()].Pos3D);
        var lHand = jointPoints[PositionIndex.lHand.Int()];
        lHand.Transform.rotation = Quaternion.LookRotation(jointPoints[PositionIndex.lThumb2.Int()].Pos3D - jointPoints[PositionIndex.lMid1.Int()].Pos3D, lf) * lHand.Inverse * lHand.InitRotation;
        var rf = TriangleNormal(jointPoints[PositionIndex.rHand.Int()].Pos3D, jointPoints[PositionIndex.rThumb2.Int()].Pos3D, jointPoints[PositionIndex.rMid1.Int()].Pos3D);
        var rHand = jointPoints[PositionIndex.rHand.Int()];
        rHand.Transform.rotation = Quaternion.LookRotation(jointPoints[PositionIndex.rThumb2.Int()].Pos3D - jointPoints[PositionIndex.rMid1.Int()].Pos3D, rf) * rHand.Inverse * rHand.InitRotation;

        yield return null;
    }

    public void InitONNX()
    {
        Onnx = Net.ReadNetFromONNX(Application.dataPath + @"\MobileNet3D2.onnx");
        for (var i = 0; i < 4; i++) outputs[i] = new Mat();
    }

    /// <summary>
    /// Predict
    /// </summary>
    /// <param name="img"></param>
    public void Predict(Mat img)
    {
        var blob = CvDnn.BlobFromImage(img, 1.0 / 255.0, new OpenCvSharp.Size(inputImageSize, inputImageSize), 0.0, false, false);
        Onnx.SetInput(blob);
        Onnx.Forward(outputs, new string[] { "369", "373", "361", "365" });

        // copy 2D outputs
        Marshal.Copy(outputs[2].Data, heatMap2Dbuf, 0, heatMap2Dbuf.Length);
        Buffer.BlockCopy(heatMap2Dbuf, 0, heatMap2D, 0, heatMap2Dbuf.Length);
        Marshal.Copy(outputs[3].Data, offset2Dbuf, 0, offset2Dbuf.Length);
        Buffer.BlockCopy(offset2Dbuf, 0, offset2D, 0, offset2Dbuf.Length);
        for (var j = 0; j < JointNum; j++)
        {
            var maxXIndex = 0;
            var maxYIndex = 0;
            jointPoints[j].score2D = 0.0f;
            for (var y = 0; y < HeatMapCol; y++)
            {
                for (var x = 0; x < HeatMapCol; x++)
                {
                    var l = new List<int>();
                    var v = heatMap2D[(HeatMapCol_Squared) * j + HeatMapCol * y + x];

                    if (v > jointPoints[j].score2D)
                    {
                        jointPoints[j].score2D = v;
                        maxXIndex = x;
                        maxYIndex = y;
                    }
                }

            }

            jointPoints[j].Pos2D.x = (offset2D[HeatMapCol_Squared * j + HeatMapCol * maxYIndex + maxXIndex] + maxXIndex / (float)HeatMapCol) * (float)inputImageSize;
            jointPoints[j].Pos2D.y = (offset2D[HeatMapCol_Squared * (j + JointNum) + HeatMapCol * maxYIndex + maxXIndex] + maxYIndex / (float)HeatMapCol) * (float)inputImageSize;
        }

        // copy 3D outputs
        Marshal.Copy(outputs[0].Data, heatMap3Dbuf, 0, heatMap3Dbuf.Length);
        Buffer.BlockCopy(heatMap3Dbuf, 0, heatMap3D, 0, heatMap3Dbuf.Length);
        Marshal.Copy(outputs[1].Data, offset3Dbuf, 0, offset3Dbuf.Length);
        Buffer.BlockCopy(offset3Dbuf, 0, offset3D, 0, offset3Dbuf.Length);
        for (var j = 0; j < JointNum; j++)
        {
            var maxXIndex = 0;
            var maxYIndex = 0;
            var maxZIndex = 0;
            jointPoints[j].score3D = 0.0f;
            for (var z = 0; z < HeatMapCol; z++)
            {
                for (var y = 0; y < HeatMapCol; y++)
                {
                    for (var x = 0; x < HeatMapCol; x++)
                    {
                        float v = heatMap3D[HeatMapCol_Cube * j + HeatMapCol_Squared * z + HeatMapCol * y + x];
                        if (v > jointPoints[j].score3D)
                        {
                            jointPoints[j].score3D = v;
                            maxXIndex = x;
                            maxYIndex = y;
                            maxZIndex = z;
                        }
                    }
                }
            }

            jointPoints[j].Now3D.x = (offset3D[HeatMapCol_Cube * j + HeatMapCol_Squared * maxZIndex + HeatMapCol * maxYIndex + maxXIndex] + (float)maxXIndex / (float)HeatMapCol) * (float)inputImageSize;
            jointPoints[j].Now3D.y = (float)inputImageSize - (offset3D[HeatMapCol_Cube * (j + JointNum) + HeatMapCol_Squared * maxZIndex + HeatMapCol * maxYIndex + maxXIndex] + (float)maxYIndex / (float)HeatMapCol) * (float)inputImageSize;
            jointPoints[j].Now3D.z = (offset3D[HeatMapCol_Cube * (j + JointNum * 2) + HeatMapCol_Squared * maxZIndex + HeatMapCol * maxYIndex + maxXIndex] + (float)(maxZIndex - 7) / (float)HeatMapCol) * (float)inputImageSize;
        }

        // Calculate hip location
        var lc = (jointPoints[PositionIndex.rThighBend.Int()].Now3D + jointPoints[PositionIndex.lThighBend.Int()].Now3D) / 2f;
        jointPoints[PositionIndex.hip.Int()].Now3D = (jointPoints[PositionIndex.abdomenUpper.Int()].Now3D + lc) / 2f;
        // Calculate neck location
        jointPoints[PositionIndex.neck.Int()].Now3D = (jointPoints[PositionIndex.rShldrBend.Int()].Now3D + jointPoints[PositionIndex.lShldrBend.Int()].Now3D) / 2f;
        // Calculate head location
        var cEar = (jointPoints[PositionIndex.rEar.Int()].Now3D + jointPoints[PositionIndex.lEar.Int()].Now3D) / 2f;
        var hv = cEar - jointPoints[PositionIndex.neck.Int()].Now3D;
        var nhv = Vector3.Normalize(hv);
        var nv = jointPoints[PositionIndex.Nose.Int()].Now3D - jointPoints[PositionIndex.neck.Int()].Now3D;
        jointPoints[PositionIndex.head.Int()].Now3D = jointPoints[PositionIndex.neck.Int()].Now3D + nhv * Vector3.Dot(nhv, nv);
        // Calculate spine location
        jointPoints[PositionIndex.spine.Int()].Now3D = jointPoints[PositionIndex.abdomenUpper.Int()].Now3D;

        // Low pass filter
        foreach (var jp in jointPoints)
        {
            jp.Pos3D = jp.PrevPos3D * 0.5f + jp.Now3D * 0.5f;
            jp.PrevPos3D = jp.Pos3D;
        }
    }

    /// <summary>
    /// Resize Texture and Convrt to Mat
    /// </summary>
    /// <param name="src"></param>
    /// <returns></returns>
    private Mat ResizeTexture(Texture2D src)
    {
        float bbLeft = clipRect.xMin;
        float bbRight = clipRect.xMax;
        float bbTop = clipRect.yMin;
        float bbBottom = clipRect.yMax;
        float bbWidth = clipRect.width;
        float bbHeight = clipRect.height;

        float videoLongSide = (videoWidth > videoHeight) ? videoWidth : videoHeight;
        float videoShortSide = (videoWidth > videoHeight) ? videoHeight : videoWidth;
        float aspectWidth = videoWidth / videoShortSide;
        float aspectHeight = videoHeight / videoShortSide;

        float left = bbLeft;
        float right = bbRight;
        float top = bbTop;
        float bottom = bbBottom;

        left /= videoShortSide;
        right /= videoShortSide;
        top /= videoShortSide;
        bottom /= videoShortSide;

        src.filterMode = FilterMode.Trilinear;
        src.Apply(true);

        RenderTexture rt = new RenderTexture(224, 224, 32);
        Graphics.SetRenderTarget(rt);
        GL.LoadPixelMatrix(left, right, bottom, top);
        GL.Clear(true, true, new Color(0, 0, 0, 0));
        Graphics.DrawTexture(new UnityEngine.Rect(0, 0, aspectWidth, aspectHeight), src);

        UnityEngine.Rect dstRect = new UnityEngine.Rect(0, 0, 224, 224);
        Texture2D dst = (Texture2D)TextureObject.GetComponent<Renderer>().material.mainTexture;
        dst.ReadPixels(dstRect, 0, 0, true);
        Graphics.SetRenderTarget(null);
        Destroy(rt);

        dst.Apply();

        TextureObject.GetComponent<Renderer>().material.mainTexture = dst;

        // Convrt to Mat
        Color32[] c = dst.GetPixels32();
        var m = new Mat(224, 224, MatType.CV_8UC3);
        var videoSourceImageData = new Vec3b[224*224];
        for (var i = 0; i < 224; i++)
        {
            for (var j = 0; j < 224; j++)
            {
                var col = c[j + i * 224];
                var vec3 = new Vec3b
                {
                    Item0 = col.b,
                    Item1 = col.g,
                    Item2 = col.r
                };
                videoSourceImageData[j + i * 224] = vec3;
            }
        }
        m.SetArray(0, 0, videoSourceImageData);

        return m.Flip(FlipMode.X);
    }
}
