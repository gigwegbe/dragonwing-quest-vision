using System.Collections;
using System.Text;
using Meta.XR;
using Unity.Collections;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Networking;

namespace QuestCameraKit.OpenAI
{
    // Floating panel that streams passthrough frames to the IQ-9075 detection
    // server and shows the annotated result. Two modes, toggled with X:
    //   detect - YOLO boxes with near/far colours
    //   depth  - whole scene coloured by distance
    public class DetectionPanel : MonoBehaviour
    {
        [Header("Server")]
        public string detectUrl = "http://192.168.1.4:8766/detect_draw";
        [Range(0.05f, 0.9f)] public float conf = 0.25f;
        public bool useDepth = true;                 // depth-in-box for detect mode

        [Header("Camera (optional - auto-found if empty)")]
        [SerializeField] private PassthroughCameraAccess cameraAccess;

        [Header("Panel")]
        [SerializeField] private float distance = 0.9f;
        [SerializeField] private float gapSeconds = 0.15f;

        private GameObject _root;
        private RawImage _image;
        private Text _status;
        private Texture2D _display;
        private Transform _head;
        private bool _paused;
        private string _mode = "detect";            // "detect" or "depth"

        private void Awake()
        {
            BuildPanel();
            PlaceInFront();
        }

        private void Start() => StartCoroutine(Loop());

        private void Update()
        {
            if (OVRInput.GetDown(OVRInput.Button.One)) _paused = !_paused;               // A = pause
            if (OVRInput.GetDown(OVRInput.Button.Two)) PlaceInFront();                    // B = recenter
            if (OVRInput.GetDown(OVRInput.Button.Three))                                  // X = cycle mode
                _mode = (_mode == "detect") ? "depth" : (_mode == "depth") ? "segment" : "detect";
        }

        private bool ResolveCamera()
        {
            if (!cameraAccess)
                cameraAccess = FindAnyObjectByType<PassthroughCameraAccess>(FindObjectsInactive.Include);
            return cameraAccess;
        }

        private Texture2D CaptureFrame()
        {
            if (!cameraAccess || !cameraAccess.IsPlaying) return null;
            var res = cameraAccess.CurrentResolution;
            if (res == Vector2Int.zero) return null;
            var colors = cameraAccess.GetColors();
            if (!colors.IsCreated || colors.Length != res.x * res.y) return null;
            var tex = new Texture2D(res.x, res.y, TextureFormat.RGBA32, false);
            tex.SetPixelData(colors, 0);
            tex.Apply();
            return tex;
        }

        private IEnumerator Loop()
        {
            while (true)
            {
                if (_paused || !ResolveCamera() || !cameraAccess.IsPlaying)
                {
                    SetStatus(_paused ? "Paused (A to resume)" : "Waiting for camera...");
                    yield return new WaitForSeconds(0.25f);
                    continue;
                }

                var frame = CaptureFrame();
                if (frame == null) { yield return new WaitForSeconds(0.25f); continue; }

                byte[] jpg = frame.EncodeToJPG(80);
                Destroy(frame);

                string b64 = System.Convert.ToBase64String(jpg);
                string body = "{\"image\":\"" + b64 + "\",\"conf\":" + conf.ToString("0.00") +
                              ",\"depth\":" + (useDepth ? "true" : "false") +
                              ",\"mode\":\"" + _mode + "\"}";

                float t0 = Time.time;
                using (var req = new UnityWebRequest(detectUrl, "POST"))
                {
                    req.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(body));
                    req.downloadHandler = new DownloadHandlerBuffer();
                    req.timeout = 20;
                    req.SetRequestHeader("Content-Type", "application/json");
                    yield return req.SendWebRequest();

                    if (req.result == UnityWebRequest.Result.Success && req.downloadHandler.data.Length > 0)
                    {
                        if (_display == null) _display = new Texture2D(2, 2, TextureFormat.RGB24, false);
                        _display.LoadImage(req.downloadHandler.data);
                        _image.texture = _display;
                        FitImage(_display.width, _display.height);
                        SetStatus($"{_mode.ToUpper()}  {(Time.time - t0) * 1000f:F0} ms  (X to switch)");
                    }
                    else
                    {
                        SetStatus("Server error: " + req.error);
                    }
                }
                yield return new WaitForSeconds(gapSeconds);
            }
        }

        private void BuildPanel()
        {
            var font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");

            _root = new GameObject("DetectionCanvas");
            var canvas = _root.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            _root.GetComponent<RectTransform>().sizeDelta = new Vector2(640, 520);
            _root.transform.localScale = Vector3.one * 0.0006f;

            var bg = NewChild("Background", _root.transform).AddComponent<Image>();
            bg.color = new Color(0f, 0f, 0f, 0.8f);
            var br = bg.rectTransform;
            br.anchorMin = Vector2.zero; br.anchorMax = Vector2.one;
            br.offsetMin = Vector2.zero; br.offsetMax = Vector2.zero;

            _image = NewChild("Frame", _root.transform).AddComponent<RawImage>();
            Place(_image.rectTransform, 20, 20, 600, 450);

            _status = NewChild("Status", _root.transform).AddComponent<Text>();
            _status.font = font; _status.fontSize = 24; _status.color = new Color(1f, 0.85f, 0.3f);
            Place(_status.rectTransform, 20, 478, 600, 36);
            _status.text = "Starting...";
        }

        private void FitImage(int w, int h)
        {
            if (w <= 0 || h <= 0) return;
            float s = Mathf.Min(600f / w, 450f / h);
            _image.rectTransform.sizeDelta = new Vector2(w * s, h * s);
        }

        private void PlaceInFront()
        {
            if (!_head && Camera.main) _head = Camera.main.transform;
            if (!_head) return;
            var fwd = _head.forward; fwd.y = 0;
            if (fwd.sqrMagnitude < 0.001f) fwd = Vector3.forward;
            fwd.Normalize();
            _root.transform.position = _head.position + fwd * distance - Vector3.up * 0.1f;
            _root.transform.rotation = Quaternion.LookRotation(fwd);
        }

        private void SetStatus(string s) { if (_status) _status.text = s; }

        private static GameObject NewChild(string name, Transform parent)
        {
            var go = new GameObject(name, typeof(RectTransform));
            go.transform.SetParent(parent, false);
            return go;
        }

        private static void Place(RectTransform r, float x, float y, float w, float h)
        {
            r.anchorMin = r.anchorMax = new Vector2(0, 1);
            r.pivot = new Vector2(0, 1);
            r.anchoredPosition = new Vector2(x, -y);
            r.sizeDelta = new Vector2(w, h);
        }
    }
}