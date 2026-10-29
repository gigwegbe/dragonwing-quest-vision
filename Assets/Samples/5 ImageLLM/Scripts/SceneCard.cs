using System.Collections;
using UnityEngine;
using UnityEngine.UI;

namespace QuestCameraKit.OpenAI
{
    // Floating card: shows the captured frame, your question, the answer,
    // and a live status line so you can see where the pipeline is.
    public class SceneCard : MonoBehaviour
    {
        public static SceneCard Instance { get; private set; }

        [SerializeField] private float distance = 0.9f;        // metres in front of you
        [SerializeField] private float heightOffset = -0.1f;   // slightly below eye level
        [SerializeField] private float hideAfterSeconds = 15f; // 0 = never hide

        private GameObject _root;
        private RawImage _image;
        private Text _status, _question, _answer;
        private Texture2D _shown;
        private Transform _head;
        private float _sentAt;
        private Coroutine _hideRoutine;

        private void Awake()
        {
            Instance = this;
            Build();
            _root.SetActive(false);
        }

        private void Start()
        {
            var stt = FindAnyObjectByType<SttManager>();
            if (!stt) { Debug.LogWarning("[card] No SttManager found"); return; }

            stt.onRequestStarted.AddListener(() =>
            {
                _question.text = "";
                _answer.text = "";
                _image.texture = null;
                Status("Listening...");
                PlaceInFront();
            });
            stt.onRequestSent.AddListener(() => Status("Transcribing on IQ-9075..."));
            stt.OnTranscriptionComplete += q =>
            {
                _question.text = "You: " + q;
                Status("Heard you. Grabbing camera frame...");
            };
        }

        // Called by ImageOpenAIConnector when the frame goes to the box.
        public void OnFrameSent(Texture2D frame, string prompt, int jpgBytes)
        {
            if (frame)
            {
                if (_shown) Destroy(_shown);
                _shown = CopyTexture(frame);
                _image.texture = _shown;
                Status($"Frame {frame.width}x{frame.height} ({jpgBytes / 1024} KB) sent. Thinking...");
            }
            else
            {
                Status("<color=#ff7070>No camera frame! Check Camera Access.</color> Thinking...");
            }
            _sentAt = Time.time;
        }

        public void OnAnswer(string answer)
        {
            _answer.text = answer;
            Status($"Answered in {Time.time - _sentAt:F1} s");
            ScheduleHide();
        }

        public void OnError(string error)
        {
            Status("<color=#ff7070>Error: " + error + "</color>");
            ScheduleHide();
        }

        private void Status(string text)
        {
            _status.text = text;
            if (!_root.activeSelf) { _root.SetActive(true); PlaceInFront(); }
            if (_hideRoutine != null) { StopCoroutine(_hideRoutine); _hideRoutine = null; }
            Debug.Log("[card] " + text);
        }

        private void ScheduleHide()
        {
            if (hideAfterSeconds <= 0) return;
            if (_hideRoutine != null) StopCoroutine(_hideRoutine);
            _hideRoutine = StartCoroutine(HideLater());
        }

        private IEnumerator HideLater()
        {
            yield return new WaitForSeconds(hideAfterSeconds);
            _root.SetActive(false);
        }

        private void PlaceInFront()
        {
            if (!_head && Camera.main) _head = Camera.main.transform;
            if (!_head) return;
            var fwd = _head.forward; fwd.y = 0;
            if (fwd.sqrMagnitude < 0.001f) fwd = Vector3.forward;
            fwd.Normalize();
            _root.transform.position = _head.position + fwd * distance + Vector3.up * heightOffset;
            _root.transform.rotation = Quaternion.LookRotation(fwd);
        }

        private static Texture2D CopyTexture(Texture2D src)
        {
            var rt = RenderTexture.GetTemporary(src.width, src.height);
            var prev = RenderTexture.active;
            Graphics.Blit(src, rt);
            RenderTexture.active = rt;
            var copy = new Texture2D(src.width, src.height, TextureFormat.RGBA32, false);
            copy.ReadPixels(new Rect(0, 0, src.width, src.height), 0, 0);
            copy.Apply();
            RenderTexture.active = prev;
            RenderTexture.ReleaseTemporary(rt);
            return copy;
        }

        // ---------- build the UI in code ----------
        private void Build()
        {
            var font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");

            _root = new GameObject("SceneCardCanvas");
            var canvas = _root.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            var rt = _root.GetComponent<RectTransform>();
            rt.sizeDelta = new Vector2(600, 800);
            _root.transform.localScale = Vector3.one * 0.0005f; // 30 cm x 40 cm

            var bg = NewChild("Background", _root.transform).AddComponent<Image>();
            bg.color = new Color(0f, 0f, 0f, 0.75f);
            Stretch(bg.rectTransform, 0, 0, 0, 0);

            _image = NewChild("Snapshot", _root.transform).AddComponent<RawImage>();
            Place(_image.rectTransform, 20, 20, 560, 420);

            _status = NewText("Status", font, 24, new Color(1f, 0.85f, 0.3f));
            Place(_status.rectTransform, 20, 450, 560, 60);

            _question = NewText("Question", font, 28, new Color(0.7f, 0.85f, 1f));
            Place(_question.rectTransform, 20, 515, 560, 80);

            _answer = NewText("Answer", font, 28, Color.white);
            Place(_answer.rectTransform, 20, 600, 560, 180);
        }

        private Text NewText(string name, Font font, int size, Color color)
        {
            var t = NewChild(name, _root.transform).AddComponent<Text>();
            t.font = font; t.fontSize = size; t.color = color;
            t.horizontalOverflow = HorizontalWrapMode.Wrap;
            t.verticalOverflow = VerticalWrapMode.Truncate;
            t.supportRichText = true;
            return t;
        }

        private static GameObject NewChild(string name, Transform parent)
        {
            var go = new GameObject(name, typeof(RectTransform));
            go.transform.SetParent(parent, false);
            return go;
        }

        private static void Stretch(RectTransform r, float l, float t, float rr, float b)
        {
            r.anchorMin = Vector2.zero; r.anchorMax = Vector2.one;
            r.offsetMin = new Vector2(l, b); r.offsetMax = new Vector2(-rr, -t);
        }

        // x, y from the top-left corner of the card
        private static void Place(RectTransform r, float x, float y, float w, float h)
        {
            r.anchorMin = r.anchorMax = new Vector2(0, 1);
            r.pivot = new Vector2(0, 1);
            r.anchoredPosition = new Vector2(x, -y);
            r.sizeDelta = new Vector2(w, h);
        }
    }
}