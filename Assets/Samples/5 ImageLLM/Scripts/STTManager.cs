using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.Networking;
using UnityEngine.UI;

namespace QuestCameraKit.OpenAI
{
    public class SttManager : MonoBehaviour
    {
        public event Action<string> OnTranscriptionComplete;
        [Header("STT Settings")]
        [SerializeField, Min(1)] private int recordingMaximum = 5;
        [SerializeField] private int sampleRate = 48000;
        [Header("Box Server (offline)")]
        [Tooltip("Your IQ-9075 Whisper endpoint")]
        public string sttBoxUrl = "http://192.168.1.4:8765/stt";
        [Header("Canvas Settings")]
        [SerializeField] private bool useCanvas = true;
        [Header("UI Components (Optional)")]
        [SerializeField] private Button recordButton;
        [SerializeField] private Text transcriptionText;
        [SerializeField] private Dropdown microphoneDropdown;
        [Header("Events")] public UnityEvent onRequestStarted;
        public UnityEvent onRequestSent;

        private ImageOpenAIConnector _imageOpenAIConnector;
        private AudioClip _clip;
        private bool _isRecording;
        private bool _isSending;
        private float _time;
        private UnityWebRequest _request;

        private void Start()
        {
            _imageOpenAIConnector = FindAnyObjectByType<ImageOpenAIConnector>();
            if (!_imageOpenAIConnector)
            {
                Debug.LogError("SttManager requires an ImageOpenAIConnector.");
                enabled = false;
                return;
            }
            if (useCanvas && recordButton) recordButton.onClick.AddListener(ToggleRecording);
        }

        private void ToggleRecording()
        {
            if (!isActiveAndEnabled || _isSending) return;
            if (_isRecording) EndRecording();
            else StartRecording();
        }

        private void StartRecording()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            if (!UnityEngine.Android.Permission.HasUserAuthorizedPermission(UnityEngine.Android.Permission.Microphone))
            {
                Debug.Log("[mic] permission not yet granted, requesting");
                var callbacks = new UnityEngine.Android.PermissionCallbacks();
                callbacks.PermissionGranted += _ => { if (this && isActiveAndEnabled) StartRecording(); };
                callbacks.PermissionDenied += _ => { if (this) SetStatus("Microphone permission denied."); };
                UnityEngine.Android.Permission.RequestUserPermission(UnityEngine.Android.Permission.Microphone, callbacks);
                return;
            }
#endif
            if (_isRecording || _isSending) return;
            var devices = Microphone.devices;
            Debug.Log($"[mic] devices=[{string.Join(", ", devices)}] using default mic at {sampleRate} Hz");
            if (devices.Length == 0)
            {
                SetStatus("No microphone found.");
                return;
            }

            _clip = Microphone.Start(null, false, Mathf.Max(1, recordingMaximum), sampleRate);
            if (!_clip)
            {
                Debug.Log("[mic] Microphone.Start returned null");
                SetStatus("Could not start microphone.");
                return;
            }
            _time = 0f;
            _isRecording = true;
            Debug.Log("[mic] recording started");
            SetStatus("Recording...");
            onRequestStarted?.Invoke();
        }

        private async void EndRecording()
        {
            if (!_isRecording) return;
            var frames = Microphone.GetPosition(null);
            if (frames <= 0 && _clip && _time >= Mathf.Max(1, recordingMaximum)) frames = _clip.samples;
            Microphone.End(null);
            _isRecording = false;
            _isSending = true;
            if (useCanvas && recordButton) recordButton.interactable = false;
            try
            {
                if (!_clip || frames <= 0) throw new InvalidOperationException("No audio was recorded.");

                var peakSamples = new float[_clip.samples * _clip.channels];
                _clip.GetData(peakSamples, 0);
                float peak = 0f;
                foreach (var s in peakSamples) { var a = Mathf.Abs(s); if (a > peak) peak = a; }
                Debug.Log($"[mic] frames={frames} clipSamples={_clip.samples} rate={_clip.frequency} peak={peak:F4}");

                var audio = SaveWav.Save("output.wav", _clip, frames);
                Destroy(_clip);
                _clip = null;
                SetStatus("Processing...");
                onRequestSent?.Invoke();
                var transcription = await SendToBox(audio);
                if (!this || !isActiveAndEnabled) return;
                _imageOpenAIConnector.StopProcessingSound("");
                SetStatus(transcription);
                if (!string.IsNullOrWhiteSpace(transcription)) OnTranscriptionComplete?.Invoke(transcription);
            }
            catch (Exception e)
            {
                if (this && isActiveAndEnabled)
                {
                    if (_imageOpenAIConnector) _imageOpenAIConnector.StopProcessingSound("");
                    SetStatus("Transcription failed: " + e.Message);
                    Debug.LogWarning("Transcription failed: " + e.Message);
                }
            }
            finally
            {
                _isSending = false;
                if (this)
                {
                    if (_clip) Destroy(_clip);
                    _clip = null;
                    if (useCanvas && recordButton) recordButton.interactable = true;
                }
            }
        }

        private async Task<string> SendToBox(byte[] audioData)
        {
            if (audioData == null || audioData.Length == 0 || audioData.Length > 25 * 1024 * 1024)
                throw new InvalidOperationException("Recording must contain between 1 byte and 25 MB.");
            var form = new List<IMultipartFormSection>
            {
                new MultipartFormFileSection("file", audioData, "output.wav", "audio/wav")
            };
            using var request = UnityWebRequest.Post(sttBoxUrl, form);
            request.timeout = 60;
            _request = request;
            try
            {
                await request.SendWebRequest();
                if (request.result != UnityWebRequest.Result.Success)
                    throw new InvalidOperationException($"HTTP {request.responseCode}: {request.error}");
                return request.downloadHandler.text.Trim();
            }
            finally { _request = null; }
        }

        private void SetStatus(string text)
        {
            if (useCanvas && transcriptionText) transcriptionText.text = text;
        }

        private void OnDisable()
        {
            if (_isRecording) Microphone.End(null);
            _isRecording = false;
            _request?.Abort();
            if (_clip) Destroy(_clip);
            _clip = null;
        }

        private void OnDestroy()
        {
            if (recordButton) recordButton.onClick.RemoveListener(ToggleRecording);
        }

        private void Update()
        {
            if (_isRecording)
            {
                _time += Time.deltaTime;
                if (_time >= Mathf.Max(1, recordingMaximum)) EndRecording();
            }
            if (OVRInput.GetDown(OVRInput.Button.Start)) ToggleRecording();
        }
    }
}


// using System;
// using System.Collections.Generic;
// using System.Threading.Tasks;
// using UnityEngine;
// using UnityEngine.Events;
// using UnityEngine.Networking;
// using UnityEngine.UI;

// namespace QuestCameraKit.OpenAI
// {
//     public class SttManager : MonoBehaviour
//     {
//         public event Action<string> OnTranscriptionComplete;
//         [Header("STT Settings")]
//         [SerializeField, Min(1)] private int recordingMaximum = 5;
//         [SerializeField] private int sampleRate = 48000;
//         [Header("Box Server (offline)")]
//         [Tooltip("Your IQ-9075 Whisper endpoint")]
//         public string sttBoxUrl = "http://192.168.1.4:8765/stt";
//         [Header("Canvas Settings")]
//         [SerializeField] private bool useCanvas = true;
//         [Header("UI Components (Optional)")]
//         [SerializeField] private Button recordButton;
//         [SerializeField] private Text transcriptionText;
//         [SerializeField] private Dropdown microphoneDropdown;
//         [Header("Events")] public UnityEvent onRequestStarted;
//         public UnityEvent onRequestSent;

//         private ImageOpenAIConnector _imageOpenAIConnector;
//         private AudioClip _clip;
//         private bool _isRecording;
//         private bool _isSending;
//         private float _time;
//         private UnityWebRequest _request;

//         private void Start()
//         {
//             _imageOpenAIConnector = FindAnyObjectByType<ImageOpenAIConnector>();
//             if (!_imageOpenAIConnector)
//             {
//                 Debug.LogError("SttManager requires an ImageOpenAIConnector.");
//                 enabled = false;
//                 return;
//             }
//             if (useCanvas && recordButton) recordButton.onClick.AddListener(ToggleRecording);
//         }

//         private void ToggleRecording()
//         {
//             if (!isActiveAndEnabled || _isSending) return;
//             if (_isRecording) EndRecording();
//             else StartRecording();
//         }

//         private void StartRecording()
//         {
// #if UNITY_ANDROID && !UNITY_EDITOR
//             if (!UnityEngine.Android.Permission.HasUserAuthorizedPermission(UnityEngine.Android.Permission.Microphone))
//             {
//                 Debug.Log("[mic] permission not yet granted, requesting");
//                 var callbacks = new UnityEngine.Android.PermissionCallbacks();
//                 callbacks.PermissionGranted += _ => { if (this && isActiveAndEnabled) StartRecording(); };
//                 callbacks.PermissionDenied += _ => { if (this) SetStatus("Microphone permission denied."); };
//                 UnityEngine.Android.Permission.RequestUserPermission(UnityEngine.Android.Permission.Microphone, callbacks);
//                 return;
//             }
// #endif
//             if (_isRecording || _isSending) return;
//             var devices = Microphone.devices;
//             Debug.Log($"[mic] devices=[{string.Join(", ", devices)}] using default mic at {sampleRate} Hz");
//             if (devices.Length == 0)
//             {
//                 SetStatus("No microphone found.");
//                 return;
//             }

//             _clip = Microphone.Start(null, false, Mathf.Max(1, recordingMaximum), sampleRate);
//             if (!_clip)
//             {
//                 Debug.Log("[mic] Microphone.Start returned null");
//                 SetStatus("Could not start microphone.");
//                 return;
//             }
//             _time = 0f;
//             _isRecording = true;
//             Debug.Log("[mic] recording started");
//             SetStatus("Recording...");
//             onRequestStarted?.Invoke();
//         }

//         private async void EndRecording()
//         {
//             if (!_isRecording) return;
//             var frames = Microphone.GetPosition(null);
//             if (frames <= 0 && _clip && _time >= Mathf.Max(1, recordingMaximum)) frames = _clip.samples;
//             Microphone.End(null);
//             _isRecording = false;
//             _isSending = true;
//             if (useCanvas && recordButton) recordButton.interactable = false;
//             try
//             {
//                 if (!_clip || frames <= 0) throw new InvalidOperationException("No audio was recorded.");

//                 var peakSamples = new float[_clip.samples * _clip.channels];
//                 _clip.GetData(peakSamples, 0);
//                 float peak = 0f;
//                 foreach (var s in peakSamples) { var a = Mathf.Abs(s); if (a > peak) peak = a; }
//                 Debug.Log($"[mic] frames={frames} clipSamples={_clip.samples} rate={_clip.frequency} peak={peak:F4}");

//                 var audio = SaveWav.Save("output.wav", _clip, frames);
//                 Destroy(_clip);
//                 _clip = null;
//                 SetStatus("Processing...");
//                 onRequestSent?.Invoke();
//                 var transcription = await SendToBox(audio);
//                 if (!this || !isActiveAndEnabled) return;
//                 _imageOpenAIConnector.StopProcessingSound("");
//                 SetStatus(transcription);
//                 if (!string.IsNullOrWhiteSpace(transcription)) OnTranscriptionComplete?.Invoke(transcription);
//             }
//             catch (Exception e)
//             {
//                 if (this && isActiveAndEnabled)
//                 {
//                     if (_imageOpenAIConnector) _imageOpenAIConnector.StopProcessingSound("");
//                     SetStatus("Transcription failed: " + e.Message);
//                     Debug.LogWarning("Transcription failed: " + e.Message);
//                 }
//             }
//             finally
//             {
//                 _isSending = false;
//                 if (this)
//                 {
//                     if (_clip) Destroy(_clip);
//                     _clip = null;
//                     if (useCanvas && recordButton) recordButton.interactable = true;
//                 }
//             }
//         }

//         private async Task<string> SendToBox(byte[] audioData)
//         {
//             if (audioData == null || audioData.Length == 0 || audioData.Length > 25 * 1024 * 1024)
//                 throw new InvalidOperationException("Recording must contain between 1 byte and 25 MB.");
//             var form = new List<IMultipartFormSection>
//             {
//                 new MultipartFormFileSection("file", audioData, "output.wav", "audio/wav")
//             };
//             using var request = UnityWebRequest.Post(sttBoxUrl, form);
//             request.timeout = 60;
//             _request = request;
//             try
//             {
//                 await request.SendWebRequest();
//                 if (request.result != UnityWebRequest.Result.Success)
//                     throw new InvalidOperationException($"HTTP {request.responseCode}: {request.error}");
//                 return request.downloadHandler.text.Trim();
//             }
//             finally { _request = null; }
//         }

//         private void SetStatus(string text)
//         {
//             if (useCanvas && transcriptionText) transcriptionText.text = text;
//         }

//         private void OnDisable()
//         {
//             if (_isRecording) Microphone.End(null);
//             _isRecording = false;
//             _request?.Abort();
//             if (_clip) Destroy(_clip);
//             _clip = null;
//         }

//         private void OnDestroy()
//         {
//             if (recordButton) recordButton.onClick.RemoveListener(ToggleRecording);
//         }

//         private void Update()
//         {
//             if (_isRecording)
//             {
//                 _time += Time.deltaTime;
//                 if (_time >= Mathf.Max(1, recordingMaximum)) EndRecording();
//             }
//             if (OVRInput.GetDown(OVRInput.Button.Start)) ToggleRecording();
//         }
//     }
// }

