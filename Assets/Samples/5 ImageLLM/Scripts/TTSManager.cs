using System.Text;
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;

namespace QuestCameraKit.OpenAI
{
    public class TtsManager : MonoBehaviour
    {
        [SerializeField] private AudioPlayer audioPlayer;

        [Header("Box Server (offline)")]
        [Tooltip("Your IQ-9075 Piper endpoint")]
        public string ttsBoxUrl = "http://192.168.1.4:8765/tts";

        private void Awake()
        {
            if (!audioPlayer) audioPlayer = GetComponentInChildren<AudioPlayer>();
        }

        public void SynthesizeAndPlay(string text)
        {
            if (!audioPlayer || string.IsNullOrWhiteSpace(text)) return;
            StartCoroutine(RequestTts(text));
        }

        private IEnumerator RequestTts(string text)
        {
            var payloadJson = "{\"text\":\"" + EscapeJson(text) + "\"}";
            using var request = new UnityWebRequest(ttsBoxUrl, "POST");
            request.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(payloadJson));
            request.downloadHandler = new DownloadHandlerBuffer();
            request.timeout = 60;
            request.SetRequestHeader("Content-Type", "application/json");

            yield return request.SendWebRequest();

            if (request.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError("Box TTS error: " + request.error);
                yield break;
            }

            var audioData = request.downloadHandler.data;
            if (audioData != null && audioData.Length > 0 && this && audioPlayer)
            {
                Debug.Log("Playing audio.");
                audioPlayer.ProcessAudioBytes(audioData);
            }
        }

        private string EscapeJson(string s)
        {
            return s.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\n").Replace("\r", "");
        }
    }
}