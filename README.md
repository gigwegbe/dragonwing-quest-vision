# Dragonwing Quest Vision

Real-time spatial vision on a Meta Quest 3S, with all of the AI running on a Qualcomm Dragonwing IQ-9075 board. The headset streams its passthrough camera to the board, the board runs the vision model, and the result is drawn back over your view. No cloud, fully offline.

Three modes in one app, switched with a controller button:

- **Detect** object detection with boxes and labels (YOLO26, on the Hexagon NPU).
- **Depth** a depth heatmap, red is near and green is far (Depth Anything V2 Small).
- **Segment** object masks (YOLO26-Seg, on the NPU).

The Quest is just the camera and the display. Every model runs on the IQ-9075.

## What happens

1. The headset captures the passthrough camera frame.
2. It sends the frame to the board over your local Wi-Fi.
3. The board runs whichever mode is active and draws the result onto the image.
4. The processed image comes back and shows on a panel in the headset.

Detection on the NPU runs in roughly 8 ms per frame, segmentation around 7 ms. Depth runs on the CPU.

## Controls

- **X** cycle through detect, depth and segment.
- **A** pause.
- **B** recenter the panel.

Colour code for depth and for the near/far cue: red is near, yellow is mid, green is far.

## The pieces

**On the Quest (Unity app)**

Built on [QuestCameraKit](https://github.com/xrdevrob/QuestCameraKit) by Robert Rodriguez (xrdevrob). The passthrough frames come from Meta's Passthrough Camera API.

Main scripts (in `Assets/Samples/5 ImageLLM/Scripts`):

- `DetectionPanel.cs` captures frames, posts them to the box, draws the returned image, and handles the X, A and B buttons.

Scene: `Assets/Samples/5 ImageLLM/DetectionScene.unity`.

**On the IQ-9075 (box server)**

`detect_server.py` is a FastAPI server on port 8766. The single `/detect_draw` endpoint takes a `mode` parameter (detect, depth or segment) and returns the annotated image.

- Detection and segmentation use the Hexagon NPU through the QNN HTP delegate.
- Detection uses the precompiled YOLO26 model and `yolo_common.py` from Samuel Alexander's work (see Credits).
- Depth uses Depth Anything V2 Small on the CPU.

## Box setup

On the IQ-9075, with the Qualcomm AI stack installed:

```
pip install fastapi uvicorn ai-edge-litert opencv-python numpy
```

Get the models (not included in the repo):

- YOLO26 detection: the precompiled `yolo26n_det_qcs9075.tflite` from Samuel Alexander's repo.
- YOLO26-Seg: export it yourself with the Qualcomm AI Hub tooling (`qai-hub-models install yolo26_seg` first, then export for the QCS9075 target).
- Depth Anything V2 Small.

The QNN TFLite delegate is at `/usr/lib/libQnnTFLiteDelegate.so`, loaded with `backend_type: htp`.

Run:

```
python detect_server.py
```

The server listens on `0.0.0.0:8766`.

## Building the Quest app

1. Open the project in Unity 6000.3.12f1 with the Android module.
2. Open `Assets/Samples/5 ImageLLM/DetectionScene.unity`.
3. In Player Settings: Allow downloads over HTTP set to Always allowed (the app talks to the box over plain HTTP).
4. In Build Profiles, click Add Open Scenes and tick only DetectionScene.
5. Plug in the Quest and Build and Run.

Grant the Headset Cameras permission on first launch (Settings, Privacy and Safety, App Permissions), then restart the app.

## Known limitations and next steps

- The box address (`192.168.1.4:8766`) is hard-coded in `DetectionPanel.cs`. Anyone rebuilding this has to change it to their own board's IP. Automatic discovery on the local network is the obvious next improvement.
- The model files are not in the repo. See Box setup for how to get them.
- An inpainting mode (AOT-GAN) is a possible future addition. It needs a Places2 checkpoint to fill rooms well; the faces-trained CelebA-HQ checkpoint on AI Hub is not suitable.

## Credits and licences

- [QuestCameraKit](https://github.com/xrdevrob/QuestCameraKit) by Robert Rodriguez (xrdevrob) for the passthrough camera access.
- `yolo_common.py` and the precompiled YOLO26 model come from Samuel Alexander's work. Check the licence on his repo before redistributing; his YOLO26 inference code is derived from the Ultralytics YOLO stack, which is AGPL-3.0, so treat the detection path as AGPL-3.0 unless his repo states otherwise.
- Depth Anything V2 and the Qualcomm AI Hub tooling for the on-device models.
