# detect_server.py  -  spatial vision endpoint for the IQ-9075 (Project 2)
# Run from the iq-quest folder with:
#   vr-env/bin/python -m uvicorn detect_server:app --host 0.0.0.0 --port 8766
#
# Needs in this folder:
#   yolo_common.py
#   yolo26n_det_qcs9075.tflite
#   export_assets/yolo26_seg-tflite-float/yolo26_seg.tflite
#
# POST /detect_draw  JSON {"image": b64, "conf": .25, "depth": true,
#                          "mode": "detect"|"depth"|"segment"}  -> annotated JPEG

import base64, time
import numpy as np, cv2
from fastapi import FastAPI, Request
from fastapi.responses import Response
from transformers import pipeline
from PIL import Image
from yolo_common import YoloDetector, nms, COCO
from ai_edge_litert.interpreter import Interpreter, load_delegate

app = FastAPI()

DET_MODEL = "yolo26n_det_qcs9075.tflite"
SEG_MODEL = "export_assets/yolo26_seg-tflite-float/yolo26_seg.tflite"
QNN_DELEGATE = "/usr/lib/libQnnTFLiteDelegate.so"
DEFAULT_CONF = 0.25


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class SegDetector:
    """YOLO26-Seg on the NPU: returns (label, score, box, mask) per instance."""
    def __init__(self, model_path, conf=0.25, backend="htp"):
        if backend:
            dele = load_delegate(QNN_DELEGATE, options={"backend_type": backend})
            self.interp = Interpreter(model_path=model_path, experimental_delegates=[dele])
        else:
            self.interp = Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.outs = self.interp.get_output_details()
        shape = self.inp["shape"]
        self.nchw = shape[1] == 3
        self.ih, self.iw = (shape[2], shape[3]) if self.nchw else (shape[1], shape[2])
        self.conf = conf

    def _out(self, frag, idx):
        for o in self.outs:
            if frag in o["name"]:
                return self.interp.get_tensor(o["index"])
        return self.interp.get_tensor(self.outs[idx]["index"])

    def segment(self, rgb):
        h0, w0 = rgb.shape[:2]
        x = cv2.resize(rgb, (self.iw, self.ih)).astype(np.float32)[None] / 255.0
        if self.nchw:
            x = x.transpose(0, 3, 1, 2)
        self.interp.set_tensor(self.inp["index"], x.astype(self.inp["dtype"]))
        self.interp.invoke()

        boxes = self._out("boxes", 0)[0].astype(np.float32)
        scores = self._out("scores", 1)[0].astype(np.float32)
        coeffs = self._out("coeff", 2)[0].astype(np.float32)
        cls = self._out("class", 3)[0].astype(np.int32)
        protos = self._out("proto", 4)[0].astype(np.float32)   # (160,160,32)

        if scores.min() < 0 or scores.max() > 1.0:
            scores = sigmoid(scores)
        m = scores >= self.conf
        boxes, scores, coeffs, cls = boxes[m], scores[m], coeffs[m], cls[m]
        if len(scores) == 0:
            return []
        keep = nms(boxes, scores)
        boxes, scores, coeffs, cls = boxes[keep], scores[keep], coeffs[keep], cls[keep]

        ph, pw, _ = protos.shape
        masks = sigmoid(coeffs @ protos.reshape(-1, 32).T).reshape(-1, ph, pw)
        sx, sy = w0 / self.iw, h0 / self.ih

        results = []
        for i in range(len(keep)):
            x1, y1, x2, y2 = boxes[i, 0]*sx, boxes[i, 1]*sy, boxes[i, 2]*sx, boxes[i, 3]*sy
            full = cv2.resize(masks[i], (w0, h0)) > 0.5
            crop = np.zeros((h0, w0), bool)
            xa, ya = max(0, int(x1)), max(0, int(y1))
            xb, yb = min(w0, int(x2)), min(h0, int(y2))
            crop[ya:yb, xa:xb] = full[ya:yb, xa:xb]
            label = COCO[cls[i]] if 0 <= cls[i] < len(COCO) else str(cls[i])
            results.append((label, float(scores[i]), (x1, y1, x2, y2), crop))
        return results


print("Loading YOLO26 detection on NPU...")
det = YoloDetector(DET_MODEL, conf=DEFAULT_CONF, backend="htp")
print("Loading YOLO26 segmentation on NPU...")
seg = SegDetector(SEG_MODEL, conf=DEFAULT_CONF, backend="htp")
print("Loading Depth-Anything-V2-Small (CPU)...")
depth_est = pipeline("depth-estimation",
                     model="depth-anything/Depth-Anything-V2-Small-hf", device=-1)

PALETTE = [(80,175,76),(0,152,255),(243,150,33),(54,67,244),(176,39,156),
           (212,188,0),(99,30,233),(57,220,205),(181,81,63),(7,193,255)]


def run_depth(rgb):
    depth = np.array(depth_est(Image.fromarray(rgb))["depth"], dtype=np.float32)
    dmin = float(depth.min()); span = max(float(depth.max()) - dmin, 1e-6)
    return depth, dmin, span


def detect_objects(bgr, want_depth):
    H, W = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = time.time()
    dets = det.detect(rgb)
    yolo_ms = (time.time() - t) * 1000
    depth = dmin = span = None
    if want_depth:
        depth, dmin, span = run_depth(rgb)
    objs = []
    for label, score, (x1, y1, x2, y2) in dets:
        near = None
        if depth is not None:
            dh, dw = depth.shape
            mx1 = max(0, int(x1/W*dw)); mx2 = min(dw, max(mx1+1, int(x2/W*dw)))
            my1 = max(0, int(y1/H*dh)); my2 = min(dh, max(my1+1, int(y2/H*dh)))
            patch = depth[my1:my2, mx1:mx2]
            near = round((float(np.median(patch)) - dmin)/span, 3) if patch.size else None
        objs.append((label, score, (x1, y1, x2, y2), near))
    return objs, round(yolo_ms, 1)


def color_for(near):
    if near is None:   return (0, 255, 0)
    if near >= 0.6:    return (0, 0, 255)
    if near >= 0.3:    return (0, 255, 255)
    return (0, 255, 0)


def decode(data):
    return cv2.imdecode(np.frombuffer(base64.b64decode(data["image"]), np.uint8), cv2.IMREAD_COLOR)


@app.post("/detect_draw")
async def detect_draw(req: Request):
    data = await req.json()
    mode = data.get("mode", "detect")
    conf = float(data.get("conf", DEFAULT_CONF))
    img = decode(data)
    if img is None:
        return Response(content=b"", media_type="image/jpeg")

    if mode == "depth":
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = time.time(); depth, dmin, span = run_depth(rgb); ms = (time.time()-t)*1000
        norm = ((depth - dmin)/span*255).astype(np.uint8)
        out = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
        if out.shape[:2] != img.shape[:2]:
            out = cv2.resize(out, (img.shape[1], img.shape[0]))
        hud = f"DEPTH  {ms:.0f}ms"

    elif mode == "segment":
        seg.conf = conf
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = time.time(); results = seg.segment(rgb); ms = (time.time()-t)*1000
        out = img
        for i, (label, score, (x1, y1, x2, y2), mask) in enumerate(results):
            col = np.array(PALETTE[i % len(PALETTE)], np.uint8)
            out[mask] = (0.5*out[mask] + 0.5*col).astype(np.uint8)
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)),
                          tuple(int(v) for v in col), 2)
            txt = f"{label} {score:.2f}"
            cv2.putText(out, txt, (int(x1)+2, max(14, int(y1)-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        hud = f"SEGMENT  {ms:.0f}ms  {len(results)} obj"

    else:  # detect
        det.conf = conf
        objs, ms = detect_objects(img, bool(data.get("depth", True)))
        out = img
        for label, score, (x1, y1, x2, y2), near in objs:
            col = color_for(near)
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
            near_txt = "" if near is None else f" {near:.2f}"
            txt = f"{label}{near_txt}"
            cv2.putText(out, txt, (int(x1)+2, max(14, int(y1)-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(out, txt, (int(x1)+2, max(14, int(y1)-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        hud = f"DETECT  {ms:.0f}ms  {len(objs)} obj"

    cv2.putText(out, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    print("[" + mode + "] " + hud)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(content=buf.tobytes(), media_type="image/jpeg")