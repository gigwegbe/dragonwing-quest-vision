"""Shared YOLO26 inference pipeline for the AI Hub-compiled .tflite model.

Model I/O (qai_hub_models export, postprocessing included in the graph):
  input:  [1, 640, 640, 3] float32 RGB, range 0-1
  output: boxes [1, N, 4] xyxy in input-pixel space, scores [1, N], class_idx [1, N]
"""
import zlib

import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter, load_delegate
except ImportError:
    from tflite_runtime.interpreter import Interpreter, load_delegate

# QNN TFLite delegate, ships with the board's libqnn packages. Routes the same
# .tflite to the Hexagon NPU (backend "htp") at runtime.
QNN_DELEGATE = "/usr/lib/libQnnTFLiteDelegate.so"

COCO = (
    "person bicycle car motorcycle airplane bus train truck boat traffic-light "
    "fire-hydrant stop-sign parking-meter bench bird cat dog horse sheep cow "
    "elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee "
    "skis snowboard sports-ball kite baseball-bat baseball-glove skateboard "
    "surfboard tennis-racket bottle wine-glass cup fork knife spoon bowl banana "
    "apple sandwich orange broccoli carrot hot-dog pizza donut cake chair couch "
    "potted-plant bed dining-table toilet tv laptop mouse remote keyboard "
    "cell-phone microwave oven toaster sink refrigerator book clock vase "
    "scissors teddy-bear hair-drier toothbrush"
).split()


def nms(boxes, scores, iou_thr=0.45):
    """Standard greedy non-maximum suppression. Returns indices to keep."""
    idxs = scores.argsort()[::-1]
    keep = []
    while len(idxs):
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[idxs[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[idxs[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[idxs[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[idxs[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_o = (boxes[idxs[1:], 2] - boxes[idxs[1:], 0]) * (
            boxes[idxs[1:], 3] - boxes[idxs[1:], 1]
        )
        iou = inter / (area_i + area_o - inter + 1e-9)
        idxs = idxs[1:][iou < iou_thr]
    return keep


class YoloDetector:
    """Loads the .tflite once, then detects on RGB numpy frames of any size.

    backend=None runs on CPU; backend="htp" (Hexagon NPU) or "gpu" routes the
    same model file through the QNN delegate instead. num_threads only applies
    to the CPU path (LiteRT defaults to a single thread unless asked).
    """

    def __init__(self, model_path, conf=0.4, backend=None, num_threads=None):
        if backend:
            delegate = load_delegate(QNN_DELEGATE, options={"backend_type": backend})
            self.interp = Interpreter(model_path=model_path,
                                      experimental_delegates=[delegate])
        elif num_threads:
            self.interp = Interpreter(model_path=model_path, num_threads=num_threads)
        else:
            self.interp = Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.outs = self.interp.get_output_details()
        # AI Hub exports are usually NHWC (1,H,W,3); a manually-driven compile
        # (e.g. a quantize job run through the raw API) can come back NCHW
        # (1,3,H,W) instead. Detect by finding which axis is the channel dim.
        shape = self.inp["shape"]
        self.nchw = shape[1] == 3
        if self.nchw:
            _, _, self.ih, self.iw = shape
        else:
            _, self.ih, self.iw, _ = shape
        self.conf = conf

    def _out(self, name_frag, fallback_idx):
        for o in self.outs:
            if name_frag in o["name"]:
                return self.interp.get_tensor(o["index"])
        return self.interp.get_tensor(self.outs[fallback_idx]["index"])

    def detect(self, rgb):
        """rgb: HxWx3 uint8. Returns list of (label, score, (x1,y1,x2,y2)) in frame coords."""
        h0, w0 = rgb.shape[:2]
        import cv2
        x = cv2.resize(rgb, (self.iw, self.ih)).astype(np.float32)[None] / 255.0
        if self.nchw:
            x = x.transpose(0, 3, 1, 2)
        x = x.astype(self.inp["dtype"])
        self.interp.set_tensor(self.inp["index"], x)
        self.interp.invoke()

        boxes = self._out("boxes", 0)[0].astype(np.float32)
        scores = self._out("scores", 1)[0].astype(np.float32)
        cls = self._out("class", 2)[0].astype(np.int32)

        m = scores >= self.conf
        boxes, scores, cls = boxes[m], scores[m], cls[m]
        keep = nms(boxes, scores)

        sx, sy = w0 / self.iw, h0 / self.ih
        results = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            label = COCO[cls[i]] if cls[i] < len(COCO) else str(cls[i])
            results.append(
                (label, float(scores[i]), (x1 * sx, y1 * sy, x2 * sx, y2 * sy))
            )
        return results


# fixed palette (BGR) so every class keeps a stable, readable color between frames
PALETTE = [
    (80, 175, 76), (0, 152, 255), (243, 150, 33), (54, 67, 244), (176, 39, 156),
    (212, 188, 0), (99, 30, 233), (57, 220, 205), (181, 81, 63), (7, 193, 255),
]


def draw_detections(frame, results, hud_lines):
    """Draw labeled boxes and a HUD pill onto a BGR frame, in place.

    results: list of (label, score, (x1, y1, x2, y2)) as returned by
    YoloDetector.detect(). hud_lines: two strings, a headline (FPS or latency)
    and a smaller detail line (the model file name).
    """
    import cv2
    for label, score, (x1, y1, x2, y2) in results:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        color = PALETTE[zlib.crc32(label.encode()) % len(PALETTE)]
        cv2.rectangle(frame, p1, p2, color, 3)
        text = f"{label} {score:.2f}"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = p1[1] - th - base - 6
        if ty < 0:  # box touches the top edge: put the label inside instead
            ty = max(p1[1] + 4, 4)
        cv2.rectangle(frame, (p1[0] - 1, ty), (p1[0] + tw + 8, ty + th + base + 6),
                      color, -1)
        cv2.putText(frame, text, (p1[0] + 4, ty + th + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    # HUD: one dark pill, headline in green, model name below in gray
    (w0, h0), b0 = cv2.getTextSize(hud_lines[0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    (w1, h1), b1 = cv2.getTextSize(hud_lines[1], cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (8, 8), (32 + max(w0, w1), 8 + h0 + b0 + h1 + b1 + 30),
                  (30, 30, 30), -1)
    cv2.putText(frame, hud_lines[0], (20, 20 + h0), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (120, 255, 120), 2)
    cv2.putText(frame, hud_lines[1], (20, 20 + h0 + b0 + 8 + h1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
