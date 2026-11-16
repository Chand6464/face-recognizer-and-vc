"""
NumPy port of face-api.js's age and gender network (TinyXception).
No TensorFlow, no PyTorch, no JavaScript: only numpy, opencv and safetensors.

Used by voice_bot_gui.py, but you can also test it alone:
    python faceapi_agegender.py --cam        live webcam test (press q to quit)
    python faceapi_agegender.py photo.jpg    one photo

Weights: models/age_gender_model.safetensors  (next to this file)

Optional better face detector: put face_detection_yunet_2023mar.onnx (OpenCV zoo)
into the same models/ folder and it is used automatically instead of the Haar detector.
"""
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(HERE, "models", "age_gender_model.safetensors")
WEIGHTS_URL = "https://huggingface.co/CognitiveOS/vision/resolve/main/age_gender_model.safetensors"
YUNET_FILE = "face_detection_yunet_2023mar.onnx"

INPUT_SIZE = 112
MEAN_RGB = np.array([122.782, 117.001, 104.298], dtype=np.float32)   # same numbers as face-api.js
# face-api.js reads output index 0 of the gender softmax as the probability of MALE.
# If your results come out with men and women swapped, change this to 1.
MALE_INDEX = 1


# ----------------------------------------------------------------------------
# weights
# ----------------------------------------------------------------------------
def download_weights(path=DEFAULT_WEIGHTS):
    """Download the 1.7 MB weight file if it is missing."""
    if os.path.isfile(path):
        return path
    import requests
    os.makedirs(os.path.dirname(path), exist_ok=True)
    r = requests.get(WEIGHTS_URL, timeout=60)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return path


def load_params(path):
    """Read the safetensors file into nested dicts: params['entry_flow']['conv_in']['filters']"""
    from safetensors import safe_open
    params = {}
    with safe_open(path, framework="np") as f:
        for key in f.keys():
            node = params
            parts = key.split("/")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = np.ascontiguousarray(f.get_tensor(key), dtype=np.float32)
    return params


# ----------------------------------------------------------------------------
# layers (all arrays are height x width x channels, like TensorFlow.js)
# ----------------------------------------------------------------------------
def relu(x):
    return np.maximum(x, 0)


def conv_in(x, filters, bias):
    """3x3 convolution, stride 2, 'same' padding (TensorFlow puts the extra pad bottom/right)."""
    h, w, _ = x.shape
    oh, ow = h // 2, w // 2
    xp = np.pad(x, ((0, 1), (0, 1), (0, 0)))
    cols = [xp[i:i + 2 * oh:2, j:j + 2 * ow:2, :] for i in range(3) for j in range(3)]
    patches = np.concatenate(cols, axis=2)                          # (oh, ow, 27)
    kh, kw, cin, cout = filters.shape
    out = patches.reshape(-1, kh * kw * cin) @ filters.reshape(kh * kw * cin, cout)
    return out.reshape(oh, ow, cout) + bias


def conv1x1_stride2(x, filters, bias):
    xs = x[::2, ::2, :]
    h, w, c = xs.shape
    out = xs.reshape(-1, c) @ filters[0, 0]
    return out.reshape(h, w, -1) + bias


def separable_conv(x, p):
    """Depthwise 3x3 (stride 1, same) then pointwise 1x1, then bias."""
    h, w, c = x.shape
    k = p["depthwise_filter"][:, :, :, 0]                           # (3, 3, C)
    xp = np.pad(x, ((1, 1), (1, 1), (0, 0)))
    y = xp[0:h, 0:w] * k[0, 0]
    for i in range(3):
        for j in range(3):
            if i or j:
                y += xp[i:i + h, j:j + w] * k[i, j]
    pw = p["pointwise_filter"][0, 0]                                # (Cin, Cout)
    y = (y.reshape(-1, c) @ pw).reshape(h, w, -1)
    return y + p["bias"]


def max_pool_3x3_stride2(x):
    h, w, _ = x.shape
    oh, ow = (h + 1) // 2, (w + 1) // 2
    xp = np.pad(x, ((0, 1), (0, 1), (0, 0)), constant_values=-np.inf)
    out = None
    for i in range(3):
        for j in range(3):
            part = xp[i:i + 2 * oh:2, j:j + 2 * ow:2, :]
            out = part if out is None else np.maximum(out, part)
    return out


def reduction_block(x, p, activate_input=True):
    out = relu(x) if activate_input else x
    out = separable_conv(out, p["separable_conv0"])
    out = separable_conv(relu(out), p["separable_conv1"])
    out = max_pool_3x3_stride2(out)
    return out + conv1x1_stride2(x, p["expansion_conv"]["filters"], p["expansion_conv"]["bias"])


def main_block(x, p):
    out = separable_conv(relu(x), p["separable_conv0"])
    out = separable_conv(relu(out), p["separable_conv1"])
    out = separable_conv(relu(out), p["separable_conv2"])
    return out + x


class AgeGenderNet:
    def __init__(self, weights_path):
        self.p = load_params(weights_path)
        self.n_main = sum(1 for k in self.p["middle_flow"] if k.startswith("main_block_"))

    def forward(self, face_rgb):
        """face_rgb: 112x112x3 float32, values 0..255, RGB. Returns (age_years, male_probability)."""
        p = self.p
        x = (face_rgb - MEAN_RGB) / 256.0
        x = relu(conv_in(x, p["entry_flow"]["conv_in"]["filters"], p["entry_flow"]["conv_in"]["bias"]))
        x = reduction_block(x, p["entry_flow"]["reduction_block_0"], activate_input=False)
        x = reduction_block(x, p["entry_flow"]["reduction_block_1"])
        for i in range(self.n_main):
            x = main_block(x, p["middle_flow"][f"main_block_{i}"])
        x = reduction_block(x, p["exit_flow"]["reduction_block"])
        x = relu(separable_conv(x, p["exit_flow"]["separable_conv"]))
        pooled = x.mean(axis=(0, 1))                                # 7x7 average pool -> 512 numbers
        age = float(pooled @ p["fc"]["age"]["weights"][:, 0] + p["fc"]["age"]["bias"][0])
        logits = pooled @ p["fc"]["gender"]["weights"] + p["fc"]["gender"]["bias"]
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        return age, float(probs[MALE_INDEX])

    def predict_crop(self, crop_bgr):
        """Prepare a face crop the way face-api.js does: square (centered, black padding), 112x112."""
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        m = max(h, w)
        square = np.zeros((m, m, 3), dtype=np.uint8)
        y0, x0 = (m - h) // 2, (m - w) // 2
        square[y0:y0 + h, x0:x0 + w] = rgb
        interp = cv2.INTER_AREA if m > INPUT_SIZE else cv2.INTER_LINEAR
        small = cv2.resize(square, (INPUT_SIZE, INPUT_SIZE), interpolation=interp)
        return self.forward(small.astype(np.float32))


# ----------------------------------------------------------------------------
# face finder + analyzer used by the GUI
# ----------------------------------------------------------------------------
class FaceApiAnalyzer:
    """analyze(frame_bgr) returns a list shaped like DeepFace.analyze's output:
       [{'region': {x,y,w,h}, 'age': int, 'gender': {'Woman': %, 'Man': %}, 'dominant_emotion': str|None}]
       Raises ValueError('Face could not be detected.') when there is no face."""

    def __init__(self, weights_path=DEFAULT_WEIGHTS, box_padding=0.1, emotion_fn=None):
        self.net = AgeGenderNet(weights_path)
        self.pad = float(box_padding)
        self.emotion_fn = emotion_fn
        self.detector_name = "Haar"
        self._yunet = None
        yunet_path = os.path.join(os.path.dirname(weights_path), YUNET_FILE)
        if os.path.isfile(yunet_path) and hasattr(cv2, "FaceDetectorYN"):
            try:
                self._yunet = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320), 0.8, 0.3, 5000)
                self.detector_name = "YuNet"
            except Exception as e:
                print("YuNet could not be loaded, using Haar instead:", e)
        self._haar = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
        if self._haar.empty():
            raise RuntimeError("OpenCV's Haar face file could not be loaded.")
        print(f"Face detector: {self.detector_name}")

    def _detect(self, frame):
        if self._yunet is not None:
            h, w = frame.shape[:2]
            self._yunet.setInputSize((w, h))
            _, faces = self._yunet.detect(frame)
            if faces is None:
                return []
            return [tuple(int(v) for v in f[:4]) for f in faces]
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        rects = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        return [tuple(int(v) for v in r) for r in rects]

    def analyze(self, frame_bgr):
        H, W = frame_bgr.shape[:2]
        results = []
        for (x, y, w, h) in self._detect(frame_bgr):
            x, y = max(x, 0), max(y, 0)
            w, h = min(w, W - x), min(h, H - y)
            if w < 20 or h < 20:
                continue
            px, py = int(w * self.pad), int(h * self.pad)
            crop = frame_bgr[max(y - py, 0):min(y + h + py, H), max(x - px, 0):min(x + w + px, W)]
            if crop.size == 0:
                continue
            age, male = self.net.predict_crop(crop)
            emotion = None
            if self.emotion_fn is not None:
                try:
                    emotion = self.emotion_fn(crop)
                except Exception:
                    emotion = None
            results.append({
                "region": {"x": x, "y": y, "w": w, "h": h},
                "age": int(round(max(age, 1.0))),
                "gender": {"Woman": (1.0 - male) * 100.0, "Man": male * 100.0},
                "dominant_emotion": emotion,
            })
        if not results:
            raise ValueError("Face could not be detected.")
        return results


# ----------------------------------------------------------------------------
# stand-alone test
# ----------------------------------------------------------------------------
def _label(face):
    g = face["gender"]
    who = "Woman" if g["Woman"] > g["Man"] else "Man"
    return f"{who} {max(g.values()):.0f}%, age {face['age']}"


def _test_photo(path):
    img = cv2.imread(path)
    if img is None:
        print("Could not read the image:", path)
        return
    an = FaceApiAnalyzer(download_weights())
    try:
        for f in an.analyze(img):
            print(_label(f), f["region"])
    except ValueError as e:
        print(e)


def _open_camera():
    """Try camera 0 and 1 with different Windows backends; return one that really gives a picture."""
    for index in (0, 1):
        for name, backend in (("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("default", cv2.CAP_ANY)):
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                for _ in range(30):                      # cameras often need a moment to start
                    ok, _frame = cap.read()
                    if ok:
                        print(f"Camera {index} works with {name}.")
                        return cap
                    time.sleep(0.1)
            cap.release()
    print("No camera gave a picture. Check: another app (or an old python window) is using it, "
          "Windows Settings > Privacy > Camera allows desktop apps, or the lens cover is closed.")
    return None


def _test_camera():
    from collections import deque
    an = FaceApiAnalyzer(download_weights())
    cap = _open_camera()
    if cap is None:
        return
    ages, men = deque(maxlen=10), deque(maxlen=10)
    print("Live test running. Press q in the video window to quit.")
    failed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            failed += 1
            if failed > 50:
                print("The camera stopped giving pictures. Close other apps that use the camera and try again.")
                break
            time.sleep(0.05)
            continue
        failed = 0
        t0 = time.time()
        text = "No face"
        try:
            faces = an.analyze(frame)
            f = max(faces, key=lambda r: r["region"]["w"] * r["region"]["h"])
            ages.append(f["age"])
            men.append(f["gender"]["Man"])
            avg_man = sum(men) / len(men)
            who = "Man" if avg_man > 50 else "Woman"
            text = f"{who} {max(avg_man, 100 - avg_man):.0f}%, age {sum(ages) / len(ages):.0f}"
            r = f["region"]
            cv2.rectangle(frame, (r["x"], r["y"]), (r["x"] + r["w"], r["y"] + r["h"]), (0, 255, 0), 2)
        except ValueError:
            pass
        ms = (time.time() - t0) * 1000
        cv2.putText(frame, f"{text}   ({ms:.0f} ms)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("faceapi test (q = quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cam":
        _test_camera()
    elif len(sys.argv) > 1:
        _test_photo(sys.argv[1])
    else:
        print(__doc__)