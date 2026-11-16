"""
Face-aware voice chatbot with a Tkinter window (free services only).

Flow:
  1. Live camera  -> face-api.js's age/gender model (ported to Python, see faceapi_agegender.py)
                     guesses age and gender; runs locally, nothing is saved.
                     Like predict-age.com, the shown numbers are an AVERAGE of the most
                     recent readings, so they stay steady instead of jumping around.
  2. Results card -> updates live in the window, and the bot says its first guess out loud
  3. Chat         -> talk into the mic (or type) -> Google free speech recognition
                     -> Groq free LLM -> answer shown in the chat box and spoken aloud

Languages: English and Hindi. Pick the language you will SPEAK in the drop-down at the top.
The bot answers in the language you used (Hindi in Devanagari script is spoken with a
Hindi Windows voice, which must be installed - see the note above pick_voice).

Needs internet for speech recognition and the LLM. The face part works offline.
Say "goodbye" (or "alvida" in Hindi) to end a conversation, then click "New person".
"""
import base64
import os
import queue
import re
import statistics
import threading
import time
import tkinter as tk
from collections import deque
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import requests
import sounddevice as sd
import speech_recognition as sr
import pyttsx3

# ----------------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------------
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# If this model name stops working, pick a current one from https://console.groq.com/docs/models
# and either edit it here or set the GROQ_MODEL environment variable.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

# Which model guesses age and gender:
#   "faceapi"  = the model behind predict-age.com (tiny, fast, no TensorFlow needed)
#   "deepface" = the DeepFace models used before (heavy)
FACE_BACKEND = "faceapi"
EMOTION_WITH_DEEPFACE = False   # faceapi mode only: let DeepFace's small emotion model add the mood.
                               # Set False to drop TensorFlow completely (mood then shows "-").
BOX_PADDING = 0.1              # faceapi mode: extra margin around the detected face (0 to 0.3)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FACEAPI_WEIGHTS = os.path.join(SCRIPT_DIR, "models", "age_gender_model.safetensors")

# Languages you can choose in the drop-down (name shown -> Google speech recognition code)
LANGUAGES = {"English": "en-IN", "हिन्दी (Hindi)": "hi-IN"}
LANGUAGE = "en-IN"          # starting language; changed live from the drop-down in the window
SPEECH_RATE = 170           # bot speaking speed

SCAN_SAMPLES = 5            # readings needed before the bot announces its guess
SCAN_TIMEOUT = 20           # seconds to wait for a face before asking the person to look again
WINDOW = 10                 # the shown age/gender/mood is the average of the last N readings
ANALYZE_EVERY = 0.5         # seconds between face readings while scanning (the CPU may be slower)
LIVE_DURING_CHAT = True     # keep the camera and the live numbers running during the chat
CHAT_ANALYZE_EVERY = 3.0    # slower readings during the chat to leave CPU for speech (old laptop)
PREVIEW_FPS = 15            # camera picture refresh rate in the window
TF_THREADS = 2              # limit TensorFlow CPU threads so the window stays responsive
ANALYZE_WIDTH = 640         # bigger = more accurate but slower
DETECTOR = "opencv"         # deepface mode only: try "yunet", "mtcnn" or "retinaface"

SAMPLE_RATE = 16000
MIC_MIN_THRESHOLD = 250     # raise if the mic hears noise as speech, lower if it misses you
SILENCE_END = 1.2           # seconds of silence that end your sentence
MAX_WAIT = 8                # seconds to wait for you to start talking
MAX_UTTERANCE = 15          # longest single sentence in seconds

VIEW_W, VIEW_H = 400, 300   # size of the camera picture in the window

EMOTION_WORDS = {
    "happy": "happy", "sad": "a little sad", "angry": "a bit angry",
    "fear": "a bit nervous", "surprise": "surprised",
    "neutral": "calm", "disgust": "a bit unimpressed",
}
MOOD_LABELS = {
    "happy": "Happy", "sad": "Sad", "angry": "Angry", "fear": "Nervous",
    "surprise": "Surprised", "neutral": "Calm", "disgust": "Unimpressed",
}

recognizer = sr.Recognizer()

DeepFace = None
if FACE_BACKEND == "deepface" or EMOTION_WITH_DEEPFACE:
    from deepface import DeepFace
    try:
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(TF_THREADS)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass


def deepface_analyze(frame):
    """All three DeepFace models in one go (used when FACE_BACKEND = "deepface")."""
    return DeepFace.analyze(frame, actions=["age", "gender", "emotion"], detector_backend=DETECTOR,
                            enforce_detection=True, silent=True)


def deepface_emotion(face_bgr):
    """Just the small emotion model, on an already-cropped face."""
    try:
        r = DeepFace.analyze(face_bgr, actions=["emotion"], detector_backend="skip",
                             enforce_detection=False, silent=True)
        return r[0]["dominant_emotion"]
    except Exception:
        return None


def make_analyzer(app):
    """Returns a function(frame_bgr) -> list of faces shaped like DeepFace.analyze's output."""
    if FACE_BACKEND == "deepface":
        app.status("Loading face models (the first time takes a while)...")
        DeepFace.analyze(np.zeros((224, 224, 3), dtype=np.uint8), actions=["age", "gender", "emotion"],
                         detector_backend=DETECTOR, enforce_detection=False, silent=True)
        return deepface_analyze

    import faceapi_agegender as fa
    app.status("Loading the age and gender model...")
    fa.download_weights(FACEAPI_WEIGHTS)                    # 1.7 MB, only if missing
    analyzer = fa.FaceApiAnalyzer(FACEAPI_WEIGHTS, box_padding=BOX_PADDING,
                                  emotion_fn=deepface_emotion if EMOTION_WITH_DEEPFACE else None)
    if EMOTION_WITH_DEEPFACE:
        app.status("Loading the mood model...")
        deepface_emotion(np.zeros((224, 224, 3), dtype=np.uint8))   # warm-up
    return analyzer.analyze


GOODBYE = re.compile(r"\b(bye|goodbye|quit|exit)\b", re.IGNORECASE)
GOODBYE_HINDI = ("अलविदा", "बाय", "गुडबाय", "alvida")


def is_goodbye(text):
    low = text.lower()
    return bool(GOODBYE.search(text)) or any(w in low for w in GOODBYE_HINDI)


# ----------------------------------------------------------------------------
# TEXT TO SPEECH
# ----------------------------------------------------------------------------
def init_com():
    """pyttsx3 uses Windows COM, and every thread that speaks must initialise it."""
    try:
        import pythoncom
        pythoncom.CoInitialize()
    except Exception:
        pass
    try:
        import comtypes
        comtypes.CoInitialize()
    except Exception:
        pass


def clean_for_speech(text):
    """Remove markdown and emojis, but keep every language's letters (including Hindi vowel signs)."""
    text = re.sub(r"[*_`#>~|]", "", text)
    text = re.sub(r"[\U00010000-\U0010ffff\u2600-\u27bf]", "", text)   # emojis
    return re.sub(r"\s+", " ", text).strip()


def pick_voice(engine, hindi):
    """Choose a Windows voice by language. Hindi needs a Hindi voice installed:
    Windows Settings > Time & language > Speech > add a voice for Hindi (India).
    To list your voices:  python -c "import pyttsx3;[print(v.name) for v in pyttsx3.init().getProperty('voices')]"
    """
    word = "hindi" if hindi else "english"
    for v in engine.getProperty("voices"):
        if word in v.name.lower():
            engine.setProperty("voice", v.id)
            return True
    return False


def speak(app, text):
    text = clean_for_speech(text)
    app.chat("bot", text)
    app.status("Speaking...")
    hindi = bool(re.search(r"[\u0900-\u097F]", text))        # Devanagari letters -> use a Hindi voice
    try:
        engine = pyttsx3.init()          # fresh engine each time avoids a known hang
        engine.setProperty("rate", SPEECH_RATE)
        if not pick_voice(engine, hindi) and hindi:
            app.chat("sys", "(No Hindi voice is installed in Windows, so I can only show the Hindi text. "
                            "Add one in Settings > Time & language > Speech.)")
            return
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        app.chat("sys", f"(speech error: {e})")


# ----------------------------------------------------------------------------
# SPEECH TO TEXT
# ----------------------------------------------------------------------------
def rms(chunk):
    return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))


def record_utterance(app):
    """Record from the mic until the person stops talking. Returns audio bytes or None."""
    block = int(SAMPLE_RATE * 0.1)
    preroll = deque(maxlen=3)
    frames = []
    started = False
    silent = 0.0
    waited = 0.0

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=block) as stream:
        noise = [rms(stream.read(block)[0]) for _ in range(5)]   # measure background noise
        threshold = max(MIC_MIN_THRESHOLD, float(np.mean(noise)) * 2.5)

        while True:
            if app.should_stop() or not app.typed.empty():
                return None                      # new person, quitting, or the user typed something
            data, _ = stream.read(block)
            level = rms(data)
            preroll.append(data)

            if level > threshold:
                if not started:
                    started = True
                    app.status("Hearing you...")
                    frames = list(preroll)
                else:
                    frames.append(data)
                silent = 0.0
            elif started:
                frames.append(data)
                silent += 0.1
            else:
                waited += 0.1
                if waited >= MAX_WAIT:
                    return None

            if started and silent >= SILENCE_END:
                break
            if started and len(frames) * 0.1 >= MAX_UTTERANCE:
                break

    if len(frames) < 5:      # too short, probably a click or cough
        return None
    return np.concatenate(frames).tobytes()


def listen(app):
    """Returns the recognized text, '' if speech was not understood, or None if nothing to process."""
    try:
        raw = record_utterance(app)
    except Exception as e:
        app.chat("sys", f"Microphone problem: {e}\nThe microphone was turned off. You can type instead.")
        app.mic_on = False
        app.post("mic_off", None)
        return None
    if raw is None:
        return None
    app.status("Understanding...")
    audio = sr.AudioData(raw, SAMPLE_RATE, 2)
    try:
        return recognizer.recognize_google(audio, language=app.language)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        app.chat("sys", f"Speech recognition needs internet: {e}")
        time.sleep(1)
        return None


# ----------------------------------------------------------------------------
# LIVE FACE TRACKER (camera preview + rolling average of readings)
# ----------------------------------------------------------------------------
class FaceTracker:
    """Owns the camera. One thread shows the live preview with a box and label, another
    runs the face model again and again. The numbers shown are averages of the last WINDOW
    readings, like predict-age.com."""

    def __init__(self, app):
        self.app = app
        self.cap = None
        self.interval = ANALYZE_EVERY
        self.total = 0                     # readings taken so far in this session
        self.last_view = None              # last annotated frame, used as a snapshot
        self.ages = deque(maxlen=WINDOW)
        self.genders = deque(maxlen=WINDOW)   # (woman %, man %) pairs
        self.emotions = deque(maxlen=WINDOW)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._latest = None
        self._box = None
        self._threads = []

    def start(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = None
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        for target in (self._capture_loop, self._analysis_loop):
            t = threading.Thread(target=target, daemon=True)
            t.start()
            self._threads.append(t)
        return True

    def stop(self):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=5)
        self._threads = []
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def stats(self):
        """Smoothed result from the recent readings, or None if there are too few."""
        with self._lock:
            n = len(self.ages)
            if n < 2:
                return None
            ages = sorted(self.ages)
            trimmed = ages[1:-1] if n >= 5 else ages        # ignore the highest and lowest reading
            woman = statistics.mean(w for w, _ in self.genders)
            man = statistics.mean(m for _, m in self.genders)
            emotions = list(self.emotions)
            return {
                "age": int(round(statistics.mean(trimmed))),
                "gender": "Woman" if woman > man else "Man",
                "emotion": max(set(emotions), key=emotions.count),   # None if no mood model
                "woman": woman,
                "man": man,
            }

    def wait_stable(self, timeout):
        """Wait until enough readings exist. Returns the smoothed stats, or None on timeout."""
        end = time.time() + timeout
        while time.time() < end and not self.app.should_stop():
            if self.total >= SCAN_SAMPLES:
                return self.stats()
            time.sleep(0.1)
        return None

    # ---- background threads ----
    def _capture_loop(self):
        frame_time = 1.0 / PREVIEW_FPS
        while not self._stop.is_set():
            t0 = time.time()
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self._lock:
                self._latest = frame
                box = None if self._box is None else self._box.astype(int)

            st = self.stats()
            if box is None:
                label = "Looking for a face..."
            elif st is None or self.total < SCAN_SAMPLES:
                label = f"Scanning... {min(self.total, SCAN_SAMPLES)}/{SCAN_SAMPLES}"
            else:
                label = f"{st['gender']}, {st['age']}"

            view = frame.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            if box is not None:
                x, y, w, h = (int(v) for v in box)
                cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)
                (tw, th), _ = cv2.getTextSize(label, font, 0.7, 2)
                top = max(y - th - 12, 0)
                cv2.rectangle(view, (x, top), (x + tw + 10, top + th + 10), (0, 255, 0), -1)
                cv2.putText(view, label, (x + 5, top + th + 3), font, 0.7, (0, 0, 0), 2)
            else:
                cv2.putText(view, label, (10, 30), font, 0.8, (0, 255, 0), 2)

            self.last_view = view
            self.app.show_frame(view)
            wait = frame_time - (time.time() - t0)
            if wait > 0:
                self._stop.wait(wait)

    def _analysis_loop(self):
        misses = 0
        while not self._stop.is_set():
            with self._lock:
                frame = self._latest
            if frame is None:
                time.sleep(0.05)
                continue

            t0 = time.time()
            try:
                h, w = frame.shape[:2]
                small = cv2.resize(frame, (ANALYZE_WIDTH, int(h * ANALYZE_WIDTH / w)))
                faces = self.app.analyze(small)
                if not faces:
                    raise ValueError("could not be detected")
                face = max(faces, key=lambda f: f["region"]["w"] * f["region"]["h"])
                r, k = face["region"], w / ANALYZE_WIDTH
                new_box = np.array([r["x"] * k, r["y"] * k, r["w"] * k, r["h"] * k], dtype=float)
                with self._lock:
                    # blend with the old box so it glides instead of jittering
                    self._box = new_box if self._box is None else 0.5 * self._box + 0.5 * new_box
                    self.ages.append(face["age"])
                    self.genders.append((face["gender"].get("Woman", 0.0), face["gender"].get("Man", 0.0)))
                    self.emotions.append(face["dominant_emotion"])
                    self.total += 1
                misses = 0
                print(f"  reading {self.total}: age {face['age']}, "
                      f"Woman {face['gender'].get('Woman', 0):.0f}% / Man {face['gender'].get('Man', 0):.0f}%")
                st = self.stats()
                if st and not self._stop.is_set():
                    self.app.post("results", st)
            except ValueError as e:
                if "could not be detected" in str(e):
                    misses += 1
                    if misses >= 3:                  # face really gone, drop the box
                        with self._lock:
                            self._box = None
                else:
                    print("Face analysis error:", e)
            except Exception as e:
                print("Face analysis error:", e)
                time.sleep(0.5)

            delay = self.interval - (time.time() - t0)
            if delay > 0:
                self._stop.wait(delay)


# ----------------------------------------------------------------------------
# LLM (Groq free tier)
# ----------------------------------------------------------------------------
def load_api_key():
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "groq_key.txt")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                key = f.read().strip()
    return key


def ask_llm(api_key, messages):
    payload = {"model": GROQ_MODEL, "messages": messages, "max_tokens": 800, "temperature": 0.7}
    if GROQ_MODEL.startswith("openai/gpt-oss"):
        payload["reasoning_effort"] = "low"      # faster, shorter thinking for voice
    try:
        r = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
    except requests.RequestException as e:
        print("Network error:", e)
        return "Sorry, I can't reach the internet right now."

    if r.status_code == 429:
        return "I'm getting too many questions right now. Please ask me again in a moment."
    if r.status_code != 200:
        print("Groq error:", r.status_code, r.text[:300])
        return "Sorry, my brain had a problem. Please try again."

    content = (r.json()["choices"][0]["message"].get("content") or "").strip()
    return content or "Sorry, I didn't come up with an answer. Could you ask again?"


def build_system_prompt(info):
    prompt = (
        "You are a friendly voice assistant at a school project demo. Your replies are read "
        "aloud, so answer in one to three short spoken sentences. No markdown, no lists, no "
        "emojis. Keep everything appropriate for a school. If you are unsure, say so. "
        "Always answer in the language the person last spoke: English, or Hindi written in "
        "Devanagari script (never Hindi in Roman letters)."
    )
    if info:
        who = "man" if info["gender"] == "Man" else "woman"
        mood = EMOTION_WORDS.get(info["emotion"], info["emotion"]) if info["emotion"] else None
        prompt += (
            f" A camera guessed the person is a {who}, about {info['age']} years old"
            + (f", and looks {mood}" if mood else "")
            + ". These are rough guesses and may be wrong. Use them only to adjust your tone "
            "and word choice. Do not mention them unless asked, and accept it if the person "
            "corrects you."
        )
    return prompt


# ----------------------------------------------------------------------------
# ONE CONVERSATION (runs in the worker thread)
# ----------------------------------------------------------------------------
def run_session(app, api_key):
    app.post("reset", None)
    tracker = FaceTracker(app)
    camera_ok = tracker.start()
    try:
        if not camera_ok:
            app.chat("sys", "Could not open the camera. Continuing without face info.")
        speak(app, "Hello! I'm going to use the camera to guess your age and gender. "
                   "Nothing is saved. Please look at the camera.")
        if app.should_stop():
            return

        info = None
        if camera_ok:
            for _ in range(2):
                app.status("Scanning your face. Please look at the camera...")
                info = tracker.wait_stable(SCAN_TIMEOUT)
                if info or app.should_stop():
                    break
                speak(app, "I couldn't see a face. Please look straight at the camera.")
        if app.should_stop():
            return

        if info:
            who = "man" if info["gender"] == "Man" else "woman"
            mood = EMOTION_WORDS.get(info["emotion"], info["emotion"]) if info["emotion"] else None
            intro = (f"My guess is that you're a {who}, around {info['age']} years old"
                     + (f", and you look {mood}" if mood else "")
                     + ". Tell me if I'm wrong! What would you like to talk about?")
        else:
            intro = "I couldn't get a good look at you, but that's okay. What would you like to talk about?"

        if camera_ok and info:
            if LIVE_DURING_CHAT:
                tracker.interval = CHAT_ANALYZE_EVERY      # keep updating, but go easy on the CPU
            else:
                snap = tracker.last_view
                tracker.stop()
                if snap is not None:
                    app.show_frame(snap)                   # freeze the picture
        speak(app, intro)

        history = [{"role": "assistant", "content": intro}]
        system = {"role": "system", "content": build_system_prompt(info)}

        while not app.should_stop():
            try:
                text = app.typed.get_nowait()
            except queue.Empty:
                text = None

            if text is None:
                if not app.mic_on:
                    app.status("Microphone is off. Type your message below.")
                    time.sleep(0.2)
                    continue
                app.status("Listening... speak now")
                text = listen(app)
                if text is None:             # silence, aborted, or a problem: just look again
                    continue
                if text == "":
                    speak(app, "Sorry, I didn't catch that.")
                    continue

            app.chat("you", text)
            if is_goodbye(text):
                speak(app, "It was nice talking to you. Goodbye!")
                return

            history.append({"role": "user", "content": text})
            app.status("Thinking...")
            reply = ask_llm(api_key, [system] + history[-12:])
            history.append({"role": "assistant", "content": reply})
            speak(app, reply)
    finally:
        tracker.stop()


def worker_main(app):
    init_com()
    api_key = load_api_key()
    if not api_key:
        app.chat("sys",
                 "No Groq API key found.\n"
                 "1. Get a free key at https://console.groq.com\n"
                 "2. In PowerShell run:  $env:GROQ_API_KEY=\"your_key_here\"\n"
                 "   or save the key in a file named groq_key.txt next to this script.\n"
                 "Then restart the program.")
        app.status("Missing API key")
        return

    app.chat("sys", "Privacy note: your voice is sent to Google for speech recognition and your "
                    "words to Groq for the answer. Camera images stay on this computer and are not saved.")
    try:
        app.analyze = make_analyzer(app)
    except Exception as e:
        app.chat("sys",
                 f"Could not load the face model: {e}\n"
                 "If it is the age/gender file, download age_gender_model.safetensors from\n"
                 "https://huggingface.co/CognitiveOS/vision/resolve/main/age_gender_model.safetensors\n"
                 f"and put it here: {FACEAPI_WEIGHTS}\n"
                 "Then restart the program.")
        app.status("Face model missing")
        return

    while not app.quitting:
        app.restart.clear()
        try:
            run_session(app, api_key)
        except Exception as e:
            app.chat("sys", f"Something went wrong: {e}")
        if app.quitting:
            break
        if not app.restart.is_set():
            app.status("Session ended. Click 'New person' to start again.")
            app.restart.wait()


# ----------------------------------------------------------------------------
# THE WINDOW
# ----------------------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        self.q = queue.Queue()           # worker -> window messages
        self.typed = queue.Queue()       # window -> worker typed messages
        self.restart = threading.Event()
        self.quitting = False
        self.mic_on = True
        self.language = LANGUAGE         # speech recognition language, changed by the drop-down
        self._photo = None
        self.analyze = None              # set by the worker once the face model is loaded

        root.title("Face + Voice Chatbot")
        root.geometry("1020x640")
        root.minsize(900, 560)
        self._build()
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        threading.Thread(target=worker_main, args=(self,), daemon=True).start()
        self.poll()

    # ---- API used by the worker thread (thread-safe: only touches the queue) ----
    def post(self, kind, data):
        self.q.put((kind, data))

    def chat(self, role, text):
        self.post("chat", (role, text))

    def status(self, text):
        self.post("status", text)

    def should_stop(self):
        return self.quitting or self.restart.is_set()

    def show_frame(self, frame):
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (VIEW_W, int(h * VIEW_W / w)))
        ok, png = cv2.imencode(".png", small, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if ok:
            self.post("frame", base64.b64encode(png.tobytes()).decode("ascii"))

    # ---- building the layout ----
    def _build(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        bg = "#f4f6fb"
        self.root.configure(bg=bg)
        style.configure("TFrame", background=bg)
        style.configure("Card.TFrame", background="white", relief="flat")
        style.configure("TLabel", background=bg, font=("Segoe UI", 10))
        style.configure("Card.TLabel", background="white", font=("Segoe UI", 10))
        style.configure("Big.TLabel", background="white", font=("Segoe UI", 22, "bold"), foreground="#1f2937")
        style.configure("Small.TLabel", background="white", font=("Segoe UI", 9), foreground="#6b7280")
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), foreground="#1f2937")
        style.configure("Status.TLabel", font=("Segoe UI", 10, "italic"), foreground="#4b5563")
        style.configure("TButton", font=("Segoe UI", 10), padding=6)
        style.configure("TCheckbutton", background=bg, font=("Segoe UI", 10))

        # top bar
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=14, pady=(12, 6))
        ttk.Label(top, text="Face + Voice Chatbot", style="Title.TLabel").pack(side="left")
        ttk.Button(top, text="New person", command=self.new_person).pack(side="right")
        self.mic_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Use microphone", variable=self.mic_var,
                        command=self._mic_toggled).pack(side="right", padx=14)
        self.lang_var = tk.StringVar(value=next(n for n, c in LANGUAGES.items() if c == LANGUAGE))
        lang_box = ttk.Combobox(top, textvariable=self.lang_var, values=list(LANGUAGES),
                                state="readonly", width=16, font=("Segoe UI", 10))
        lang_box.pack(side="right", padx=(0, 4))
        lang_box.bind("<<ComboboxSelected>>", lambda e: self._lang_changed())
        ttk.Label(top, text="I will speak:").pack(side="right", padx=(0, 4))

        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=14, pady=6)

        # left column: camera + results
        left = ttk.Frame(body)
        left.pack(side="left", fill="y")

        cam = tk.Frame(left, width=VIEW_W, height=VIEW_H, bg="#111827")
        cam.pack()
        cam.pack_propagate(False)
        self.video = tk.Label(cam, text="Camera picture appears here", fg="#9ca3af",
                              bg="#111827", font=("Segoe UI", 11))
        self.video.pack(expand=True, fill="both")

        card = ttk.Frame(left, style="Card.TFrame", padding=14)
        card.pack(fill="x", pady=(10, 0))
        ttk.Label(card, text="What I see", style="Card.TLabel",
                  font=("Segoe UI", 11, "bold")).grid(row=0, column=0, columnspan=3, sticky="w")
        self.age_var = tk.StringVar(value="-")
        self.gender_var = tk.StringVar(value="-")
        self.mood_var = tk.StringVar(value="-")
        for col, (name, var) in enumerate((("Age", self.age_var), ("Gender", self.gender_var),
                                           ("Mood", self.mood_var))):
            ttk.Label(card, text=name, style="Small.TLabel").grid(row=1, column=col, sticky="w", padx=(0, 18), pady=(8, 0))
            ttk.Label(card, textvariable=var, style="Big.TLabel").grid(row=2, column=col, sticky="w", padx=(0, 18))
        self.scores_var = tk.StringVar(value="")
        ttk.Label(card, textvariable=self.scores_var, style="Small.TLabel").grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Label(card, text="These are rough guesses from a camera and can be wrong.",
                  style="Small.TLabel").grid(row=4, column=0, columnspan=3, sticky="w")

        # right column: chat
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(14, 0))

        self.chat_box = ScrolledText(right, wrap="word", font=("Segoe UI", 11), state="disabled",
                                     bg="white", relief="flat", padx=12, pady=10)
        self.chat_box.pack(fill="both", expand=True)
        self.chat_box.tag_configure("name_you", foreground="#1d4ed8", font=("Segoe UI", 10, "bold"))
        self.chat_box.tag_configure("msg_you", foreground="#111827", lmargin1=10, lmargin2=10)
        self.chat_box.tag_configure("name_bot", foreground="#047857", font=("Segoe UI", 10, "bold"))
        self.chat_box.tag_configure("msg_bot", foreground="#111827", lmargin1=10, lmargin2=10)
        self.chat_box.tag_configure("msg_sys", foreground="#6b7280", font=("Segoe UI", 10, "italic"))

        self.status_var = tk.StringVar(value="Starting...")
        ttk.Label(right, textvariable=self.status_var, style="Status.TLabel").pack(anchor="w", pady=(6, 2))

        entry_row = ttk.Frame(right)
        entry_row.pack(fill="x")
        self.entry = ttk.Entry(entry_row, font=("Segoe UI", 11))
        self.entry.pack(side="left", fill="x", expand=True, ipady=4)
        self.entry.bind("<Return>", lambda e: self.send_typed())
        ttk.Button(entry_row, text="Send", command=self.send_typed).pack(side="left", padx=(8, 0))
        self.entry.focus_set()

    # ---- button handlers ----
    def new_person(self):
        self.restart.set()

    def _mic_toggled(self):
        self.mic_on = self.mic_var.get()

    def _lang_changed(self):
        self.language = LANGUAGES[self.lang_var.get()]

    def send_typed(self):
        text = self.entry.get().strip()
        if text:
            self.entry.delete(0, "end")
            self.typed.put(text)

    def on_close(self):
        self.quitting = True
        self.restart.set()
        self.root.destroy()

    # ---- applying worker messages (main thread only) ----
    def poll(self):
        latest_frame = None
        try:
            for _ in range(300):
                kind, data = self.q.get_nowait()
                if kind == "frame":
                    latest_frame = data
                elif kind == "chat":
                    self._add_chat(*data)
                elif kind == "status":
                    self.status_var.set(data)
                elif kind == "results":
                    self._set_results(data)
                elif kind == "reset":
                    latest_frame = None
                    self._reset()
                elif kind == "mic_off":
                    self.mic_var.set(False)
        except queue.Empty:
            pass

        if latest_frame is not None:
            self._photo = tk.PhotoImage(data=latest_frame)
            self.video.configure(image=self._photo, text="")

        if not self.quitting:
            self.root.after(40, self.poll)

    def _add_chat(self, role, text):
        self.chat_box.configure(state="normal")
        if role == "you":
            self.chat_box.insert("end", "You\n", ("name_you",))
            self.chat_box.insert("end", text + "\n\n", ("msg_you",))
        elif role == "bot":
            self.chat_box.insert("end", "Bot\n", ("name_bot",))
            self.chat_box.insert("end", text + "\n\n", ("msg_bot",))
        else:
            self.chat_box.insert("end", text + "\n\n", ("msg_sys",))
        self.chat_box.configure(state="disabled")
        self.chat_box.see("end")

    def _set_results(self, info):
        self.age_var.set(f"about {info['age']}")
        self.gender_var.set(info["gender"])
        emo = info["emotion"]
        self.mood_var.set("-" if not emo else MOOD_LABELS.get(emo, emo.title()))
        self.scores_var.set(f"Gender scores: Woman {info['woman']:.0f}%  |  Man {info['man']:.0f}%")

    def _reset(self):
        self.age_var.set("-")
        self.gender_var.set("-")
        self.mood_var.set("-")
        self.scores_var.set("")
        self._photo = None
        self.video.configure(image="", text="Camera picture appears here")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()