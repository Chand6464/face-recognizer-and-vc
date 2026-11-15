"""
Face-aware voice chatbot (free services only).

Flow:
  1. Camera scan  -> DeepFace guesses age / gender / emotion (runs locally, nothing saved)
  2. Bot speaks   -> tells the person its guess (pyttsx3, offline text-to-speech)
  3. Conversation -> microphone -> Google free speech recognition -> Groq free LLM -> speaks the answer

Needs internet for speech recognition and the LLM. The face part works offline.
Say "goodbye" to end a conversation.
"""
import os
import re
import time
import statistics
from collections import Counter, deque

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import requests
import sounddevice as sd
import speech_recognition as sr
import pyttsx3
from deepface import DeepFace

# ----------------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------------
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# If this model name stops working, pick a current one from https://console.groq.com/docs/models
# and either edit it here or set the GROQ_MODEL environment variable.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

LANGUAGE = "en-IN"          # speech recognition language (en-US, en-GB, hi-IN, ...)
SPEECH_RATE = 170           # bot speaking speed

SCAN_SAMPLES = 5            # face readings to average
SCAN_TIMEOUT = 45           # seconds before giving up on the scan
ANALYZE_WIDTH = 640         # bigger = more accurate but slower
DETECTOR = "opencv"         # try "yunet" (small download), "mtcnn" or "retinaface" for better accuracy
WINDOW = "Face scan"

SAMPLE_RATE = 16000
MIC_MIN_THRESHOLD = 250     # raise if the mic hears noise as speech, lower if it misses you
SILENCE_END = 1.2           # seconds of silence that end your sentence
MAX_WAIT = 8                # seconds to wait for you to start talking
MAX_UTTERANCE = 15          # longest single sentence in seconds
MAX_IDLE_TURNS = 3          # stop after this many silent turns in a row

EMOTION_WORDS = {
    "happy": "happy", "sad": "a little sad", "angry": "a bit angry",
    "fear": "a bit nervous", "surprise": "surprised",
    "neutral": "calm", "disgust": "a bit unimpressed",
}

recognizer = sr.Recognizer()


# ----------------------------------------------------------------------------
# TEXT TO SPEECH
# ----------------------------------------------------------------------------
def clean_for_speech(text):
    text = re.sub(r"[*_`#>~|]", "", text)
    text = re.sub(r"[^\w\s.,!?'\-:;()%$/]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def speak(text):
    text = clean_for_speech(text)
    print("BOT:", text)
    try:
        engine = pyttsx3.init()          # fresh engine each time avoids a known hang
        engine.setProperty("rate", SPEECH_RATE)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print("(speech error:", e, ")")


# ----------------------------------------------------------------------------
# SPEECH TO TEXT
# ----------------------------------------------------------------------------
def rms(chunk):
    return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))


def record_utterance():
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
        print("(listening...)")

        while True:
            data, _ = stream.read(block)
            level = rms(data)
            preroll.append(data)

            if level > threshold:
                if not started:
                    started = True
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


def listen():
    """Returns the recognized text, '' if nothing understood, or None if no speech at all."""
    try:
        raw = record_utterance()
    except Exception as e:
        print("Microphone error:", e)
        return None
    if raw is None:
        return None
    audio = sr.AudioData(raw, SAMPLE_RATE, 2)
    try:
        return recognizer.recognize_google(audio, language=LANGUAGE)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        print("Speech recognition needs internet:", e)
        return ""


# ----------------------------------------------------------------------------
# FACE SCAN
# ----------------------------------------------------------------------------
def scan_face(cap):
    ages, genders, emotions = [], [], []   # genders holds {"Woman": %, "Man": %} per reading
    start = time.time()

    while len(ages) < SCAN_SAMPLES and time.time() - start < SCAN_TIMEOUT:
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (ANALYZE_WIDTH, int(h * ANALYZE_WIDTH / w)))
        box = None

        try:
            faces = DeepFace.analyze(
                small,
                actions=["age", "gender", "emotion"],
                detector_backend=DETECTOR,
                enforce_detection=True,
                silent=True,
            )
            face = max(faces, key=lambda f: f["region"]["w"] * f["region"]["h"])
            ages.append(face["age"])
            genders.append(face["gender"])
            print(f"  reading: age {face['age']}, "
                  f"Woman {face['gender'].get('Woman', 0):.0f}% / Man {face['gender'].get('Man', 0):.0f}%")
            emotions.append(face["dominant_emotion"])
            r, k = face["region"], w / ANALYZE_WIDTH
            box = (int(r["x"] * k), int(r["y"] * k), int(r["w"] * k), int(r["h"] * k))
        except ValueError as e:
            if "could not be detected" not in str(e):
                print("Face analysis error:", e)
        except Exception as e:
            print("Face analysis error:", e)
            time.sleep(0.5)

        if box:
            x, y, bw, bh = box
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(frame, f"Scanning... {len(ages)}/{SCAN_SAMPLES}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(WINDOW, frame)
        cv2.waitKey(1)

    if len(ages) < 2:
        return None
    woman = statistics.mean(g.get("Woman", 0) for g in genders)
    man = statistics.mean(g.get("Man", 0) for g in genders)
    print(f"Average gender scores: Woman {woman:.0f}% / Man {man:.0f}%")
    return {
        "age": int(round(statistics.median(ages))),
        "gender": "Woman" if woman > man else "Man",
        "emotion": Counter(emotions).most_common(1)[0][0],
    }


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
        "emojis. Keep everything appropriate for a school. If you are unsure, say so."
    )
    if info:
        who = "man" if info["gender"] == "Man" else "woman"
        prompt += (
            f" A camera guessed the person is a {who}, about {info['age']} years old, and looks "
            f"{EMOTION_WORDS.get(info['emotion'], info['emotion'])}. These are rough guesses and "
            "may be wrong. Use them only to adjust your tone and word choice. Do not mention "
            "them unless asked, and accept it if the person corrects you."
        )
    return prompt


# ----------------------------------------------------------------------------
# ONE CONVERSATION
# ----------------------------------------------------------------------------
GOODBYE = re.compile(r"\b(bye|goodbye|quit|exit)\b", re.IGNORECASE)


def run_session(api_key):
    speak("Hello! I'm going to use the camera to guess your age and gender. "
          "Nothing is saved. Please look at the camera.")

    info = None
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if cap.isOpened():
        for _ in range(2):
            info = scan_face(cap)
            if info:
                break
            speak("I couldn't see a face. Please look straight at the camera.")
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    else:
        print("Could not open the camera. Continuing without face info.")

    history = []
    if info:
        who = "man" if info["gender"] == "Man" else "woman"
        mood = EMOTION_WORDS.get(info["emotion"], info["emotion"])
        intro = (f"My guess is that you're a {who}, around {info['age']} years old, "
                 f"and you look {mood}. Tell me if I'm wrong! What would you like to talk about?")
    else:
        intro = "I couldn't get a good look at you, but that's okay. What would you like to talk about?"
    speak(intro)
    history.append({"role": "assistant", "content": intro})

    system = {"role": "system", "content": build_system_prompt(info)}
    idle = 0

    while True:
        text = listen()

        if text is None or text == "":
            idle += 1
            if idle >= MAX_IDLE_TURNS:
                speak("I can't hear you, so I'll stop listening now. Goodbye!")
                return
            speak("Sorry, I didn't catch that." if text == "" else "I'm listening, go ahead.")
            continue

        idle = 0
        print("YOU:", text)

        if GOODBYE.search(text):
            speak("It was nice talking to you. Goodbye!")
            return

        history.append({"role": "user", "content": text})
        reply = ask_llm(api_key, [system] + history[-12:])
        history.append({"role": "assistant", "content": reply})
        speak(reply)


def main():
    api_key = load_api_key()
    if not api_key:
        print("No Groq API key found.\n"
              "1. Get a free key at https://console.groq.com\n"
              "2. In PowerShell run:  $env:GROQ_API_KEY=\"your_key_here\"\n"
              "   or save the key in a file named groq_key.txt next to this script.")
        return

    print("Note: your voice is sent to Google for speech recognition and your words to Groq "
          "for the answer. Camera images stay on this computer and are not saved.\n")
    print("Loading face models (first time takes a while)...")
    try:
        DeepFace.analyze(np.zeros((224, 224, 3), dtype=np.uint8),
                         actions=["age", "gender", "emotion"],
                         detector_backend=DETECTOR, enforce_detection=False, silent=True)
    except Exception as e:
        print("Warm-up warning:", e)

    while True:
        run_session(api_key)
        if input("\nPress Enter for the next person, or type q to quit: ").strip().lower() == "q":
            break


if __name__ == "__main__":
    main()