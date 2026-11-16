# Face + Voice Chatbot: setup on the school PC

Camera guesses age and gender, the bot announces it, then you chat by voice (English or Hindi).
Everything runs in a **venv**. All commands are for **PowerShell**. Copy one block at a time.

The app needs only about 150 MB of packages. It does **not** need DeepFace or TensorFlow
(keep `EMOTION_WITH_DEEPFACE = False` in `voice_bot_gui.py`).

---

## 0. What the school PC needs

- 64-bit Windows 10 or 11
- Python **3.13** (64-bit) with the default "tcl/tk" option ticked
- A webcam and a microphone
- Internet for **speech recognition (Google)** and the **chat answers (Groq)**, plus a free Groq key from https://console.groq.com

Check Python:

```powershell
py -3.13 --version
```

If that fails, install it (pick one):

```powershell
winget install Python.Python.3.13
```

or download from https://www.python.org/downloads/ and tick **"Add python.exe to PATH"**. Then close and reopen PowerShell.

---

## 1. Get the project files (2 files + optional models)

Copy this folder from the USB drive / Google Drive to `C:\face-chatbot`:

```
face-chatbot\
  voice_bot_gui.py
  faceapi_agegender.py
  models\                       (optional, see below: it downloads itself if missing)
    age_gender_model.safetensors
    face_detection_yunet_2023mar.onnx   (optional, better face finder)
  wheels\                       (optional, only for the offline setup in 2B)
```

---

## 2. Install (choose A or B)

Run this first in both cases:

```powershell
cd C:\face-chatbot
py -3.13 -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

(If activation is blocked, skip it and put `.\venv\Scripts\` in front of `python` and `pip`, e.g. `.\venv\Scripts\python voice_bot_gui.py`.)

### 2A. School has fast internet: download everything automatically

```powershell
pip install numpy "opencv-python<5" safetensors requests sounddevice SpeechRecognition comtypes pywin32
pip install --no-deps pyttsx3
python -c "import faceapi_agegender as fa; print(fa.download_weights())"
```

The last line downloads the 1.7 MB age/gender model into `models\` (skipped if already there).

### 2B. Slow or blocked internet: offline install from the `wheels` folder

```powershell
pip install --no-index --find-links=wheels numpy "opencv-python<5" safetensors requests sounddevice SpeechRecognition comtypes pywin32
pip install --no-index --find-links=wheels --no-deps pyttsx3
```

Make the `wheels` folder **at home once** in Google Colab (run in a cell, then download the folder):

```
!pip download numpy "opencv-python<5" safetensors requests sounddevice SpeechRecognition comtypes pywin32 pyttsx3 --platform win_amd64 --python-version 3.13 --only-binary=:all: -d wheels
```

Also put `models\age_gender_model.safetensors` in the kit (download it from
https://huggingface.co/CognitiveOS/vision/resolve/main/age_gender_model.safetensors).

### Optional: better face finder (YuNet, 230 KB, used automatically if present)

```powershell
New-Item -ItemType Directory -Force models | Out-Null
Invoke-WebRequest -Uri "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" -OutFile "models\face_detection_yunet_2023mar.onnx"
```

If the file is only a few hundred bytes, the download failed: copy it from home instead.

---

## 3. Add your Groq key (free)

```powershell
Set-Content -Path groq_key.txt -Value "PASTE_YOUR_KEY_HERE" -Encoding ascii
```

Never upload `groq_key.txt` to GitHub. Delete it after the demo: `Remove-Item groq_key.txt`.

---

## 4. Quick checks (30 seconds, do them before the demo)

Camera + face model (a window opens, press **q** to quit; you should see a green box and `Man 90%, age 17`):

```powershell
python faceapi_agegender.py --cam
```

Internet reachable (both should print a number such as `200` or `401`, not an error):

```powershell
python -c "import requests; print(requests.get('https://api.groq.com/openai/v1/models', timeout=10).status_code)"
python -c "import requests; print(requests.get('https://www.google.com', timeout=10).status_code)"
```

Voices installed (Hindi needs one with "Hindi" in its name):

```powershell
python -c "import pyttsx3; [print(v.name) for v in pyttsx3.init().getProperty('voices')]"
```

No Hindi voice: Windows Settings > Time & language > Speech > add a voice for Hindi (India). This downloads data, so do it before the demo.

---

## 5. Run

```powershell
cd C:\face-chatbot
.\venv\Scripts\Activate.ps1
python voice_bot_gui.py
```

Pick the language you will **speak** in the drop-down at the top. Say "goodbye" to end, then click **New person**.

---

## Settings to check at the top of `voice_bot_gui.py`

| Setting | Value on the school PC |
|---|---|
| `FACE_BACKEND` | `"faceapi"` |
| `EMOTION_WITH_DEEPFACE` | `False` (mood shows "-") |
| `BOX_PADDING` | `0.1` (try `0` or `0.2` if gender looks wrong) |
| `CHAT_ANALYZE_EVERY` | `3.0` or higher on a slow PC |
| `LIVE_DURING_CHAT` | `False` if the window feels slow |
| `MIC_MIN_THRESHOLD` | lower (e.g. `150`) if it doesn't hear you |

---

## If something goes wrong

| Problem | Fix |
|---|---|
| `py` is not recognized | Python is not installed or not in PATH: redo step 0 and reopen PowerShell |
| Activation script blocked | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, or use `.\venv\Scripts\python` directly |
| `No module named 'faceapi_agegender'` | `faceapi_agegender.py` is not in the same folder as `voice_bot_gui.py` |
| `No module named ...` for something else | Re-run the install block in step 2 (inside the venv) |
| Camera window closes at once | Close other apps using the camera; Windows Settings > Privacy > Camera: allow desktop apps |
| No Groq answer | Check `groq_key.txt`, the internet check in step 4, and that the school network doesn't block Groq or Google |
| Gender swapped (men shown as women) | In `faceapi_agegender.py` change `MALE_INDEX = 0` to `1` |
| Bot shows Hindi text but is silent | No Hindi voice installed (step 4) |
| Slow / laggy | Raise `CHAT_ANALYZE_EVERY`, set `LIVE_DURING_CHAT = False` |
| Typing works but mic doesn't | Untick "Use microphone" and type; check the Windows microphone privacy setting |