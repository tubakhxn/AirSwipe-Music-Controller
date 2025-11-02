# Gesture-controlled Local Music Player

This project implements a gesture-controlled music player using OpenCV and MediaPipe for hand tracking and local playback using `pygame`.

Features
- Toggle Play/Pause with a fist ✊
- Next Track with a swipe right 👉
- Previous Track with a swipe left 👈
- Volume control with two fingers up ✌️ (vertical position maps to volume)
- Visual overlays: hand landmarks, current gesture text, cooldown indicator, small volume bar
- Gesture smoothing and cooldowns to avoid accidental repeated triggers
- Local playback using `pygame` (place audio files into `local_music/`)

Quick setup
1. Create a virtualenv and install requirements:

```powershell
python -m venv venv; .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Add audio files
- Create a folder named `local_music` in the project root and copy some `.mp3`, `.wav`, or `.ogg` files there.

4. (Optional) System volume control — Windows only

- Install `pycaw` (already in `requirements.txt`) to enable system master volume control.
- Run the script with `--system-volume` to map the gesture volume to the Windows master volume instead of the pygame player's volume.

3. Run the script:

```powershell
python gesture_spotify_player.py
```

Notes and tips
- Run locally for webcam access and low-latency controls (Colab isn't practical for live webcam gesture control).
- If gestures misfire: tune thresholds in `gesture_spotify_player.py` (buffer lengths, swipe sensitivity).
- If you want system-wide volume control on Windows, enable `pycaw` support (optional dependency).

If you'd like, I can now tune gesture sensitivity, add a small on-screen cooldown meter, or enable pycaw-based system volume control.

Colab note: A Colab notebook `gestures_spotify_colab.ipynb` is included. Colab webcam usage is less reliable and higher-latency than a local run; use local for best results.
