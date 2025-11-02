"""
Gesture-controlled music player using OpenCV + MediaPipe + Spotipy.

Controls (default mapping):
- Fist (no fingers) -> toggle play/pause
- Swipe right -> next track
- Swipe left -> previous track
- Two fingers up -> volume control by vertical position (higher = louder)

Requirements: opencv-python, mediapipe, spotipy, pycaw (optional for system volume), pygame

Before running set SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET and SPOTIPY_REDIRECT_URI as env vars
or edit the script to provide them directly (not recommended for security).

Run locally (Windows recommended):
python gesture_spotify_player.py

"""

import os
import time
import argparse
from collections import deque

import cv2
import numpy as np
import math

try:
    import mediapipe as mp
except Exception:
    raise RuntimeError("mediapipe is required: pip install mediapipe")

try:
    import pygame
except Exception:
    pygame = None

    # spotipy is optional (we currently run local-only). If you later add it, the code will try to use it.
    try:
        import spotipy
        from spotipy.oauth2 import SpotifyOAuth
    except Exception:
        spotipy = None

try:
    # pycaw is optional and Windows-only. Use to set system volume if desired.
    from ctypes import POINTER, cast
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    _has_pycaw = True
except Exception:
    _has_pycaw = False


### --- Utilities ----------------------------------------------------------------

def now():
    return time.time()


### --- Hand Detector -------------------------------------------------------------

class HandDetector:
    """Wrapper around MediaPipe Hands that returns normalized landmarks and hand center."""
    def __init__(self, max_num_hands=1, detection_conf=0.7, tracking_conf=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=max_num_hands,
                                         min_detection_confidence=detection_conf,
                                         min_tracking_confidence=tracking_conf)
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, frame, draw=True):
        # frame: BGR image
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)
        hands_data = []
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                lm = []
                for id, lm_pt in enumerate(hand_landmarks.landmark):
                    lm.append((lm_pt.x, lm_pt.y, lm_pt.z))
                # compute center
                cx = int(np.mean([p[0] for p in lm]) * w)
                cy = int(np.mean([p[1] for p in lm]) * h)
                hands_data.append({
                    'lm': lm,
                    'center': (cx, cy),
                    'raw': hand_landmarks
                })
                if draw:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return frame, hands_data


### --- Gesture Recognizer -------------------------------------------------------

class GestureRecognizer:
    """Recognize gestures robustly: fist, two fingers up, swipes based on recent centers.

    Uses timestamped center buffer to compute swipe velocity, smoothing for finger counts,
    and cooldowns to avoid repeated triggers. Tweak parameters below for sensitivity.
    """

    def __init__(self, buffer_len=6, swipe_vpx=450.0, cooldown=0.9):
        # center buffer stores tuples (x, y, t)
        self.center_buf = deque(maxlen=buffer_len)
        self.finger_buf = deque(maxlen=buffer_len)
        self.last_trigger = {}
        self.swipe_vpx = swipe_vpx  # pixels per second threshold
        self.cooldown = cooldown

    def fingers_up(self, lm):
        # lm: list of (x,y,z) normalized
        # Simple heuristic per finger (index->pinky): tip y < pip y => finger up
        tips = [8, 12, 16, 20]
        count = 0
        for tip in tips:
            try:
                if lm[tip][1] < lm[tip - 2][1]:
                    count += 1
            except Exception:
                pass
        # thumb: check if thumb tip is away from palm center horizontally
        try:
            # approximate palm center as wrist (0)
            if abs(lm[4][0] - lm[0][0]) > 0.06:
                count += 1
        except Exception:
            pass
        return count

    def smooth_count(self, cnt):
        self.finger_buf.append(cnt)
        return int(round(np.median(list(self.finger_buf))))

    def add_center(self, center):
        # center is (x_px, y_px)
        self.center_buf.append((center[0], center[1], time.time()))

    def detect_swipe(self):
        if len(self.center_buf) < 3:
            return None
        x0, y0, t0 = self.center_buf[0]
        x1, y1, t1 = self.center_buf[-1]
        dt = max(1e-3, t1 - t0)
        vx = (x1 - x0) / dt
        # require significant horizontal velocity and not too much vertical drift
        vy = (y1 - y0) / dt
        # tuned: require horizontal speed > threshold and vertical drift smaller than half horizontal
        if abs(vx) > self.swipe_vpx and abs(vy) < abs(vx) * 0.5:
            return 'right' if vx > 0 else 'left'
        return None

    def cooldown_ok(self, action):
        t = time.time()
        last = self.last_trigger.get(action, 0)
        if t - last >= self.cooldown:
            self.last_trigger[action] = t
            return True
        return False

    def recognize(self, hand):
        if not hand:
            # no hand detected
            return None, None
        lm = hand['lm']
        center = hand['center']
        self.add_center(center)
        raw_cnt = self.fingers_up(lm)
        cnt = self.smooth_count(raw_cnt)

        # Two-finger volume gesture: index + middle up, ring & pinky down
        if cnt >= 2:
            # ensure index and middle are up specifically
            idx_up = lm[8][1] < lm[6][1]
            mid_up = lm[12][1] < lm[10][1]
            ring_up = lm[16][1] < lm[14][1]
            pinky_up = lm[20][1] < lm[18][1]
            if idx_up and mid_up and not ring_up and not pinky_up:
                # volume by average vertical position of index and middle tips
                vol_norm = 1.0 - np.mean([lm[8][1], lm[12][1]])
                vol = int(np.clip(vol_norm * 100, 0, 100))
                return 'volume', vol

        # Fist detection: all finger tips are near the wrist or folded (tips below pip)
        tips = [4, 8, 12, 16, 20]
        folded = 0
        for tip in tips:
            try:
                if lm[tip][1] > lm[tip - 2][1]:
                    folded += 1
            except Exception:
                pass
        if folded >= 4:
            return 'fist', None

        # Swipe detection
        swipe = self.detect_swipe()
        if swipe == 'right':
            return 'swipe_right', None
        elif swipe == 'left':
            return 'swipe_left', None

        return None, None


### --- Spotify Controller -------------------------------------------------------

class SpotifyController:
    """Controls playback via Spotipy. Requires SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI as env vars or will prompt."""

    def __init__(self, scope='user-modify-playback-state user-read-playback-state user-read-currently-playing'):
        if spotipy is None:
            raise RuntimeError('spotipy is not installed')
        client_id = os.environ.get('SPOTIPY_CLIENT_ID')
        client_secret = os.environ.get('SPOTIPY_CLIENT_SECRET')
        redirect_uri = os.environ.get('SPOTIPY_REDIRECT_URI')
        if not (client_id and client_secret and redirect_uri):
            print('Spotify credentials not found in env. You will be prompted to login via a URL.')
        # Create SpotifyOAuth; spotipy will open a local server for redirect when possible
        self.auth_manager = SpotifyOAuth(scope=scope)
        self.sp = spotipy.Spotify(auth_manager=self.auth_manager)

    def is_available(self):
        try:
            cur = self.sp.current_playback()
            # If API works but no device, return True (we can still issue playback requests)
            return cur is not None
        except Exception:
            return False

    def toggle_play_pause(self):
        try:
            cur = self.sp.current_playback()
            if cur and cur.get('is_playing'):
                self.sp.pause_playback()
            else:
                # start playback on user's active device
                self.sp.start_playback()
            return True
        except Exception as e:
            print('Spotify play/pause error:', e)
            return False

    def next(self):
        try:
            self.sp.next_track()
            return True
        except Exception as e:
            print('Spotify next error:', e)
            return False

    def previous(self):
        try:
            self.sp.previous_track()
            return True
        except Exception as e:
            print('Spotify prev error:', e)
            return False

    def set_volume(self, vol_percent):
        try:
            self.sp.volume(vol_percent)
            return True
        except Exception as e:
            print('Spotify set volume error:', e)
            return False

    def currently_playing(self):
        try:
            cur = self.sp.current_playback()
            if cur and cur.get('item'):
                return f"{cur['item']['name']} - {', '.join([a['name'] for a in cur['item']['artists']])}"
            return None
        except Exception:
            return None


### --- Local Controller (fallback) ---------------------------------------------

class LocalController:
    def __init__(self, music_folder='local_music'):
        if pygame is None:
            raise RuntimeError('pygame is required for local playback')
        pygame.mixer.init()
        self.music_folder = music_folder
        self.track_paths = []
        if os.path.isdir(music_folder):
            for f in os.listdir(music_folder):
                if f.lower().endswith(('.mp3', '.wav', '.ogg')):
                    self.track_paths.append(os.path.join(music_folder, f))
        self.idx = 0
        self.paused = True
        self.volume = 0.5
        pygame.mixer.music.set_volume(self.volume)
        if self.track_paths:
            pygame.mixer.music.load(self.track_paths[self.idx])

    def is_available(self):
        return len(self.track_paths) > 0

    def toggle_play_pause(self):
        if not self.track_paths:
            return False
        if pygame.mixer.music.get_busy() and not self.paused:
            pygame.mixer.music.pause()
            self.paused = True
        else:
            if self.paused:
                pygame.mixer.music.unpause()
                self.paused = False
            else:
                pygame.mixer.music.play()
                self.paused = False
        return True

    def next(self):
        if not self.track_paths:
            return False
        self.idx = (self.idx + 1) % len(self.track_paths)
        pygame.mixer.music.load(self.track_paths[self.idx])
        pygame.mixer.music.play()
        self.paused = False
        return True

    def previous(self):
        if not self.track_paths:
            return False
        self.idx = (self.idx - 1) % len(self.track_paths)
        pygame.mixer.music.load(self.track_paths[self.idx])
        pygame.mixer.music.play()
        self.paused = False
        return True

    def set_volume(self, vol_percent):
        v = max(0, min(100, vol_percent)) / 100.0
        self.volume = v
        pygame.mixer.music.set_volume(self.volume)
        return True

    def currently_playing(self):
        if not self.track_paths:
            return None
        return os.path.basename(self.track_paths[self.idx])


### --- Main app -----------------------------------------------------------------

def draw_overlay(frame, text, vol=None):
    h, w = frame.shape[:2]
    # background box
    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(frame, text or '', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if vol is not None:
        # draw small volume bar
        cv2.rectangle(frame, (w - 140, 10), (w - 20, 30), (50, 50, 50), -1)
        cv2.rectangle(frame, (w - 140, 10), (w - 140 + int(vol / 100 * 120), 30), (50, 220, 50), -1)
        cv2.putText(frame, f'{vol} %', (w - 190, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_instrument_panel(frame, instruments, current_idx, vols=None, play_pulse=False, t=0.0):
    """Draw a horizontal instrument rack at the bottom of the frame.
    instruments: list of names
    current_idx: selected instrument index
    vols: optional list of volumes for each instrument (0-100)
    play_pulse: if True, animate a pulsing ring around current instrument
    t: current time for animation
    """
    h, w = frame.shape[:2]
    pad = 12
    panel_h = 90
    y0 = h - panel_h - pad
    cv2.rectangle(frame, (0, y0), (w, h), (10, 10, 10), -1)
    n = len(instruments)
    if n == 0:
        return
    slot_w = min(180, int((w - 2 * pad) / n))
    start_x = (w - (slot_w * n)) // 2
    for i, name in enumerate(instruments):
        x = start_x + i * slot_w
        y = y0 + 10
        # box
        color = (80, 80, 80)
        if i == current_idx:
            color = (70, 180, 70)
        cv2.rectangle(frame, (x + 6, y), (x + slot_w - 6, y + 56), color, -1)
        # instrument name
        cv2.putText(frame, name, (x + 12, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        # volume bar
        if vols is not None and i < len(vols):
            vv = int(vols[i])
            bx0 = x + 12
            by0 = y + 36
            bx1 = x + slot_w - 18
            by1 = y + 46
            cv2.rectangle(frame, (bx0, by0), (bx1, by1), (50, 50, 50), -1)
            fill_w = int((bx1 - bx0) * (vv / 100.0))
            cv2.rectangle(frame, (bx0, by0), (bx0 + fill_w, by1), (50, 220, 50), -1)
    # play pulse ring
    if play_pulse:
        # draw a pulsing circle around current slot
        cx = start_x + current_idx * slot_w + slot_w // 2
        cy = y0 + 28
        pulse = int(8 + 6 * (0.5 + 0.5 * math.sin(t * 6.0)))
        cv2.circle(frame, (cx, cy), 28 + pulse, (0, 200, 0), 2)


def main(debug=False):
    print('Starting gesture-controlled local player (debug=' + str(debug) + ')')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Cannot open webcam')
        return

    detector = HandDetector()
    recognizer = GestureRecognizer()

    # Initialize controllers: prefer Spotify if credentials present, fallback to local
    controller = None
    using = None
    if debug:
        print('Debug mode: not initializing audio controller. Gestures will be printed but no audio will play.')
    else:
        # try Spotify if installed and credentials provided
        if spotipy is not None:
            try:
                spc = SpotifyController()
                # If Spotify API accessible we will use it; otherwise fall back
                controller = spc
                using = 'spotify'
                print('Initialized Spotify controller (will control your active Spotify device).')
            except Exception as e:
                print('Spotify init error (will try local):', e)
                controller = None

        if controller is None:
            try:
                lc = LocalController()
                controller = lc
                using = 'local'
                if lc.is_available():
                    print('Using local playback from', lc.music_folder)
                else:
                    print('Local controller initialized but no tracks found in local_music/. Place files to enable playback.')
            except Exception as e:
                print('Local controller error or pygame missing:', e)

    # Which gestures map to which actions
    gesture_map = {
        'fist': 'toggle',
        'swipe_right': 'next',
        'swipe_left': 'prev',
        'volume': 'volume'
    }

    gesture_text = ''
    current_volume = None
    cooldown_indicator = ''
    # instrument UI state
    instruments = ['Drums', 'Bass', 'Guitar', 'Piano', 'Synth']
    current_instrument = 0
    instrument_vols = [60 for _ in instruments]
    last_action_time = 0
    action_text = ''
    play_pulse = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            out_frame, hands = detector.find_hands(frame, draw=True)
            hand = hands[0] if hands else None

            gesture, data = recognizer.recognize(hand)

            # default overlay text
            overlay_text = ''

            # act on gestures
            if gesture == 'volume':
                vol = data
                if controller:
                    controller.set_volume(vol)
                overlay_text = f'VOLUME {vol}%'
                current_volume = vol
                # update current instrument volume as visual feedback
                instrument_vols[current_instrument] = vol
                print(f'[GESTURE] volume -> {vol}%')
            elif gesture == 'fist' and recognizer.cooldown_ok('toggle'):
                ok = False
                if controller:
                    ok = controller.toggle_play_pause()
                overlay_text = 'PAUSE/PLAY' if ok else 'TOGGLE FAILED'
                action_text = 'PAUSE/PLAY'
                last_action_time = time.time()
                play_pulse = True
                print('[GESTURE] fist -> toggle play/pause')
            elif gesture == 'swipe_right' and recognizer.cooldown_ok('next'):
                ok = False
                if controller:
                    ok = controller.next()
                overlay_text = 'NEXT' if ok else 'NEXT FAILED'
                # cycle instrument forward for presentation
                current_instrument = (current_instrument + 1) % len(instruments)
                action_text = f'NEXT -> {instruments[current_instrument]}'
                last_action_time = time.time()
                play_pulse = False
                print('[GESTURE] swipe right -> next track')
            elif gesture == 'swipe_left' and recognizer.cooldown_ok('prev'):
                ok = False
                if controller:
                    ok = controller.previous()
                overlay_text = 'PREVIOUS' if ok else 'PREV FAILED'
                # cycle instrument backward for presentation
                current_instrument = (current_instrument - 1) % len(instruments)
                action_text = f'PREV -> {instruments[current_instrument]}'
                last_action_time = time.time()
                play_pulse = False
                print('[GESTURE] swipe left -> previous track')
            else:
                # compute remaining cooldown for toggle (for UI)
                if recognizer.last_trigger.get('toggle'):
                    tleft = max(0, recognizer.cooldown - (time.time() - recognizer.last_trigger.get('toggle', 0)))
                    cooldown_indicator = f'Toggle cooldown: {tleft:.1f}s' if tleft > 0 else ''
                try:
                    cur = controller.currently_playing() if controller else None
                except Exception:
                    cur = None
                overlay_text = cur or ('(no track)' if controller and not controller.is_available() else '')

            # show main overlay
            draw_overlay(out_frame, overlay_text, current_volume)
            # show instrument panel and action banners
            now_t = time.time()
            # if an action just happened, show action_text for 1.0s
            if now_t - last_action_time < 1.0 and action_text:
                cv2.putText(out_frame, action_text, (50, 120), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 220, 80), 3)
            draw_instrument_panel(out_frame, instruments, current_instrument, instrument_vols, play_pulse, now_t)
            if cooldown_indicator:
                cv2.putText(out_frame, cooldown_indicator, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,50), 2)

            cv2.imshow('Gesture Spotify Player', out_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gesture-controlled local music player')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode (no audio playback, prints gestures)')
    args = parser.parse_args()
    main(debug=args.debug)
