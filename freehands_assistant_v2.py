"""
╔══════════════════════════════════════════════════════════════╗
║      FreeHands Ultimate Assistant - v2 (Gesture Edition)     ║
║   • All original features preserved                          ║
║   • Improved cursor control (CursorController)               ║
║   • Hand gesture control (GestureController + preview)       ║
║   • Voice toggle: "start hand control" / "stop hand control" ║
║   • SINGLE FILE – no external freehands_* modules needed     ║
╚══════════════════════════════════════════════════════════════╝

INSTALL DEPENDENCIES (run once in your venv):
    pip install mediapipe opencv-python pillow pyautogui
    pip install speechrecognition pyttsx3 wikipedia pyjokes
    pip install psutil requests beautifulsoup4 lxml
"""

# ── Standard library ──────────────────────────────────────────
import ctypes
import datetime
import json
import math
import os
import queue
import re
import smtplib
import subprocess
import sys
import threading
import time
import webbrowser
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# ── Third-party ───────────────────────────────────────────────
import cv2
import mediapipe as mp
import psutil
import pyautogui
import pyjokes
import pyttsx3
import requests
import speech_recognition as sr
import wikipedia
from bs4 import BeautifulSoup
from PIL import Image, ImageTk

# ── GUI ───────────────────────────────────────────────────────
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0.05


# ══════════════════════════════════════════════════════════════
#  CURSOR CONTROLLER
# ══════════════════════════════════════════════════════════════

class CursorController:
    """Voice-driven cursor controller.

    Handles commands like:
        "move cursor up / down / left / right"
        "set speed fast / slow / normal"
        "start moving right"   (continuous drift)
        "stop cursor"
        "click", "right click", "double click"
        "scroll up / down"
    """

    STEP  = {"slow": 10,  "normal": 40,  "fast": 120}
    DRIFT = {"slow":  4,  "normal": 12,  "fast":  30}
    TICK  = 0.03   # seconds between drift ticks

    def __init__(self, talk_fn):
        self.talk        = talk_fn
        self.speed       = "normal"
        self._drift_dir  = None
        self._drift_lock = threading.Lock()
        self._drift_thread = None

    # ── public ───────────────────────────────────────────────

    def handle(self, cmd: str) -> bool:
        """Return True if cmd was consumed, False otherwise."""
        cmd = cmd.lower().strip()

        # Speed
        if "set speed fast" in cmd or "speed fast" in cmd:
            self.speed = "fast";   self.talk("Cursor speed set to fast.");   return True
        if "set speed slow" in cmd or "speed slow" in cmd:
            self.speed = "slow";   self.talk("Cursor speed set to slow.");   return True
        if "set speed normal" in cmd or "speed normal" in cmd:
            self.speed = "normal"; self.talk("Cursor speed set to normal."); return True

        # Stop continuous movement
        if any(x in cmd for x in ["stop cursor", "stop moving", "cursor stop"]):
            self._stop_drift()
            self.talk("Cursor stopped.")
            return True

        # Continuous drift
        if "start moving" in cmd:
            d = self._parse_direction(cmd)
            if d:
                self._start_drift(d)
                self.talk(f"Cursor moving {d[2]}.")
                return True

        # Single nudge
        for phrase in ["move cursor", "cursor"]:
            if phrase in cmd:
                d = self._parse_direction(cmd)
                if d:
                    amt = self._step_amount(cmd)
                    pyautogui.moveRel(d[0] * amt, d[1] * amt)
                    self.talk(f"Cursor moved {d[2]} by {amt} pixels.")
                    return True

        # Click helpers
        if cmd in ("click", "left click", "mouse click"):
            pyautogui.click();       self.talk("Clicked.");        return True
        if cmd in ("right click", "right-click"):
            pyautogui.rightClick();  self.talk("Right-clicked.");  return True
        if cmd in ("double click", "double-click"):
            pyautogui.doubleClick(); self.talk("Double-clicked."); return True
        if cmd == "scroll up":
            pyautogui.scroll(5);  self.talk("Scrolled up.");   return True
        if cmd == "scroll down":
            pyautogui.scroll(-5); self.talk("Scrolled down."); return True

        return False

    # ── internals ────────────────────────────────────────────

    def _parse_direction(self, cmd):
        if "up"    in cmd: return ( 0, -1, "up")
        if "down"  in cmd: return ( 0,  1, "down")
        if "left"  in cmd: return (-1,  0, "left")
        if "right" in cmd: return ( 1,  0, "right")
        return None

    def _step_amount(self, cmd) -> int:
        if "fast" in cmd: return self.STEP["fast"]
        if "slow" in cmd: return self.STEP["slow"]
        return self.STEP[self.speed]

    def _start_drift(self, direction_tuple):
        self._stop_drift()
        with self._drift_lock:
            self._drift_dir = direction_tuple
        self._drift_thread = threading.Thread(
            target=self._drift_loop, daemon=True)
        self._drift_thread.start()

    def _stop_drift(self):
        with self._drift_lock:
            self._drift_dir = None

    def _drift_loop(self):
        while True:
            with self._drift_lock:
                d = self._drift_dir
            if d is None:
                break
            amt = self.DRIFT[self.speed]
            try:
                pyautogui.moveRel(d[0] * amt, d[1] * amt)
            except Exception:
                break
            time.sleep(self.TICK)


# ══════════════════════════════════════════════════════════════
#  GESTURE CONTROLLER
# ══════════════════════════════════════════════════════════════

# MediaPipe landmark indices
_WRIST      = 0
_THUMB_TIP  = 4
_INDEX_TIP  = 8
_INDEX_MCP  = 5
_MIDDLE_TIP = 12
_MIDDLE_MCP = 9
_RING_TIP   = 16
_RING_MCP   = 13
_PINKY_TIP  = 20
_PINKY_MCP  = 17


def _lm_dist(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _finger_up(lm, tip: int, mcp: int) -> bool:
    return lm[tip].y < lm[mcp].y


class GestureController:
    """
    Runs MediaPipe Hands in a background thread and translates
    hand poses into mouse events.

    Gestures:
        ☝️  Index only      → move cursor
        🤏  Pinch           → left click
        ✊  Fist            → drag
        ✌️  Two fingers     → scroll
        🖐️  Open palm ≥1s  → right click
        🤟  Three fingers   → screenshot
    """

    SMOOTH_FRAMES = 5
    PINCH_THRESH  = 0.055
    PALM_HOLD     = 1.0

    def __init__(self, talk_fn, status_fn):
        self.talk   = talk_fn
        self.status = status_fn

        self._running = False
        self._thread: threading.Thread | None = None

        self._frame_lock   = threading.Lock()
        self._latest_frame = None   # numpy array shared with preview window

        self._xs: list[float] = []
        self._ys: list[float] = []

        self._dragging        = False
        self._pinch_down      = False
        self._palm_start_time: float | None = None
        self._rclick_done     = False
        self._prev_scroll_y: float | None = None

        self._screen_w, self._screen_h = pyautogui.size()

    # ── public ───────────────────────────────────────────────

    def is_running(self) -> bool:
        return self._running

    def start(self, cam_idx: int = -1):
        if self._running:
            return
        self._running = True
        self._xs.clear(); self._ys.clear()
        self._dragging = False; self._pinch_down = False
        self._palm_start_time = None; self._rclick_done = False
        self._prev_scroll_y = None
        self._thread = threading.Thread(
            target=self._loop, args=(cam_idx,), daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self.status("✋ Gesture OFF", "#888888")
        self.talk("Hand control stopped.")

    def get_latest_frame(self):
        with self._frame_lock:
            return self._latest_frame

    # ── camera detection ─────────────────────────────────────

    @staticmethod
    def list_cameras(max_idx: int = 5) -> list[int]:
        """Return list of working camera indices (0-based)."""
        found = []
        for idx in range(max_idx):
            for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
                try:
                    cap = cv2.VideoCapture(idx, backend)
                    if not cap.isOpened():
                        cap.release()
                        continue
                    # warm-up: try up to 5 frames
                    ok, f = False, None
                    for _ in range(5):
                        ok, f = cap.read()
                        if ok and f is not None:
                            break
                    cap.release()
                    if ok and f is not None and idx not in found:
                        found.append(idx)
                        break   # found this index, skip other backends
                except Exception:
                    pass
        return found

    def _open_camera(self, preferred_idx: int = -1):
        """
        Open camera with robust fallback.
        preferred_idx = -1 means auto-detect.
        Returns opened VideoCapture or None.
        """
        indices = ([preferred_idx] if preferred_idx >= 0
                   else list(range(5)))
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]

        for idx in indices:
            for backend in backends:
                try:
                    cap = (cv2.VideoCapture(idx, backend)
                           if backend is not None
                           else cv2.VideoCapture(idx))
                    if not cap.isOpened():
                        cap.release(); continue

                    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS,           30)

                    # warm-up — some cams return blank first frames
                    # warm-up — some cams return blank first frames
                    for _ in range(10):
                        ok, frame = cap.read()
                        if ok and frame is not None:
                            bname = {cv2.CAP_DSHOW:"DSHOW",
                                     cv2.CAP_MSMF:"MSMF"}.get(backend,"AUTO")
                            self.status(f"📷 Camera {idx} ({bname})", "#00ff88")
                            return cap
                    cap.release()
                except Exception:
                    continue
        return None

    # ── main loop ────────────────────────────────────────────

    def _loop(self, cam_idx: int = -1):
        mp_hands = mp.solutions.hands
        mp_draw  = mp.solutions.drawing_utils

        self.status("📷 Opening camera…", "#ffaa00")
        cap = self._open_camera(cam_idx)
        if cap is None:
            self.status("❌ No camera found", "#ff0000")
            self.talk("Could not open any camera. Check that no other app is using it.")
            self._running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS,          30)

        self.status("✋ Gesture ON", "#00ff88")
        self.talk("Hand control started. Point your index finger to move the cursor.")

        consecutive_failures = 0

        with mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.6) as hands:

            while self._running:
                ok, frame = cap.read()

                if not ok or frame is None or frame.size == 0:
                    consecutive_failures += 1
                    if consecutive_failures == 10:
                        self.status("⚠️ Camera stalled – retrying…", "#ff8800")
                    if consecutive_failures > 30:
                        cap.release()
                        time.sleep(1.0)
                        cap = self._open_camera()
                        if cap is None:
                            self.status("❌ Camera lost", "#ff0000")
                            self.talk("Camera disconnected.")
                            break
                        consecutive_failures = 0
                    else:
                        time.sleep(0.05)
                    continue

                consecutive_failures = 0

                frame = cv2.flip(frame, 1)
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res   = hands.process(rgb)

                if res.multi_hand_landmarks:
                    lm = res.multi_hand_landmarks[0].landmark
                    mp_draw.draw_landmarks(
                        frame,
                        res.multi_hand_landmarks[0],
                        mp_hands.HAND_CONNECTIONS)
                    self._process(lm, frame)
                else:
                    if self._dragging:
                        pyautogui.mouseUp()
                        self._dragging = False
                    self._palm_start_time = None
                    self._rclick_done     = False
                    self._prev_scroll_y   = None

                label = self._gesture_label(
                    res.multi_hand_landmarks[0].landmark
                    if res.multi_hand_landmarks else None)
                cv2.putText(frame, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 128), 2)
                cv2.putText(frame, "FreeHands v2", (10, 465),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (100, 100, 255), 1)

                with self._frame_lock:
                    self._latest_frame = frame.copy()

        cap.release()
        with self._frame_lock:
            self._latest_frame = None

    # ── gesture recognition ──────────────────────────────────

    def _fingers_up(self, lm) -> list[bool]:
        thumb  = lm[_THUMB_TIP].x < lm[_THUMB_TIP - 1].x
        index  = _finger_up(lm, _INDEX_TIP,  _INDEX_MCP)
        middle = _finger_up(lm, _MIDDLE_TIP, _MIDDLE_MCP)
        ring   = _finger_up(lm, _RING_TIP,   _RING_MCP)
        pinky  = _finger_up(lm, _PINKY_TIP,  _PINKY_MCP)
        return [thumb, index, middle, ring, pinky]

    def _gesture_label(self, lm) -> str:
        if lm is None:
            return "No hand"
        f  = self._fingers_up(lm)
        up = sum(f)
        if up == 0:
            return "Fist (drag)"
        if f[1] and not f[2] and not f[3] and not f[4]:
            pinch = _lm_dist(lm[_THUMB_TIP], lm[_INDEX_TIP])
            return "Pinch (click)" if pinch < self.PINCH_THRESH else "Index (move)"
        if f[1] and f[2] and not f[3] and not f[4]:
            return "Two fingers (scroll)"
        if up >= 4:
            return "Open palm (right-click)"
        if f[1] and f[2] and f[3] and not f[4]:
            return "Three fingers (screenshot)"
        return "Unknown"

    def _process(self, lm, frame):
        f     = self._fingers_up(lm)
        up    = sum(f)
        pinch = _lm_dist(lm[_THUMB_TIP], lm[_INDEX_TIP])

        # Index only → MOVE / PINCH-CLICK
        if f[1] and not f[2] and not f[3] and not f[4]:
            if pinch >= self.PINCH_THRESH:
                if self._dragging:
                    pyautogui.mouseUp()
                    self._dragging = False
                sx, sy = self._smooth(lm[_INDEX_TIP].x, lm[_INDEX_TIP].y)
                pyautogui.moveTo(sx, sy)
                self._palm_start_time = None
                self._rclick_done     = False
                self._pinch_down      = False
            else:
                if not self._pinch_down:
                    pyautogui.click()
                    self._pinch_down = True
                    self.status("🖱 Click!", "#00ff88")

        # Fist → DRAG
        elif up == 0:
            sx, sy = self._smooth(lm[_WRIST].x, lm[_WRIST].y)
            if not self._dragging:
                pyautogui.mouseDown()
                self._dragging = True
                self.status("✊ Dragging…", "#ff8800")
            else:
                pyautogui.moveTo(sx, sy)
            self._palm_start_time = None

        # Two fingers → SCROLL
        elif f[1] and f[2] and not f[3] and not f[4]:
            if self._dragging:
                pyautogui.mouseUp()
                self._dragging = False
            cur_y = lm[_INDEX_TIP].y
            if self._prev_scroll_y is not None:
                dy = self._prev_scroll_y - cur_y
                pyautogui.scroll(int(dy * 30))
            self._prev_scroll_y   = cur_y
            self._palm_start_time = None

        # Open palm ≥4 fingers → RIGHT CLICK (hold 1 s)
        elif up >= 4:
            self._prev_scroll_y = None
            if self._dragging:
                pyautogui.mouseUp()
                self._dragging = False
            if not self._rclick_done:
                if self._palm_start_time is None:
                    self._palm_start_time = time.time()
                elif time.time() - self._palm_start_time >= self.PALM_HOLD:
                    pyautogui.rightClick()
                    self._rclick_done     = True
                    self._palm_start_time = None
                    self.status("🖱 Right-click!", "#ff00ff")

        # Three fingers → SCREENSHOT
        elif f[1] and f[2] and f[3] and not f[4]:
            if self._dragging:
                pyautogui.mouseUp()
                self._dragging = False
            fn = f"screenshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            pyautogui.screenshot(fn)
            self.talk(f"Screenshot saved: {fn}")
            time.sleep(1.5)

        else:
            self._palm_start_time = None
            self._prev_scroll_y   = None

    def _smooth(self, nx: float, ny: float):
        self._xs.append(nx)
        self._ys.append(ny)
        if len(self._xs) > self.SMOOTH_FRAMES:
            self._xs.pop(0); self._ys.pop(0)
        sx = sum(self._xs) / len(self._xs)
        sy = sum(self._ys) / len(self._ys)
        sx = max(0.0, min(1.0, (sx - 0.1) / 0.8))
        sy = max(0.0, min(1.0, (sy - 0.1) / 0.8))
        return int(sx * self._screen_w), int(sy * self._screen_h)


# ══════════════════════════════════════════════════════════════
#  GESTURE PREVIEW WINDOW
# ══════════════════════════════════════════════════════════════

class GesturePreviewWindow:
    """Floating window: live camera feed, camera selector, gesture legend."""

    FPS  = 24
    W, H = 480, 360

    def __init__(self, parent: tk.Tk, controller: GestureController):
        self.ctrl   = controller
        self.parent = parent

        self.win = tk.Toplevel(parent)
        self.win.title("✋ Hand Gesture Control")
        self.win.geometry("540x600")
        self.win.configure(bg="#0d0d1a")
        self.win.resizable(False, False)

        # Title
        tk.Label(self.win, text="✋ Hand Gesture Control",
                 font=("Segoe UI", 14, "bold"),
                 bg="#0d0d1a", fg="#00d9ff").pack(pady=(10, 2))

        # ── Camera selector row ───────────────────────────
        cam_row = tk.Frame(self.win, bg="#0d0d1a")
        cam_row.pack(fill=tk.X, padx=14, pady=(4, 2))

        tk.Label(cam_row, text="📷 Camera:",
                 font=("Segoe UI", 10, "bold"),
                 bg="#0d0d1a", fg="#aaaaaa").pack(side=tk.LEFT, padx=(0, 6))

        self._cam_var = tk.StringVar(value="Auto-detect")
        self._cam_combo = ttk.Combobox(
            cam_row, textvariable=self._cam_var,
            width=18, state="readonly",
            values=["Auto-detect", "Camera 0", "Camera 1",
                    "Camera 2", "Camera 3"])
        self._cam_combo.pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(cam_row, text="🔍 Scan",
                  font=("Segoe UI", 9, "bold"),
                  bg="#223355", fg="#ffffff", relief=tk.FLAT,
                  cursor="hand2", padx=6,
                  command=self._scan_cameras).pack(side=tk.LEFT, padx=2)

        self._cam_info = tk.Label(cam_row, text="",
                                  font=("Segoe UI", 9, "italic"),
                                  bg="#0d0d1a", fg="#888888")
        self._cam_info.pack(side=tk.LEFT, padx=6)

        # ── Live preview canvas ───────────────────────────
        self.canvas = tk.Canvas(self.win, width=self.W, height=self.H,
                                bg="#111111", highlightthickness=1,
                                highlightbackground="#333355")
        self.canvas.pack(padx=10, pady=6)

        # "No camera" placeholder text
        self.canvas.create_text(
            self.W // 2, self.H // 2,
            text="Click  ▶ Start  to open camera",
            fill="#444466", font=("Segoe UI", 12), tags="placeholder")

        # ── Status label ─────────────────────────────────
        self.lbl_status = tk.Label(self.win, text="⏹ Not running",
                                   font=("Segoe UI", 11, "bold"),
                                   bg="#0d0d1a", fg="#ffaa00")
        self.lbl_status.pack(pady=(2, 4))

        # ── Start / Stop buttons ──────────────────────────
        btn_row = tk.Frame(self.win, bg="#0d0d1a")
        btn_row.pack(pady=4)

        tk.Button(btn_row, text="▶  Start",
                  font=("Segoe UI", 11, "bold"),
                  bg="#005544", fg="#ffffff", width=10, height=2,
                  relief=tk.FLAT, cursor="hand2",
                  command=self._on_start).pack(side=tk.LEFT, padx=6)

        tk.Button(btn_row, text="■  Stop",
                  font=("Segoe UI", 11, "bold"),
                  bg="#550000", fg="#ffffff", width=10, height=2,
                  relief=tk.FLAT, cursor="hand2",
                  command=self._on_stop).pack(side=tk.LEFT, padx=6)

        # ── Gesture legend ────────────────────────────────
        legend = (
            "☝️  Index finger  →  Move cursor        "
            "🤏  Pinch  →  Left click\n"
            "✊  Fist  →  Drag                           "
            "✌️  Two fingers  →  Scroll\n"
            "🖐️  Open palm (hold 1 s)  →  Right click      "
            "🤟  Three fingers  →  Screenshot"
        )
        tk.Label(self.win, text=legend,
                 font=("Segoe UI", 8), justify=tk.LEFT,
                 bg="#0d0d1a", fg="#777777").pack(padx=14, pady=(4, 10))

        self._photo = None
        self._refresh()

    # ── helpers ──────────────────────────────────────────────

    def _selected_cam_idx(self) -> int:
        """Return chosen camera index, or -1 for auto-detect."""
        v = self._cam_var.get()
        if v.startswith("Camera "):
            try:
                return int(v.split()[-1])
            except ValueError:
                pass
        return -1

    def _scan_cameras(self):
        """Scan for available cameras and update the combo values."""
        self._cam_info.config(text="Scanning…")
        self.win.update_idletasks()

        found = GestureController.list_cameras()
        if found:
            vals = ["Auto-detect"] + [f"Camera {i}" for i in found]
            self._cam_combo.config(values=vals)
            self._cam_info.config(
                text=f"Found: {', '.join(str(i) for i in found)}",
                fg="#00ff88")
        else:
            self._cam_info.config(
                text="No cameras found!", fg="#ff4444")

    def _on_start(self):
        if self.ctrl.is_running():
            return
        idx = self._selected_cam_idx()
        self.lbl_status.config(text="⏳ Opening camera…", fg="#ffaa00")
        self.canvas.delete("placeholder")
        threading.Thread(
            target=self.ctrl.start, args=(idx,), daemon=True).start()

    def _on_stop(self):
        self.ctrl.stop()
        self.lbl_status.config(text="⏹ Stopped", fg="#ff4444")
        self.canvas.delete("all")
        self.canvas.create_text(
            self.W // 2, self.H // 2,
            text="Click  ▶ Start  to open camera",
            fill="#444466", font=("Segoe UI", 12), tags="placeholder")
        self._photo = None

    def _refresh(self):
        if not self.win.winfo_exists():
            return
        frame = self.ctrl.get_latest_frame()
        if frame is not None:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = img.resize((self.W, self.H), Image.LANCZOS)
            self._photo = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)
            self.lbl_status.config(text="🟢 Running – show your hand!",
                                   fg="#00ff88")
        elif not self.ctrl.is_running():
            if self.lbl_status.cget("text") not in (
                    "⏹ Not running", "⏹ Stopped"):
                self.lbl_status.config(text="⏹ Stopped", fg="#ff4444")
        self.win.after(int(1000 / self.FPS), self._refresh)


# ══════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════

class FreeHandsUltimateAssistant:

    # ── init ─────────────────────────────────────────────────
    def __init__(self, root):
        self.root = root
        self.root.title("🎤 FreeHands Ultimate Assistant v2")
        self.root.geometry("980x760")
        self.root.configure(bg="#0d0d1a")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.tts_queue = queue.Queue()
        threading.Thread(target=self._tts_worker, daemon=True).start()

        self.listener = sr.Recognizer()
        self.listener.energy_threshold         = 300
        self.listener.dynamic_energy_threshold  = True
        self.listener.pause_threshold           = 0.5

        self.always_listening     = False
        self.conversation_mode    = False
        self.conversation_history = []
        self.user_settings        = self.load_settings()
        self.user_name            = self.user_settings.get('user_name', 'Friend')
        self.personality          = self.user_settings.get('personality', 'friendly')

        self.cursor          = CursorController(self.talk)
        self.gesture         = GestureController(
            talk_fn=self.talk, status_fn=self._status)
        self._gesture_window = None

        self.setup_ui()
        self.root.after(1500, self.start_listening)

    # ── TTS worker ───────────────────────────────────────────
    def _tts_worker(self):
        try:
            eng    = pyttsx3.init('sapi5')
            voices = eng.getProperty('voices')
            eng.setProperty('voice',   voices[0].id)
            eng.setProperty('rate',    155)
            eng.setProperty('volume',  1.0)
        except Exception:
            eng = None

        while True:
            text = self.tts_queue.get()
            if text is None:
                break
            if eng:
                try:
                    eng.say(text); eng.runAndWait()
                except Exception:
                    try:
                        eng.stop(); eng.say(text); eng.runAndWait()
                    except Exception:
                        pass
            self.tts_queue.task_done()

    def talk(self, text):
        self.root.after(0, lambda t=text: self.add_message("FreeHands", t))
        self.tts_queue.put(text)

    def _status(self, text, color="#ffff00"):
        self.root.after(0, lambda: self.status_label.config(text=text, fg=color))

    # ── UI ───────────────────────────────────────────────────
    def setup_ui(self):
        tk.Label(self.root, text="🎤 FreeHands Ultimate Assistant v2",
                 font=("Segoe UI", 24, "bold"),
                 bg="#0d0d1a", fg="#00d9ff").pack(pady=(12, 2))

        tk.Label(self.root,
                 text="Voice Control  •  Hand Gestures  •  AI Conversations  •  Always Listening",
                 font=("Segoe UI", 10, "italic"),
                 bg="#0d0d1a", fg="#888888").pack()

        self.mode_label = tk.Label(
            self.root,
            text="🎮 MODE: COMMAND  |  💬 Chat: OFF  |  ✋ Gesture: OFF",
            font=("Segoe UI", 11, "bold"),
            bg="#0d0d1a", fg="#ffaa00")
        self.mode_label.pack(pady=(8, 2))

        self.status_label = tk.Label(
            self.root, text="⏳ Starting up…",
            font=("Segoe UI", 13, "bold"),
            bg="#0d0d1a", fg="#ffaa00")
        self.status_label.pack(pady=(2, 4))

        # Hint bar
        hint = tk.Frame(self.root, bg="#1a1a3e")
        hint.pack(fill=tk.X, padx=20, pady=4)
        tk.Label(hint,
                 text='💡 "Hey Assistant" → command  |  '
                      '"Let\'s chat" → AI mode  |  '
                      '"Start hand control" → gesture mode  |  '
                      '"Help" → all commands',
                 font=("Segoe UI", 9), bg="#1a1a3e",
                 fg="#ffff88", wraplength=920).pack(pady=5)

        # Chat area
        chat_frame = tk.Frame(self.root, bg="#1a1a3e")
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=4)

        self.chat = scrolledtext.ScrolledText(
            chat_frame, wrap=tk.WORD, font=("Segoe UI", 11),
            bg="#0a1628", fg="#ffffff", insertbackground="white",
            relief=tk.FLAT, borderwidth=0)
        self.chat.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.chat.config(state=tk.DISABLED)

        # Button row
        btn_frame = tk.Frame(self.root, bg="#0d0d1a")
        btn_frame.pack(pady=(6, 2))

        self.listen_btn = self._btn(btn_frame, "🎤 LISTENING: OFF",
                                    "#555555", "#ffffff", 16,
                                    self.toggle_listening)
        self.listen_btn.pack(side=tk.LEFT, padx=3)

        self.chat_btn = self._btn(btn_frame, "💬 CHAT",
                                  "#cc4444", "#ffffff", 10,
                                  self.toggle_chat_mode)
        self.chat_btn.pack(side=tk.LEFT, padx=3)

        self.gesture_btn = self._btn(btn_frame, "✋ GESTURES",
                                     "#005577", "#ffffff", 11,
                                     self.open_gesture_panel)
        self.gesture_btn.pack(side=tk.LEFT, padx=3)

        self._btn(btn_frame, "🆘 HELP",     "#cc0000", "#ffffff", 8,
                  self.show_help).pack(side=tk.LEFT, padx=3)
        self._btn(btn_frame, "🗑️ Clear",   "#444444", "#ffffff", 8,
                  self.clear_chat).pack(side=tk.LEFT, padx=3)
        self._btn(btn_frame, "⚙️ Settings", "#333355", "#ffffff", 9,
                  self.open_settings).pack(side=tk.LEFT, padx=3)
        self._btn(btn_frame, "❌ Exit",     "#660000", "#ffffff", 7,
                  self._on_close).pack(side=tk.LEFT, padx=3)

        # Manual input
        inp = tk.Frame(self.root, bg="#0d0d1a")
        inp.pack(fill=tk.X, padx=20, pady=(2, 10))

        placeholder = "Type a command: open notepad | tell me a joke | start hand control"
        self.input_field = tk.Entry(inp, font=("Segoe UI", 13),
                                    bg="#0a1628", fg="#555577",
                                    insertbackground="white",
                                    relief=tk.FLAT)
        self.input_field.insert(0, placeholder)
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True,
                              ipady=6, padx=(0, 8))

        def _on_focus_in(e):
            if self.input_field.get() == placeholder:
                self.input_field.delete(0, tk.END)
                self.input_field.config(fg="#ffffff")

        def _on_focus_out(e):
            if not self.input_field.get().strip():
                self.input_field.insert(0, placeholder)
                self.input_field.config(fg="#555577")

        self.input_field.bind('<FocusIn>',  _on_focus_in)
        self.input_field.bind('<FocusOut>', _on_focus_out)
        self.input_field.bind('<Return>',   lambda _: self.manual_send())

        self._btn(inp, "Send ➤", "#00d9ff", "#000000", 10,
                  self.manual_send).pack(side=tk.LEFT)

        # Welcome message
        self.add_message("FreeHands",
            f"Hello {self.user_name}! FreeHands v2 is ready.\n\n"
            "🎤  Say 'Hey Assistant' for voice commands\n"
            "✋  Click GESTURES or say 'start hand control' for camera control\n"
            "💬  Say 'Let's chat' for AI conversation\n"
            "⚙️   Settings → enter your Anthropic API key\n\n"
            "Calibrating microphone… please wait.")

    def _btn(self, parent, text, bg, fg, width, cmd):
        return tk.Button(parent, text=text,
                         font=("Segoe UI", 10, "bold"),
                         bg=bg, fg=fg, width=width, height=2,
                         relief=tk.FLAT, cursor="hand2", command=cmd)

    # ── gesture panel ────────────────────────────────────────
    def open_gesture_panel(self):
        if (self._gesture_window is None or
                not self._gesture_window.win.winfo_exists()):
            self._gesture_window = GesturePreviewWindow(
                self.root, self.gesture)
        else:
            self._gesture_window.win.lift()

    def _update_gesture_mode_label(self):
        on = self.gesture.is_running()
        self.root.after(0, lambda: self.mode_label.config(
            text=(
                f"🎮 MODE: {'CONV' if self.conversation_mode else 'CMD'}  |  "
                f"💬 Chat: {'ON' if self.conversation_mode else 'OFF'}  |  "
                f"✋ Gesture: {'ON' if on else 'OFF'}"
            ),
            fg="#00ff88" if on or self.conversation_mode else "#ffaa00"
        ))

    # ── chat display ─────────────────────────────────────────
    def add_message(self, sender, message):
        self.chat.config(state=tk.NORMAL)
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        if sender == "You":
            self.chat.insert(tk.END, f"[{ts}] 👤 You: ",      "usr_hdr")
            self.chat.insert(tk.END, f"{message}\n\n",         "usr_msg")
        else:
            self.chat.insert(tk.END, f"[{ts}] 🤖 FreeHands: ", "bot_hdr")
            self.chat.insert(tk.END, f"{message}\n\n",          "bot_msg")

        self.chat.tag_config("usr_hdr", foreground="#00ff88",
                             font=("Segoe UI", 11, "bold"))
        self.chat.tag_config("bot_hdr", foreground="#00d9ff",
                             font=("Segoe UI", 11, "bold"))
        self.chat.tag_config("usr_msg", foreground="#ccffcc",
                             font=("Segoe UI", 11))
        self.chat.tag_config("bot_msg", foreground="#ffffff",
                             font=("Segoe UI", 11))
        self.chat.see(tk.END)
        self.chat.config(state=tk.DISABLED)

    def clear_chat(self):
        self.chat.config(state=tk.NORMAL)
        self.chat.delete(1.0, tk.END)
        self.chat.config(state=tk.DISABLED)
        self.conversation_history = []
        self.add_message("FreeHands", "Chat cleared. Memory reset. Ready!")

    # ── settings ─────────────────────────────────────────────
    def load_settings(self):
        try:
            with open('freehands_settings.json', 'r') as f:
                return json.load(f)
        except Exception:
            return {'user_name': 'Friend', 'email': '',
                    'email_password': '', 'personality': 'friendly',
                    'anthropic_api_key': ''}

    def save_settings(self):
        with open('freehands_settings.json', 'w') as f:
            json.dump(self.user_settings, f, indent=4)

    def open_settings(self):
        win = tk.Toplevel(self.root)
        win.title("Settings")
        win.geometry("680x580")
        win.configure(bg="#0d0d1a")
        win.grab_set()

        tk.Label(win, text="⚙️ FreeHands Settings",
                 font=("Segoe UI", 18, "bold"),
                 bg="#0d0d1a", fg="#00d9ff").pack(pady=12)

        def lf(title, fg="#ffffff"):
            f = tk.LabelFrame(win, text=title,
                              font=("Segoe UI", 11, "bold"),
                              bg="#0d0d1a", fg=fg)
            f.pack(fill=tk.X, padx=20, pady=6)
            return f

        def row(frame, label, row_n, show=""):
            tk.Label(frame, text=label, bg="#0d0d1a",
                     fg="#cccccc", font=("Segoe UI", 11)).grid(
                row=row_n, column=0, sticky=tk.W, padx=10, pady=6)
            e = tk.Entry(frame, width=42, font=("Segoe UI", 11),
                         bg="#1a1a3e", fg="#ffffff",
                         insertbackground="white", show=show)
            e.grid(row=row_n, column=1, padx=10, pady=6)
            return e

        pf     = lf("👤 Personal")
        name_e = row(pf, "Your Name:", 0)
        name_e.insert(0, self.user_settings.get('user_name', 'Friend'))

        tk.Label(pf, text="Personality:", bg="#0d0d1a",
                 fg="#cccccc", font=("Segoe UI", 11)).grid(
            row=1, column=0, sticky=tk.W, padx=10, pady=6)
        pers_v = tk.StringVar(value=self.user_settings.get('personality', 'friendly'))
        ttk.Combobox(pf, textvariable=pers_v, width=39, state='readonly',
                     values=['friendly', 'professional',
                             'humorous', 'enthusiastic']).grid(
            row=1, column=1, padx=10, pady=6)

        af    = lf("🔑 Anthropic API Key  ⚠️ NOT OpenAI/ChatGPT!", fg="#ff6666")
        api_e = row(af, "API Key:", 0, show="*")
        api_e.insert(0, self.user_settings.get('anthropic_api_key', ''))
        tk.Label(af,
                 text="Get FREE key → https://console.anthropic.com  (keys start with sk-ant-...)",
                 font=("Segoe UI", 9, "italic"),
                 bg="#0d0d1a", fg="#ffaa44").grid(
            row=1, column=0, columnspan=2, padx=10, pady=2)

        ef      = lf("📧 Gmail Configuration")
        email_e = row(ef, "Gmail Address:", 0)
        email_e.insert(0, self.user_settings.get('email', ''))
        pass_e  = row(ef, "App Password:", 1, show="*")
        pass_e.insert(0, self.user_settings.get('email_password', ''))

        def save():
            self.user_settings.update({
                'user_name':         name_e.get(),
                'personality':       pers_v.get(),
                'anthropic_api_key': api_e.get().strip(),
                'email':             email_e.get().strip(),
                'email_password':    pass_e.get()
            })
            self.user_name   = name_e.get()
            self.personality = pers_v.get()
            self.save_settings()
            messagebox.showinfo("Saved ✅", "Settings saved!", parent=win)
            self.talk(f"Settings saved! Hello {self.user_name}!")
            win.destroy()

        tk.Button(win, text="💾  Save Settings",
                  font=("Segoe UI", 13, "bold"),
                  bg="#00d9ff", fg="#000000",
                  relief=tk.FLAT, width=22, height=2,
                  command=save).pack(pady=16)

    # ── listening loop ───────────────────────────────────────
    WAKE_WORDS = [
        'hey assistant', 'hello assistant', 'hi assistant',
        'ok assistant', 'hey freehands', 'freehands',
        'hey free hands', 'hello free hands'
    ]

    def start_listening(self):
        if not self.always_listening:
            self.always_listening = True
            self.root.after(0, lambda: self.listen_btn.config(
                text="🎤 LISTENING: ON", bg="#00aa44"))
            threading.Thread(target=self._listen_loop, daemon=True).start()

    def toggle_listening(self):
        if self.always_listening:
            self.always_listening = False
            self.listen_btn.config(text="🎤 LISTENING: OFF", bg="#555555")
            self._status("● Standby", "#888888")
            self.talk("Listening stopped.")
        else:
            self.start_listening()

    def _listen_loop(self):
        try:
            with sr.Microphone() as src:
                self._status("⏳ Calibrating mic…", "#ffaa00")
                self.listener.adjust_for_ambient_noise(src, duration=2)
        except Exception:
            self._status("❌ Microphone error", "#ff0000")
            return

        self._status("👂 Listening for 'Hey Assistant'…", "#ffff00")
        self.talk(f"Ready! Say Hey Assistant to control your PC, {self.user_name}.")

        while self.always_listening:
            try:
                with sr.Microphone() as src:
                    try:
                        audio = self.listener.listen(src, timeout=3, phrase_time_limit=5)
                    except sr.WaitTimeoutError:
                        continue
                try:
                    heard = self.listener.recognize_google(audio).lower()
                except sr.UnknownValueError:
                    continue
                except sr.RequestError:
                    self._status("⚠️  Network error…", "#ff6600")
                    time.sleep(3)
                    continue

                if any(w in heard for w in ['conversation mode', "let's chat",
                                             'chat mode', 'talk to me']):
                    if not self.conversation_mode:
                        self.root.after(0, self.toggle_chat_mode)
                    continue

                if any(w in heard for w in ['command mode', 'stop chatting',
                                             'system control']):
                    if self.conversation_mode:
                        self.root.after(0, self.toggle_chat_mode)
                    continue

                if not any(ww in heard for ww in self.WAKE_WORDS):
                    continue

                self._status("🟢 ACTIVE – speak!", "#00ff88")
                self.tts_queue.put("Yes, listening.")

                try:
                    with sr.Microphone() as src2:
                        self.listener.adjust_for_ambient_noise(src2, duration=0.2)
                        try:
                            audio2 = self.listener.listen(src2, timeout=6,
                                                          phrase_time_limit=12)
                        except sr.WaitTimeoutError:
                            self.talk("Didn't hear a command.")
                            continue
                    command = self.listener.recognize_google(audio2).lower().strip()
                except sr.UnknownValueError:
                    self.talk("Didn't catch that.")
                    continue
                except sr.RequestError:
                    self.talk("Network issue.")
                    continue

                if command:
                    self.root.after(0, lambda c=command: self.add_message("You", c))
                    threading.Thread(target=self.process_input,
                                     args=(command,), daemon=True).start()

            except Exception:
                time.sleep(0.5)
                continue

            self._status("👂 Listening for 'Hey Assistant'…", "#ffff00")

    # ── input routing ────────────────────────────────────────

    # Wake words that should be stripped when typed manually
    _WAKE_STRIP = [
        'hey assistant', 'hello assistant', 'hi assistant',
        'ok assistant', 'hey freehands', 'freehands',
        'hey free hands', 'hello free hands',
    ]

    def manual_send(self):
        raw = self.input_field.get().strip()
        placeholder = "Type a command: open notepad | tell me a joke | start hand control"
        if not raw or raw == placeholder:
            return
        self.input_field.delete(0, tk.END)
        cmd = raw.lower()

        # Strip wake word if user typed it by mistake
        for ww in self._WAKE_STRIP:
            if cmd == ww:
                # Just the wake word — give a helpful tip
                self.add_message("You", raw)
                self.add_message("FreeHands",
                    "💡 Tip: When TYPING, skip the wake word!\n"
                    "Just type the command directly, e.g:\n"
                    "  open notepad\n"
                    "  what time is it\n"
                    "  tell me a joke\n"
                    "  play Bollywood songs\n"
                    "  start hand control\n\n"
                    "Say 'Hey Assistant' only when using your VOICE.")
                return
            if cmd.startswith(ww + ' '):
                cmd = cmd[len(ww):].strip()
                raw = cmd
                break

        self.add_message("You", raw)
        threading.Thread(target=self.process_input,
                         args=(cmd,), daemon=True).start()

    def process_input(self, text):
        text = text.strip()
        if not text:
            return

        if any(w in text for w in ["let's chat", 'conversation mode',
                                    'chat mode', 'talk to me']):
            if not self.conversation_mode:
                self.root.after(0, self.toggle_chat_mode)
            return
        if any(w in text for w in ['command mode', 'stop chatting',
                                    'system control']):
            if self.conversation_mode:
                self.root.after(0, self.toggle_chat_mode)
            return

        if self.conversation_mode:
            self._ai_conversation(text)
        else:
            self._run_command(text)

    # ── conversation mode ────────────────────────────────────
    def toggle_chat_mode(self):
        self.conversation_mode = not self.conversation_mode
        if self.conversation_mode:
            self.chat_btn.config(text="💬 CHAT: ON", bg="#00aa44")
        else:
            self.chat_btn.config(text="💬 CHAT: OFF", bg="#cc4444")
        self._update_gesture_mode_label()
        if self.conversation_mode:
            self.talk(f"Chat mode on! What's on your mind, {self.user_name}?")
        else:
            self.talk("Command mode activated.")

    def _ai_conversation(self, text):
        self.conversation_history.append({"role": "user", "content": text})
        self._status("🤔 Thinking…", "#ffaa00")

        api_key = self.user_settings.get('anthropic_api_key', '').strip()
        if not api_key:
            self.talk("No Anthropic API key. Open Settings.")
            return

        prompts = {
            'friendly':     f"You are a warm, friendly voice assistant named FreeHands for {self.user_name}. Casual, 2-4 sentences.",
            'professional': f"You are a professional voice assistant named FreeHands for {self.user_name}. Concise, 2-4 sentences.",
            'humorous':     f"You are a witty voice assistant named FreeHands for {self.user_name}. Funny, 2-4 sentences.",
            'enthusiastic': f"You are an energetic voice assistant named FreeHands for {self.user_name}. Upbeat, 2-4 sentences.",
        }
        system  = prompts.get(self.personality, prompts['friendly'])
        system += " No markdown, no bullet points."

        try:
            r = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"Content-Type": "application/json",
                         "x-api-key": api_key,
                         "anthropic-version": "2023-06-01"},
                json={"model":      "claude-sonnet-4-20250514",
                      "max_tokens": 300,
                      "system":     system,
                      "messages":   self.conversation_history[-20:]},
                timeout=20)

            if r.status_code == 200:
                reply = "".join(
                    b.get('text', '') for b in r.json().get('content', [])
                    if b.get('type') == 'text').strip()
                self.conversation_history.append(
                    {"role": "assistant", "content": reply})
                self.talk(reply)
            elif r.status_code == 401:
                self.talk("API key rejected. Use an Anthropic key starting with sk-ant.")
            else:
                self.talk(f"AI error {r.status_code}.")
        except Exception:
            self.talk("Could not reach AI.")

        self._status("👂 Listening for 'Hey Assistant'…", "#ffff00")

    # ── PC helpers ───────────────────────────────────────────
    def _wake_pc(self):
        pyautogui.move(1, 0); time.sleep(0.1); pyautogui.move(-1, 0)
        self.talk("PC is awake.")

    def _turn_off_monitor(self):
        ctypes.windll.user32.SendMessageW(0xFFFF, 0x0112, 0xF170, 2)
        self.talk("Monitor turned off.")

    def _open_app(self, app_name):
        apps = {
            'notepad':        'notepad',
            'calculator':     'calc',
            'paint':          'mspaint',
            'file explorer':  'explorer',
            'explorer':       'explorer',
            'task manager':   'taskmgr',
            'control panel':  'control',
            'cmd':            'cmd',
            'command prompt': 'cmd',
            'powershell':     'powershell',
            'wordpad':        'wordpad',
            'snipping tool':  'snippingtool',
            'settings':       'ms-settings:',
            'store':          'ms-windows-store:',
            'camera':         'microsoft.windows.camera:',
            'calendar':       'outlookcal:',
            'maps':           'bingmaps:',
            'mail':           'outlookmail:',
        }
        key = app_name.lower().strip()
        if key in apps:
            cmd = apps[key]
            if cmd.endswith(':'):
                os.startfile(cmd)
            else:
                subprocess.Popen(cmd, shell=True)
            self.talk(f"Opening {app_name}")
        else:
            subprocess.Popen(f'start "" "{app_name}"', shell=True)
            self.talk(f"Trying to open {app_name}")

    def _type_text(self, text):
        pyautogui.typewrite(text, interval=0.04)
        self.talk(f"Typed: {text}")

    # ── email ────────────────────────────────────────────────
    def _parse_email(self, command):
        em = re.search(
            r'(?:to|email)\s+([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
            command)
        if not em:
            return None, None, None
        recipient = em.group(1)
        sm      = re.search(r'subject\s+(.*?)(?:\s+message|\s*$)', command, re.IGNORECASE)
        subject = sm.group(1).strip() if sm else None
        mm      = re.search(r'message\s+(.*?)$', command, re.IGNORECASE)
        body    = mm.group(1).strip() if mm else None
        return recipient, subject, body

    def _handle_email(self, command):
        recipient, subject, body = self._parse_email(command)
        if not recipient:
            self.talk("Please say the recipient email after 'to'.")
            return
        if not subject:
            self.talk("Please say the subject after 'subject'.")
            return
        if not body:
            self.talk(f"Composing email about {subject} with AI…")
            body = self._compose_with_ai(subject)
        self._send_email(recipient, subject, body)

    def _compose_with_ai(self, subject):
        key = self.user_settings.get('anthropic_api_key', '').strip()
        if not key:
            return self._email_fallback(subject)
        try:
            r = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"Content-Type": "application/json",
                         "x-api-key": key,
                         "anthropic-version": "2023-06-01"},
                json={"model":      "claude-sonnet-4-20250514",
                      "max_tokens": 400,
                      "messages":   [{"role": "user",
                                      "content": f"Write a brief professional email body for: '{subject}'. "
                                                 "No subject line. Sign as FreeHands Assistant."}]},
                timeout=20)
            if r.status_code == 200:
                return "".join(
                    b.get('text', '') for b in r.json().get('content', [])
                    if b.get('type') == 'text').strip()
        except Exception:
            pass
        return self._email_fallback(subject)

    def _email_fallback(self, subject):
        return (f"Dear Recipient,\n\nI am writing regarding: {subject}.\n\n"
                "Please let me know if you need further information.\n\n"
                "Best regards,\nSent via FreeHands Assistant")

    def _send_email(self, recipient, subject, body):
        em = self.user_settings.get('email', '')
        pw = self.user_settings.get('email_password', '')
        if not em or not pw:
            self.talk("Email not configured. Go to Settings.")
            return
        try:
            self.talk(f"Sending email to {recipient}…")
            msg = MIMEMultipart()
            msg['From']    = em
            msg['To']      = recipient
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            s = smtplib.SMTP('smtp.gmail.com', 587)
            s.starttls(); s.login(em, pw)
            s.sendmail(em, recipient, msg.as_string())
            s.quit()
            self.talk("Email sent!")
        except smtplib.SMTPAuthenticationError:
            self.talk("Email auth failed. Check app password in Settings.")
        except Exception:
            self.talk("Could not send email.")

    # ── YouTube / web / system ───────────────────────────────
    def _play_youtube(self, query):
        url = "https://www.youtube.com/results?search_query=" + query.replace(' ', '+')
        webbrowser.open(url)
        self.talk("YouTube opened.")

    def _click_video(self, n):
        y_map = {1: 340, 2: 520, 3: 700, 4: 880}
        if n in y_map:
            w, _ = pyautogui.size()
            time.sleep(0.3)
            pyautogui.click(w // 2, y_map[n])
            self.talk(f"Clicking video {n}")

    def _system_info(self):
        cpu = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory()
        bat = psutil.sensors_battery()
        msg = f"CPU at {cpu}%. RAM at {mem.percent}%."
        if bat:
            msg += f" Battery {bat.percent}%"
            msg += ", charging." if bat.power_plugged else ", not charging."
        self.talk(msg)

    def _get_weather(self, city):
        city = re.sub(r'\b(in|at|for)\b', '', city).strip()
        if not city:
            self.talk("Which city?"); return
        try:
            r = requests.get(f"https://wttr.in/{city}?format=3", timeout=6)
            self.talk(r.text.strip() if r.ok else "Could not get weather.")
        except Exception:
            self.talk("Weather service unreachable.")

    def _get_news(self):
        self.talk("Fetching headlines…")
        try:
            r    = requests.get("https://news.google.com/rss", timeout=8)
            soup = BeautifulSoup(r.content, features='xml')
            for i, item in enumerate(soup.findAll('item', limit=5), 1):
                self.talk(f"{i}. {item.title.text}")
        except Exception:
            self.talk("Could not fetch news.")

    # ── calculator ───────────────────────────────────────────
    def _nums(self, text):
        w2n = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,
               'six':6,'seven':7,'eight':8,'nine':9,'ten':10,
               'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,
               'fifteen':15,'sixteen':16,'seventeen':17,'eighteen':18,
               'nineteen':19,'twenty':20,'thirty':30,'forty':40,
               'fifty':50,'sixty':60,'seventy':70,'eighty':80,
               'ninety':90,'hundred':100,'thousand':1000}
        for w, n in w2n.items():
            text = text.replace(w, str(n))
        return [float(x) for x in re.findall(r'-?\d+\.?\d*', text)]

    def _calc(self, cmd):
        ns = self._nums(cmd)
        if len(ns) < 2:
            self.talk("I need two numbers."); return
        a, b = ns[0], ns[1]
        if   any(x in cmd for x in ['add','plus','sum']):
            self.talk(f"{a} plus {b} equals {a+b}")
        elif any(x in cmd for x in ['subtract','minus']):
            self.talk(f"{a} minus {b} equals {a-b}")
        elif any(x in cmd for x in ['multiply','times','product']):
            self.talk(f"{a} times {b} equals {a*b}")
        elif any(x in cmd for x in ['divide','divided']):
            self.talk("Cannot divide by zero." if b == 0
                      else f"{a} divided by {b} equals {a/b:.4f}")
        else:
            self.talk("Say add, subtract, multiply, or divide.")

    def _interest(self, cmd):
        ns = self._nums(cmd)
        if len(ns) < 3:
            self.talk("I need principal, rate, and years."); return
        p, r, t = ns[0], ns[1], ns[2]
        if 'simple' in cmd:
            si = p*r*t/100
            self.talk(f"Simple interest {si:.2f}. Total {p+si:.2f}")
        else:
            amt = p * ((1 + r/100) ** t)
            self.talk(f"Compound interest {amt-p:.2f}. Total {amt:.2f}")

    def _percentage(self, cmd):
        ns = self._nums(cmd)
        if len(ns) < 2:
            self.talk("I need two numbers."); return
        if 'of' in cmd:
            self.talk(f"{ns[0]}% of {ns[1]} is {ns[0]*ns[1]/100:.2f}")
        else:
            self.talk(f"{ns[0]} is {ns[0]/ns[1]*100:.2f}% of {ns[1]}"
                      if ns[1] else "Cannot divide by zero.")

    def _square(self, cmd):
        ns = self._nums(cmd)
        if not ns:
            self.talk("Give me a number."); return
        n = ns[0]
        if 'root' in cmd:
            self.talk("Negative number." if n < 0
                      else f"Square root of {n} is {n**0.5:.4f}")
        else:
            self.talk(f"Square of {n} is {n*n}")

    def _power(self, cmd):
        ns = self._nums(cmd)
        if len(ns) < 2:
            self.talk("I need base and exponent."); return
        self.talk(f"{ns[0]} to the power {ns[1]} is {ns[0]**ns[1]}")

    # ── help ─────────────────────────────────────────────────
    def show_help(self):
        threading.Thread(target=self._speak_help, daemon=True).start()

    def _speak_help(self):
        self.talk(
            "Here are all my commands. "
            "PC CONTROL: wake computer, lock, sleep, restart, shutdown, turn off monitor. "
            "APPS: open notepad, calculator, chrome, file explorer, task manager, settings. "
            "WINDOW: close window, switch window, minimize, maximize, show desktop, new tab. "
            "EDITING: copy, paste, cut, undo, redo, select all, save file. "
            "TYPE: say type then your text. "
            "YOUTUBE: say play song name. Then say first video or second video. "
            "VOLUME: volume up, volume down, mute, unmute. "
            "WEB: search Google for something. Open YouTube, Gmail, Facebook. "
            "WEATHER: weather in city name. "
            "NEWS: latest news. "
            "EMAIL: send email to address subject topic. "
            "MATHS: add, subtract, multiply, divide, simple interest, compound interest, "
            "percentage, square root, power. "
            "SYSTEM: system info, screenshot. "
            "GESTURES: say start hand control to enable camera gesture mode. "
            "Point index finger to move cursor. Pinch to click. Fist to drag. "
            "Two fingers to scroll. Open palm one second for right click. "
            "Three fingers for screenshot. Say stop hand control to disable. "
            "CURSOR: move cursor up down left right. Set speed fast slow. "
            "Start moving right. Stop cursor. "
            "AI CHAT: say let's chat to switch to conversation mode. "
            "Always say Hey Assistant first."
        )

    # ── main command processor ───────────────────────────────
    def _run_command(self, cmd):
        self._status("⚡ Processing…", "#ffaa00")

        # Help
        if any(x in cmd for x in ['help', 'what can you do', 'commands']):
            self.show_help()

        # Time / Date
        elif 'time' in cmd:
            self.talk(datetime.datetime.now().strftime("It is %I:%M %p"))
        elif 'date' in cmd:
            self.talk(datetime.datetime.now().strftime("Today is %A, %B %d, %Y"))

        # Gesture control
        elif any(x in cmd for x in ['start hand control', 'hand mode',
                                     'gesture mode', 'start gesture',
                                     'enable gestures', 'hand control']):
            self.root.after(0, self.open_gesture_panel)
            threading.Thread(target=self.gesture.start, daemon=True).start()
            self._update_gesture_mode_label()

        elif any(x in cmd for x in ['stop hand control', 'stop gesture',
                                     'disable gesture', 'disable hands']):
            self.gesture.stop()
            self._update_gesture_mode_label()

        # PC control
        elif any(x in cmd for x in ['wake', 'wake up', 'wake computer']):
            self._wake_pc()
        elif any(x in cmd for x in ['lock computer', 'lock pc', 'lock screen']):
            self.talk("Locking computer.")
            ctypes.windll.user32.LockWorkStation()
        elif ('sleep' in cmd and 'computer' in cmd) or cmd.strip() == 'sleep':
            self.talk("Putting computer to sleep.")
            os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
        elif any(x in cmd for x in ['restart', 'reboot']):
            self.talk("Restarting in 10 seconds.")
            time.sleep(10); os.system("shutdown /r /t 0")
        elif any(x in cmd for x in ['shutdown', 'shut down', 'turn off computer']):
            self.talk("Shutting down in 10 seconds.")
            time.sleep(10); os.system("shutdown /s /t 0")
        elif 'cancel' in cmd:
            os.system("shutdown /a"); self.talk("Shutdown cancelled.")
        elif any(x in cmd for x in ['turn off monitor', 'monitor off', 'screen off']):
            self._turn_off_monitor()

        # Open apps
        elif cmd.startswith('open '):
            app = cmd.replace('open', '', 1).strip()
            if app in ['chrome', 'browser', 'google']:
                webbrowser.open("https://www.google.com"); self.talk("Opening Chrome")
            elif app in ['youtube']:
                webbrowser.open("https://www.youtube.com"); self.talk("Opening YouTube")
            elif app in ['gmail', 'email']:
                webbrowser.open("https://mail.google.com"); self.talk("Opening Gmail")
            elif app in ['facebook']:
                webbrowser.open("https://www.facebook.com"); self.talk("Opening Facebook")
            elif app in ['instagram']:
                webbrowser.open("https://www.instagram.com"); self.talk("Opening Instagram")
            elif app in ['whatsapp']:
                webbrowser.open("https://web.whatsapp.com"); self.talk("Opening WhatsApp")
            else:
                self._open_app(app)

        elif any(x in cmd for x in ['close window', 'close app', 'close tab']):
            pyautogui.hotkey('ctrl', 'w') if 'tab' in cmd else pyautogui.hotkey('alt', 'f4')
            self.talk("Closed.")
        elif any(x in cmd for x in ['switch window', 'alt tab']):
            pyautogui.hotkey('alt', 'tab'); self.talk("Switching windows.")
        elif any(x in cmd for x in ['minimize all', 'show desktop']):
            pyautogui.hotkey('win', 'd'); self.talk("Desktop shown.")
        elif 'maximize' in cmd:
            pyautogui.hotkey('win', 'up'); self.talk("Maximized.")
        elif 'minimize' in cmd:
            pyautogui.hotkey('win', 'down'); self.talk("Minimized.")
        elif 'new tab' in cmd:
            pyautogui.hotkey('ctrl', 't'); self.talk("New tab.")
        elif 'copy' in cmd:
            pyautogui.hotkey('ctrl', 'c'); self.talk("Copied.")
        elif 'paste' in cmd:
            pyautogui.hotkey('ctrl', 'v'); self.talk("Pasted.")
        elif 'cut' in cmd:
            pyautogui.hotkey('ctrl', 'x'); self.talk("Cut.")
        elif 'undo' in cmd:
            pyautogui.hotkey('ctrl', 'z'); self.talk("Undone.")
        elif 'redo' in cmd:
            pyautogui.hotkey('ctrl', 'y'); self.talk("Redone.")
        elif 'select all' in cmd:
            pyautogui.hotkey('ctrl', 'a'); self.talk("All selected.")
        elif 'save' in cmd and 'file' in cmd:
            pyautogui.hotkey('ctrl', 's'); self.talk("Saved.")
        elif cmd.startswith('type '):
            self._type_text(cmd.replace('type ', '', 1))

        elif any(x in cmd for x in ['screenshot', 'screen capture']):
            fn = f"screenshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            pyautogui.screenshot(fn)
            self.talk(f"Screenshot saved as {fn}")

        # Cursor control
        elif self.cursor.handle(cmd):
            pass

        # YouTube
        elif 'play' in cmd and 'youtube' not in cmd:
            song = re.sub(r'\b(play|song|music)\b', '', cmd).strip()
            if song:
                self.talk(f"Searching {song} on YouTube.")
                threading.Thread(target=self._play_youtube,
                                 args=(song,), daemon=True).start()
            else:
                pyautogui.press('playpause'); self.talk("Playing.")
        elif any(x in cmd for x in ['first video',  'video 1']): self._click_video(1)
        elif any(x in cmd for x in ['second video', 'video 2']): self._click_video(2)
k