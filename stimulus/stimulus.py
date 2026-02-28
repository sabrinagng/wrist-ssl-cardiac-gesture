"""
Gesture Recording with Collapsible Waveform Sidebar
====================================================
- Main area: Gesture image + Text prompt + Timer
- Collapsible sidebar: Real-time waveforms (for signal monitoring)
- Each gesture: 2 actions with brief relax in between
- Data saved per repetition in organized folder structure

"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import json
import os
import random
import time
from datetime import datetime
import threading
import struct
import numpy as np
from pathlib import Path
from collections import deque

# Try to import serial
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("⚠️ pyserial not installed. Running in SIMULATION mode.")

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️ matplotlib not installed. Waveform display disabled.")

# Try to import winsound
try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False

# ==================== CONFIGURATION ====================

CONFIG = {
    # Timing parameters (in seconds)
    "rest_duration": 10,           # Rest before each gesture (preparation)
    "action_duration": 13,         # Duration of each action
    "relax_between_actions": 3,    # Relax between two actions
    "actions_per_gesture": 2,      # Number of actions per gesture
    "repetition_rest": 30,         # Rest between repetitions
    
    # Experiment parameters
    "num_repetitions": 10,
    "subject_id": "S01",
    
    # Display settings
    "fullscreen": False,
    "window_size": "1200x800",
    "background_color": "#1a1a2e",
    "text_color": "#ffffff",
    "accent_color": "#4ecca3",
    "warning_color": "#ff6b6b",
    "relax_color": "#ffd700",
    
    # Waveform display
    "waveform_window": 5,
    "sidebar_width": 400,
    "show_waveform": True,
    
    # Paths
    "gesture_images_dir": "./gesture_images",
    "output_dir": "./recordings",
    
    # Serial/DAQ settings
    "serial_port": "COM3",
    "baud_rate": 921600,
    "enable_daq": True,
}

# Gesture definitions
GESTURES = {
    0: "Static",
    1: "Hand Close",
    2: "Hand Open",
    3: "Wrist Flexion",
    4: "Wrist Extension",
    5: "Pointing Index",
    6: "Cut Something",
    7: "Flexion of Little Finger",
    8: "Tripod Grasp",
    9: "Flexion of Thumb",
    10: "Flexion of Middle Finger",
}

# Frame definitions
FRAME_HEADER = 0xAA
EXG_TYPE = 0x55
EXG_FRAME_SIZE = 14
EXG_FS = 2000

IMU_TYPE = 0x61
IMU_FRAME_SIZE = 18
IMU_FS = 1000

PPG_TYPE = 0x62
PPG_FRAME_SIZE = 20
PPG_FS = 200


# ==================== DATA COLLECTOR ====================

class DataCollector:
    """Handles real-time biosignal acquisition with per-repetition data extraction."""
    
    def __init__(self, port: str, baud_rate: int = 921600, window_sec: int = 5):
        self.port = port
        self.baud_rate = baud_rate
        self.serial = None
        self.window_sec = window_sec
        
        # Data storage (all session data)
        self.exg_data = []
        self.imu_data = []
        self.ppg_data = []
        self.exg_timestamps = []
        self.imu_timestamps = []
        self.ppg_timestamps = []
        
        # Repetition markers (start index for current repetition)
        self.rep_start_exg = 0
        self.rep_start_imu = 0
        self.rep_start_ppg = 0
        
        # Display buffers
        exg_buf_size = window_sec * EXG_FS
        imu_buf_size = window_sec * IMU_FS
        ppg_buf_size = window_sec * PPG_FS
        
        self.exg_display = [deque([2048]*exg_buf_size, maxlen=exg_buf_size) for _ in range(4)]
        self.imu_display = [deque([0]*imu_buf_size, maxlen=imu_buf_size) for _ in range(6)]
        self.ppg_display = [deque([0]*ppg_buf_size, maxlen=ppg_buf_size) for _ in range(2)]
        
        # PPG values
        self.current_hr = 0
        self.current_spo2 = 0
        self.current_conf = 0
        
        # Counters
        self.exg_count = 0
        self.imu_count = 0
        self.ppg_count = 0
        
        # State
        self.running = False
        self.session_start_time = 0
        self.collect_thread = None
        
    def start(self):
        """Start data collection."""
        if not SERIAL_AVAILABLE:
            print("⚠️ Serial not available - running in simulation mode")
            self.running = True
            self.session_start_time = time.time()
            self.collect_thread = threading.Thread(target=self._simulate_data, daemon=True)
            self.collect_thread.start()
            return True
            
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=0.1
            )
            self.serial.reset_input_buffer()
            self.running = True
            self.session_start_time = time.time()
            
            self.collect_thread = threading.Thread(target=self._collect_loop, daemon=True)
            self.collect_thread.start()
            
            print(f"✅ Serial port {self.port} opened")
            return True
            
        except Exception as e:
            print(f"❌ Failed to open serial port: {e}")
            return False
            
    def stop(self):
        """Stop data collection."""
        self.running = False
        if self.collect_thread:
            self.collect_thread.join(timeout=2.0)
        if self.serial and self.serial.is_open:
            self.serial.close()
            
    def get_current_indices(self):
        """Get current sample indices."""
        return {
            "exg_idx": self.exg_count,
            "imu_idx": self.imu_count,
            "ppg_idx": self.ppg_count,
        }
    
    def mark_repetition_start(self):
        """Mark the start of a new repetition."""
        self.rep_start_exg = len(self.exg_data)
        self.rep_start_imu = len(self.imu_data)
        self.rep_start_ppg = len(self.ppg_data)
        
    def get_repetition_data(self):
        """Get data for current repetition only."""
        exg_data = self.exg_data[self.rep_start_exg:]
        imu_data = self.imu_data[self.rep_start_imu:]
        ppg_data = self.ppg_data[self.rep_start_ppg:]
        exg_ts = self.exg_timestamps[self.rep_start_exg:]
        imu_ts = self.imu_timestamps[self.rep_start_imu:]
        ppg_ts = self.ppg_timestamps[self.rep_start_ppg:]
        
        exg_arr = np.array(exg_data, dtype=np.uint16) if exg_data else np.zeros((0, 4), dtype=np.uint16)
        imu_arr = np.array(imu_data, dtype=np.int16) if imu_data else np.zeros((0, 6), dtype=np.int16)
        ppg_arr = np.array(ppg_data, dtype=np.uint32) if ppg_data else np.zeros((0, 6), dtype=np.uint32)
        
        return {
            'exg': exg_arr,
            'imu': imu_arr,
            'ppg': ppg_arr,
            'exg_timestamps': np.array(exg_ts, dtype=np.float64),
            'imu_timestamps': np.array(imu_ts, dtype=np.float64),
            'ppg_timestamps': np.array(ppg_ts, dtype=np.float64),
        }
        
    def _collect_loop(self):
        """Data collection loop."""
        buffer = b''
        
        while self.running:
            try:
                if self.serial.in_waiting:
                    buffer += self.serial.read(self.serial.in_waiting)
                
                while len(buffer) >= 2:
                    if buffer[0] != FRAME_HEADER:
                        buffer = buffer[1:]
                        continue
                    
                    frame_type = buffer[1]
                    current_time = time.time() - self.session_start_time
                    
                    if frame_type == EXG_TYPE:
                        if len(buffer) < EXG_FRAME_SIZE:
                            break
                        
                        frame_data = buffer[2:EXG_FRAME_SIZE]
                        seq, ecg1, ecg2, emg1, emg2 = struct.unpack('<IHHHH', frame_data)
                        
                        self.exg_data.append([ecg1, ecg2, emg1, emg2])
                        self.exg_timestamps.append(current_time)
                        
                        self.exg_display[0].append(ecg1)
                        self.exg_display[1].append(ecg2)
                        self.exg_display[2].append(emg1)
                        self.exg_display[3].append(emg2)
                        
                        self.exg_count += 1
                        buffer = buffer[EXG_FRAME_SIZE:]
                    
                    elif frame_type == IMU_TYPE:
                        if len(buffer) < IMU_FRAME_SIZE:
                            break
                        
                        frame_data = buffer[2:IMU_FRAME_SIZE]
                        seq, ax, ay, az, gx, gy, gz = struct.unpack('<Ihhhhhh', frame_data)
                        
                        self.imu_data.append([ax, ay, az, gx, gy, gz])
                        self.imu_timestamps.append(current_time)
                        
                        self.imu_display[0].append(ax)
                        self.imu_display[1].append(ay)
                        self.imu_display[2].append(az)
                        self.imu_display[3].append(gx)
                        self.imu_display[4].append(gy)
                        self.imu_display[5].append(gz)
                        
                        self.imu_count += 1
                        buffer = buffer[IMU_FRAME_SIZE:]
                    
                    elif frame_type == PPG_TYPE:
                        if len(buffer) < PPG_FRAME_SIZE:
                            break
                        
                        frame_data = buffer[2:PPG_FRAME_SIZE]
                        seq, hr, spo2, conf, status, ir, red = struct.unpack('<IHHBBII', frame_data)
                        
                        self.ppg_data.append([hr, spo2, conf, status, ir, red])
                        self.ppg_timestamps.append(current_time)
                        
                        self.ppg_display[0].append(ir)
                        self.ppg_display[1].append(red)
                        
                        self.current_hr = hr
                        self.current_spo2 = spo2
                        self.current_conf = conf
                        
                        self.ppg_count += 1
                        buffer = buffer[PPG_FRAME_SIZE:]
                    
                    else:
                        buffer = buffer[1:]
                
                time.sleep(0.001)
                
            except Exception as e:
                time.sleep(0.01)
                
    def _simulate_data(self):
        """Simulate data for testing."""
        last_exg = time.time()
        last_imu = time.time()
        last_ppg = time.time()
        
        while self.running:
            current_time = time.time()
            rel_time = current_time - self.session_start_time
            
            # Simulate ExG @ 2000Hz
            if current_time - last_exg >= 1/EXG_FS:
                ecg1 = int(2048 + 500 * np.sin(2 * np.pi * 1.2 * rel_time) + np.random.randn() * 30)
                ecg2 = int(2048 + 400 * np.sin(2 * np.pi * 1.2 * rel_time + 0.1) + np.random.randn() * 30)
                emg1 = int(2048 + np.random.randn() * 150)
                emg2 = int(2048 + np.random.randn() * 150)
                
                self.exg_data.append([ecg1, ecg2, emg1, emg2])
                self.exg_timestamps.append(rel_time)
                
                self.exg_display[0].append(ecg1)
                self.exg_display[1].append(ecg2)
                self.exg_display[2].append(emg1)
                self.exg_display[3].append(emg2)
                
                self.exg_count += 1
                last_exg = current_time
            
            # Simulate IMU @ 1000Hz
            if current_time - last_imu >= 1/IMU_FS:
                ax = int(np.random.randn() * 500)
                ay = int(np.random.randn() * 500)
                az = int(16384 + np.random.randn() * 300)
                gx = int(np.random.randn() * 50)
                gy = int(np.random.randn() * 50)
                gz = int(np.random.randn() * 50)
                
                self.imu_data.append([ax, ay, az, gx, gy, gz])
                self.imu_timestamps.append(rel_time)
                
                self.imu_display[0].append(ax)
                self.imu_display[1].append(ay)
                self.imu_display[2].append(az)
                self.imu_display[3].append(gx)
                self.imu_display[4].append(gy)
                self.imu_display[5].append(gz)
                
                self.imu_count += 1
                last_imu = current_time
            
            # Simulate PPG @ 200Hz
            if current_time - last_ppg >= 1/PPG_FS:
                hr = 72 + int(np.random.randn() * 2)
                spo2 = 98
                conf = 100
                status = 3
                ir = int(100000 + 8000 * np.sin(2 * np.pi * 1.2 * rel_time) + np.random.randn() * 500)
                red = int(80000 + 6000 * np.sin(2 * np.pi * 1.2 * rel_time) + np.random.randn() * 400)
                
                self.ppg_data.append([hr, spo2, conf, status, ir, red])
                self.ppg_timestamps.append(rel_time)
                
                self.ppg_display[0].append(ir)
                self.ppg_display[1].append(red)
                
                self.current_hr = hr
                self.current_spo2 = spo2
                self.current_conf = conf
                
                self.ppg_count += 1
                last_ppg = current_time
            
            time.sleep(0.0001)
            
    def get_all_data_arrays(self):
        """Convert all session data to numpy arrays."""
        exg_arr = np.array(self.exg_data, dtype=np.uint16) if self.exg_data else np.zeros((0, 4), dtype=np.uint16)
        imu_arr = np.array(self.imu_data, dtype=np.int16) if self.imu_data else np.zeros((0, 6), dtype=np.int16)
        ppg_arr = np.array(self.ppg_data, dtype=np.uint32) if self.ppg_data else np.zeros((0, 6), dtype=np.uint32)
        
        return {
            'exg': exg_arr,
            'imu': imu_arr,
            'ppg': ppg_arr,
            'exg_timestamps': np.array(self.exg_timestamps, dtype=np.float64),
            'imu_timestamps': np.array(self.imu_timestamps, dtype=np.float64),
            'ppg_timestamps': np.array(self.ppg_timestamps, dtype=np.float64),
        }


# ==================== EVENT LOGGER ====================

class EventLogger:
    """Records experiment events with per-repetition saving."""
    
    def __init__(self, subject_id, subject_dir, collector=None):
        self.subject_id = subject_id
        self.subject_dir = subject_dir
        self.collector = collector
        self.session_start_unix = 0
        
        # All session events
        self.all_events = []
        
        # Current repetition events
        self.rep_events = []
        self.rep_start_time = 0
        
        # Session metadata
        self.metadata = {}
        
        # Create directories
        self.data_dir = os.path.join(subject_dir, "repetitions")
        self.events_dir = os.path.join(subject_dir, "events")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.events_dir, exist_ok=True)
        
    def start_session(self):
        self.session_start_unix = time.time()
        self.metadata = {
            "subject_id": self.subject_id,
            "session_start": datetime.now().isoformat(),
            "session_start_unix": self.session_start_unix,
            "config": {
                "rest_duration": CONFIG["rest_duration"],
                "action_duration": CONFIG["action_duration"],
                "relax_between_actions": CONFIG["relax_between_actions"],
                "actions_per_gesture": CONFIG["actions_per_gesture"],
                "repetition_rest": CONFIG["repetition_rest"],
                "num_repetitions": CONFIG["num_repetitions"],
            },
            "sampling_rates": {"exg": EXG_FS, "imu": IMU_FS, "ppg": PPG_FS},
            "gestures": GESTURES,
        }
        self.log_event("SESSION_START", {})
        
    def start_repetition(self, rep_num):
        """Start a new repetition - clear rep events."""
        self.rep_events = []
        self.rep_start_time = time.time()
        
    def log_event(self, event_type, data):
        current_time = time.time()
        elapsed_ms = int((current_time - self.session_start_unix) * 1000) if self.session_start_unix else 0
        rep_elapsed_ms = int((current_time - self.rep_start_time) * 1000) if self.rep_start_time else 0
        
        event = {
            "timestamp": datetime.now().isoformat(),
            "unix_time": current_time,
            "elapsed_ms": elapsed_ms,
            "rep_elapsed_ms": rep_elapsed_ms,
            "event_type": event_type,
            "data": data,
        }
        
        if self.collector:
            event["sample_indices"] = self.collector.get_current_indices()
            
        self.all_events.append(event)
        self.rep_events.append(event)
        
        idx_str = ""
        if "sample_indices" in event:
            idx = event["sample_indices"]
            idx_str = f" [E:{idx['exg_idx']} I:{idx['imu_idx']} P:{idx['ppg_idx']}]"
        print(f"[{elapsed_ms:>7}ms]{idx_str} {event_type}")
        
    def save_repetition(self, rep_num, gesture_order):
        """Save current repetition's events to JSON."""
        events_file = os.path.join(self.events_dir, f"rep_{rep_num:02d}_events.json")
        
        output = {
            "metadata": {
                "subject_id": self.subject_id,
                "repetition": rep_num,
                "gesture_order": gesture_order,
                "config": self.metadata.get("config", {}),
                "sampling_rates": self.metadata.get("sampling_rates", {}),
                "gestures": GESTURES,
            },
            "events": self.rep_events,
            "summary": {
                "total_events": len(self.rep_events),
                "total_gestures": len([e for e in self.rep_events if e["event_type"] == "GESTURE_START"]),
                "total_actions": len([e for e in self.rep_events if e["event_type"] == "ACTION_START"]),
            }
        }
        
        with open(events_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
            
        print(f"✅ Rep {rep_num} events saved: {events_file}")
        return events_file
        
    def save_session_summary(self, completed_reps):
        """Save overall session summary."""
        summary_file = os.path.join(self.subject_dir, "session_info.json")
        
        output = {
            "metadata": self.metadata,
            "session_end": datetime.now().isoformat(),
            "completed_repetitions": completed_reps,
            "total_events": len(self.all_events),
            "folder_structure": {
                "repetitions": "Contains .npz data files for each repetition",
                "events": "Contains .json event logs for each repetition"
            }
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
            
        print(f"✅ Session summary saved: {summary_file}")
        return summary_file


# ==================== MAIN APPLICATION ====================

class GestureRecorderApp:
    """Main application with collapsible waveform sidebar."""
    
    def __init__(self, config):
        self.config = config
        self.root = tk.Tk()
        self.root.title("Gesture Recording")
        
        # State
        self.current_repetition = 0
        self.current_gesture_idx = 0
        self.current_action = 0
        self.gesture_order = []
        self.is_running = False
        self.is_paused = False
        self.experiment_started = False
        self.sidebar_visible = config["show_waveform"]
        
        # Paths
        self.subject_dir = None
        
        # Components
        self.collector = None
        self.logger = None
        
        # Waveform
        self.fig = None
        self.canvas = None
        self.lines = {}
        self.axes = []
        
        # Setup
        self._setup_window()
        self._setup_layout()
        self._load_gesture_images()
        
    def _setup_window(self):
        if self.config["fullscreen"]:
            self.root.attributes("-fullscreen", True)
        else:
            self.root.geometry(self.config["window_size"])
            
        self.root.configure(bg=self.config["background_color"])
        self.root.bind_all("<Escape>", lambda e: self._on_escape())
        self.root.bind_all("<space>", lambda e: self._toggle_pause())
        self.root.bind_all("<Return>", lambda e: self._start_experiment())
        self.root.bind_all("<w>", lambda e: self._toggle_sidebar())
        self.root.focus_force()
        
    def _setup_layout(self):
        """Setup layout: Main area (center) + Collapsible sidebar (right)"""
        bg = self.config["background_color"]
        fg = self.config["text_color"]
        accent = self.config["accent_color"]
        
        # Main container
        self.main_container = tk.Frame(self.root, bg=bg)
        self.main_container.pack(expand=True, fill="both")
        
        # ========== MAIN CONTENT AREA (CENTER) ==========
        self.main_frame = tk.Frame(self.main_container, bg=bg)
        self.main_frame.pack(side="left", expand=True, fill="both")
        
        # Top bar
        self.top_frame = tk.Frame(self.main_frame, bg=bg)
        self.top_frame.pack(fill="x", padx=20, pady=10)
        
        self.progress_label = tk.Label(
            self.top_frame,
            text="Press ENTER to start",
            font=("Arial", 16, "bold"),
            bg=bg, fg=accent
        )
        self.progress_label.pack(side="left")
        
        self.status_label = tk.Label(
            self.top_frame,
            text="Ready",
            font=("Arial", 14),
            bg=bg, fg=fg
        )
        self.status_label.pack(side="right")
        
        # DAQ info
        self.daq_label = tk.Label(
            self.top_frame,
            text="",
            font=("Arial", 10),
            bg=bg, fg="#888888"
        )
        self.daq_label.pack(side="right", padx=20)
        
        # Center content - Gesture display
        self.center_frame = tk.Frame(self.main_frame, bg=bg)
        self.center_frame.pack(expand=True, fill="both", padx=40, pady=20)
        
        # Gesture name
        self.gesture_label = tk.Label(
            self.center_frame,
            text="",
            font=("Arial", 32, "bold"),
            bg=bg, fg=fg
        )
        self.gesture_label.pack(pady=(20, 10))
        
        # Action indicator (1/2)
        self.action_label = tk.Label(
            self.center_frame,
            text="",
            font=("Arial", 18),
            bg=bg, fg="#888888"
        )
        self.action_label.pack(pady=(0, 10))
        
        # Gesture image
        self.image_label = tk.Label(self.center_frame, bg=bg)
        self.image_label.pack(expand=True, pady=20)
        
        # Timer - large and prominent
        self.timer_label = tk.Label(
            self.center_frame,
            text="00:00",
            font=("Arial", 72, "bold"),
            bg=bg, fg=fg
        )
        self.timer_label.pack(pady=(20, 10))
        
        # Phase indicator
        self.phase_label = tk.Label(
            self.center_frame,
            text="",
            font=("Arial", 24),
            bg=bg, fg=accent
        )
        self.phase_label.pack(pady=(0, 20))
        
        # Bottom bar
        self.bottom_frame = tk.Frame(self.main_frame, bg=bg)
        self.bottom_frame.pack(fill="x", padx=20, pady=10)
        
        self.progress_bar = ttk.Progressbar(
            self.bottom_frame,
            length=500,
            mode="determinate"
        )
        self.progress_bar.pack(pady=(0, 10))
        
        self.instruction_label = tk.Label(
            self.bottom_frame,
            text="ENTER = Start | SPACE = Pause | W = Toggle Waveform | ESC = Exit",
            font=("Arial", 10),
            bg=bg, fg="#666666"
        )
        self.instruction_label.pack()
        
        # ========== SIDEBAR (RIGHT) - WAVEFORMS ==========
        self.sidebar_frame = tk.Frame(
            self.main_container, 
            bg="#16213e", 
            width=self.config["sidebar_width"]
        )
        if self.sidebar_visible:
            self.sidebar_frame.pack(side="right", fill="y")
        self.sidebar_frame.pack_propagate(False)
        
        # Sidebar header
        sidebar_header = tk.Frame(self.sidebar_frame, bg="#16213e")
        sidebar_header.pack(fill="x", padx=5, pady=5)
        
        tk.Label(
            sidebar_header,
            text="Signal Monitor",
            font=("Arial", 12, "bold"),
            bg="#16213e", fg="#ffffff"
        ).pack(side="left")
        
        self.toggle_btn = tk.Button(
            sidebar_header,
            text="Hide [W]",
            command=self._toggle_sidebar,
            font=("Arial", 9),
            bg="#333333", fg="#ffffff",
            relief="flat"
        )
        self.toggle_btn.pack(side="right")
        
        # Waveform area
        self.waveform_frame = tk.Frame(self.sidebar_frame, bg="#16213e")
        self.waveform_frame.pack(expand=True, fill="both", padx=5, pady=5)
        
        if MATPLOTLIB_AVAILABLE:
            self._setup_waveform_display()
            
    def _setup_waveform_display(self):
        """Setup matplotlib figure with 5 subplots in sidebar."""
        window_sec = self.config["waveform_window"]
        
        self.fig = Figure(figsize=(4, 6), facecolor='#16213e', dpi=80)
        
        channel_info = [
            ('ECG Chest', '#ff6b6b', EXG_FS),
            ('ECG Wrist', '#4ecdc4', EXG_FS),
            ('EMG 1', '#95e1d3', EXG_FS),
            ('EMG 2', '#a8e6cf', EXG_FS),
            ('PPG', '#ffd93d', PPG_FS),
        ]
        
        self.axes = []
        self.lines = {}
        
        for i, (name, color, fs) in enumerate(channel_info):
            ax = self.fig.add_subplot(5, 1, i+1)
            ax.set_facecolor('#1a1a2e')
            ax.set_ylabel(name, fontsize=7, color='white')
            ax.tick_params(colors='white', labelsize=5)
            ax.set_xlim(0, window_sec)
            ax.grid(True, alpha=0.2, color='white')
            
            for spine in ax.spines.values():
                spine.set_color('#333333')
            
            if i < 4:
                ax.set_xticklabels([])
                ax.set_ylim(0, 4095)
            else:
                ax.set_xlabel('Time (s)', fontsize=7, color='white')
            
            t = np.linspace(0, window_sec, window_sec * fs)
            init_data = [2048 if i < 4 else 100000] * len(t)
            line, = ax.plot(t, init_data, color=color, linewidth=0.5)
            self.lines[i] = line
            self.axes.append(ax)
        
        self.fig.tight_layout(pad=0.3)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.waveform_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(expand=True, fill="both")
        
    def _toggle_sidebar(self):
        """Toggle waveform sidebar visibility."""
        if self.sidebar_visible:
            self.sidebar_frame.pack_forget()
            self.sidebar_visible = False
        else:
            self.sidebar_frame.pack(side="right", fill="y")
            self.sidebar_visible = True
            
    def _load_gesture_images(self):
        """Load gesture images."""
        self.gesture_images = {}
        img_dir = self.config["gesture_images_dir"]
        os.makedirs(img_dir, exist_ok=True)
        
        for gesture_id, gesture_name in GESTURES.items():
            img_path = os.path.join(img_dir, f"gesture_{gesture_id}.png")
            
            if os.path.exists(img_path):
                img = Image.open(img_path)
                img = img.resize((280, 280), Image.Resampling.LANCZOS)
            else:
                img = self._create_placeholder_image(gesture_id, gesture_name)
                
            self.gesture_images[gesture_id] = ImageTk.PhotoImage(img)
            
    def _create_placeholder_image(self, gesture_id, gesture_name):
        img = Image.new('RGB', (280, 280), color='#2d3436')
        draw = ImageDraw.Draw(img)
        draw.rectangle([5, 5, 275, 275], outline='#4ecca3', width=2)
        
        try:
            font_large = ImageFont.truetype("arial.ttf", 70)
            font_small = ImageFont.truetype("arial.ttf", 16)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
            
        text = str(gesture_id)
        bbox = draw.textbbox((0, 0), text, font=font_large)
        text_width = bbox[2] - bbox[0]
        draw.text(((280 - text_width) // 2, 70), text, fill='#4ecca3', font=font_large)
        
        bbox = draw.textbbox((0, 0), gesture_name, font=font_small)
        text_width = bbox[2] - bbox[0]
        draw.text(((280 - text_width) // 2, 180), gesture_name, fill='#ffffff', font=font_small)
        
        return img
    
    def _create_subject_folder(self):
        """Create folder structure for subject."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.subject_dir = os.path.join(
            self.config["output_dir"],
            f"{self.config['subject_id']}_{timestamp}"
        )
        os.makedirs(self.subject_dir, exist_ok=True)
        print(f"📁 Created subject folder: {self.subject_dir}")
        return self.subject_dir
        
    def _start_daq_and_waveform(self):
        """Start DAQ and waveform update."""
        if self.config["enable_daq"]:
            self.collector = DataCollector(
                port=self.config["serial_port"],
                baud_rate=self.config["baud_rate"],
                window_sec=self.config["waveform_window"]
            )
            if not self.collector.start():
                messagebox.showwarning("DAQ", "Could not open serial port. Running in simulation mode.")
        
        self._update_waveform()
        self._update_daq_status()
        
    def _update_waveform(self):
        """Update waveform display."""
        if self.collector and MATPLOTLIB_AVAILABLE and self.sidebar_visible:
            try:
                for i in range(4):
                    data = list(self.collector.exg_display[i])
                    self.lines[i].set_ydata(data)
                    
                    if len(data) > 0:
                        dmin, dmax = min(data), max(data)
                        margin = (dmax - dmin) * 0.1 + 50
                        self.axes[i].set_ylim(dmin - margin, dmax + margin)
                
                ppg_data = list(self.collector.ppg_display[0])
                self.lines[4].set_ydata(ppg_data)
                
                if len(ppg_data) > 0 and max(ppg_data) > 0:
                    dmin, dmax = min(ppg_data), max(ppg_data)
                    margin = (dmax - dmin) * 0.1 + 1000
                    self.axes[4].set_ylim(dmin - margin, dmax + margin)
                
                self.canvas.draw_idle()
                
            except Exception:
                pass
        
        self.root.after(50, self._update_waveform)
        
    def _update_daq_status(self):
        """Update DAQ status."""
        if self.collector and self.collector.running:
            self.daq_label.config(
                text=f"E:{self.collector.exg_count:,} I:{self.collector.imu_count:,} P:{self.collector.ppg_count:,}",
                fg="#4ecca3"
            )
        self.root.after(1000, self._update_daq_status)
        
    def _start_experiment(self):
        """Start the experiment."""
        if self.is_running:
            return
        
        if not self.experiment_started:
            subject_id = self._get_subject_id()
            if not subject_id:
                return
                
            self.config["subject_id"] = subject_id
            self._create_subject_folder()
            self._start_daq_and_waveform()
            self.experiment_started = True
            
            self.progress_label.config(text="DAQ Running - Press ENTER again to begin")
            self.status_label.config(text="Monitoring...", fg="#ffd700")
            return
        
        # Create logger with subject directory
        self.logger = EventLogger(self.config["subject_id"], self.subject_dir, self.collector)
        self.logger.start_session()
        
        self.is_running = True
        self.current_repetition = 0
        
        threading.Thread(target=self._run_experiment, daemon=True).start()
        
    def _get_subject_id(self):
        """Get subject ID from user."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Setup")
        dialog.geometry("300x180")
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(dialog, text="Subject ID:", font=("Arial", 12)).pack(pady=(20, 5))
        entry_id = tk.Entry(dialog, font=("Arial", 14), width=15)
        entry_id.insert(0, "S01")
        entry_id.pack()
        entry_id.select_range(0, tk.END)
        entry_id.focus()
        
        tk.Label(dialog, text="Serial Port:", font=("Arial", 12)).pack(pady=(10, 5))
        entry_port = tk.Entry(dialog, font=("Arial", 14), width=15)
        entry_port.insert(0, self.config["serial_port"])
        entry_port.pack()
        
        result = [None]
        
        def on_ok():
            result[0] = entry_id.get().strip()
            self.config["serial_port"] = entry_port.get().strip()
            dialog.destroy()
            
        btn_frame = tk.Frame(dialog)
        btn_frame.pack(pady=15)
        tk.Button(btn_frame, text="OK", command=on_ok, width=10).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Cancel", command=dialog.destroy, width=10).pack(side="left", padx=5)
        
        entry_id.bind("<Return>", lambda e: on_ok())
        dialog.bind("<Escape>", lambda e: dialog.destroy())
        
        self.root.wait_window(dialog)
        return result[0]
        
    def _run_experiment(self):
        """Main experiment loop."""
        total_reps = self.config["num_repetitions"]
        
        for rep in range(total_reps):
            self.current_repetition = rep + 1
            
            # Mark start of repetition in collector
            if self.collector:
                self.collector.mark_repetition_start()
            
            # Start new repetition in logger
            self.logger.start_repetition(self.current_repetition)
            
            self.logger.log_event("REPETITION_START", {
                "repetition": self.current_repetition,
                "total": total_reps
            })
            
            if not self.is_running:
                break
            
            # Generate gesture order: Static (0) first, then shuffle remaining
            other_gestures = [g for g in GESTURES.keys() if g != 0]
            random.shuffle(other_gestures)
            self.gesture_order = [0] + other_gestures
            
            self.logger.log_event("GESTURE_ORDER", {
                "repetition": self.current_repetition,
                "order": self.gesture_order,
            })
            
            # Run gestures
            for idx, gesture_id in enumerate(self.gesture_order):
                self.current_gesture_idx = idx + 1
                
                while self.is_paused:
                    time.sleep(0.1)
                    
                if not self.is_running:
                    break
                    
                self._run_gesture(gesture_id)
                
            # Log repetition end
            self.logger.log_event("REPETITION_END", {
                "repetition": self.current_repetition
            })
            
            # Save this repetition's data
            self._save_repetition_data(self.current_repetition)
            
            if not self.is_running:
                break
                
            # Rest between repetitions
            if rep < total_reps - 1:
                self.logger.log_event("REPETITION_REST_START", {
                    "repetition": self.current_repetition,
                    "duration": self.config["repetition_rest"]
                })
                self._show_rest_between_reps()
                
        self._experiment_complete()
        
    def _run_gesture(self, gesture_id):
        """Run single gesture with multiple actions."""
        gesture_name = GESTURES[gesture_id]
        num_actions = self.config["actions_per_gesture"]
        
        # REST/PREPARATION phase
        self.logger.log_event("REST_START", {
            "repetition": self.current_repetition,
            "gesture_idx": self.current_gesture_idx,
            "gesture_id": gesture_id,
            "gesture_name": gesture_name,
            "duration": self.config["rest_duration"]
        })
        
        def update_rest():
            self.progress_label.config(
                text=f"Rep {self.current_repetition}/{self.config['num_repetitions']} | "
                     f"Gesture {self.current_gesture_idx}/{len(GESTURES)}"
            )
            self.status_label.config(text="Get Ready", fg=self.config["accent_color"])
            self.gesture_label.config(text=gesture_name)
            self.action_label.config(text="")
            self.phase_label.config(text="PREPARE", fg=self.config["accent_color"])
            if gesture_id in self.gesture_images:
                self.image_label.config(image=self.gesture_images[gesture_id])
            self._update_progress_bar()
            
        self.root.after(0, update_rest)
        self._countdown(self.config["rest_duration"], "rest")
        
        # Log gesture start
        self.logger.log_event("GESTURE_START", {
            "repetition": self.current_repetition,
            "gesture_idx": self.current_gesture_idx,
            "gesture_id": gesture_id,
            "gesture_name": gesture_name,
            "num_actions": num_actions
        })
        
        # Multiple actions per gesture
        for action_num in range(1, num_actions + 1):
            self.current_action = action_num
            
            while self.is_paused:
                time.sleep(0.1)
            if not self.is_running:
                return
            
            # ACTION phase
            self._play_beep(1000, 200)
            
            self.logger.log_event("ACTION_START", {
                "repetition": self.current_repetition,
                "gesture_idx": self.current_gesture_idx,
                "gesture_id": gesture_id,
                "gesture_name": gesture_name,
                "action_num": action_num,
                "duration": self.config["action_duration"]
            })
            
            def update_action(an=action_num):
                self.status_label.config(text="RECORDING", fg=self.config["warning_color"])
                self.action_label.config(text=f"Action {an}/{num_actions}")
                self.phase_label.config(text="HOLD GESTURE", fg=self.config["warning_color"])
                
            self.root.after(0, update_action)
            self._countdown(self.config["action_duration"], "action")
            
            self._play_beep(500, 200)
            
            self.logger.log_event("ACTION_END", {
                "repetition": self.current_repetition,
                "gesture_idx": self.current_gesture_idx,
                "gesture_id": gesture_id,
                "action_num": action_num
            })
            
            # RELAX between actions (except after last action)
            if action_num < num_actions:
                self.logger.log_event("RELAX_START", {
                    "repetition": self.current_repetition,
                    "gesture_idx": self.current_gesture_idx,
                    "gesture_id": gesture_id,
                    "after_action": action_num,
                    "duration": self.config["relax_between_actions"]
                })
                
                def update_relax():
                    self.status_label.config(text="Brief Rest", fg=self.config["relax_color"])
                    self.phase_label.config(text="RELAX", fg=self.config["relax_color"])
                    
                self.root.after(0, update_relax)
                self._countdown(self.config["relax_between_actions"], "relax")
                
                self.logger.log_event("RELAX_END", {
                    "repetition": self.current_repetition,
                    "gesture_idx": self.current_gesture_idx,
                    "gesture_id": gesture_id,
                })
        
        # Log gesture end
        self.logger.log_event("GESTURE_END", {
            "repetition": self.current_repetition,
            "gesture_idx": self.current_gesture_idx,
            "gesture_id": gesture_id,
            "gesture_name": gesture_name
        })
        
    def _show_rest_between_reps(self):
        """Show rest between repetitions."""
        def update():
            self.gesture_label.config(text="Rest Break")
            self.action_label.config(text="")
            self.status_label.config(text="Relax", fg=self.config["accent_color"])
            self.phase_label.config(text="BREAK", fg=self.config["accent_color"])
            self.image_label.config(image='')
            
        self.root.after(0, update)
        self._countdown(self.config["repetition_rest"], "rest")
        
    def _countdown(self, duration, phase="action"):
        """Display countdown."""
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            remaining = max(0, duration - elapsed)
            
            if remaining <= 0:
                break
                
            while self.is_paused:
                pause_start = time.time()
                while self.is_paused:
                    time.sleep(0.1)
                start_time += (time.time() - pause_start)
                
            if not self.is_running:
                return
                
            mins = int(remaining) // 60
            secs = int(remaining) % 60
            
            if phase == "action":
                color = self.config["warning_color"] if remaining <= 3 else self.config["accent_color"]
            elif phase == "relax":
                color = self.config["relax_color"]
            else:
                color = self.config["text_color"]
                
            self.root.after(0, lambda c=color, t=f"{mins:02d}:{secs:02d}": 
                           self.timer_label.config(text=t, fg=c))
            
            time.sleep(0.1)
            
    def _update_progress_bar(self):
        """Update progress bar."""
        total = self.config["num_repetitions"] * len(GESTURES)
        current = (self.current_repetition - 1) * len(GESTURES) + self.current_gesture_idx
        self.progress_bar["value"] = (current / total) * 100
        
    def _save_repetition_data(self, rep_num):
        """Save data for a single repetition."""
        # Save events JSON
        self.logger.save_repetition(rep_num, self.gesture_order)
        
        # Save sensor data NPZ
        if self.collector:
            data = self.collector.get_repetition_data()
            npz_file = os.path.join(
                self.subject_dir, "repetitions", f"rep_{rep_num:02d}_data.npz"
            )
            
            np.savez(
                npz_file,
                exg=data['exg'],
                imu=data['imu'],
                ppg=data['ppg'],
                exg_timestamps=data['exg_timestamps'],
                imu_timestamps=data['imu_timestamps'],
                ppg_timestamps=data['ppg_timestamps'],
                fs_exg=EXG_FS,
                fs_imu=IMU_FS,
                fs_ppg=PPG_FS,
                columns_exg=['ecg_chest', 'ecg_wrist', 'emg1', 'emg2'],
                columns_imu=['ax', 'ay', 'az', 'gx', 'gy', 'gz'],
                columns_ppg=['hr', 'spo2', 'conf', 'status', 'ir', 'red'],
                subject_id=self.config['subject_id'],
                repetition=rep_num,
                gesture_order=self.gesture_order,
            )
            
            print(f"✅ Rep {rep_num} data saved: {npz_file}")
            print(f"   ExG: {data['exg'].shape} | IMU: {data['imu'].shape} | PPG: {data['ppg'].shape}")
        
    def _experiment_complete(self):
        """Handle completion."""
        self.logger.log_event("SESSION_END", {
            "total_repetitions": self.current_repetition,
            "completed": self.is_running
        })
        
        # Save session summary
        self.logger.save_session_summary(self.current_repetition)
        
        if self.collector:
            self.collector.stop()
        
        self.is_running = False
        
        def update():
            self.progress_label.config(text="Complete!")
            self.gesture_label.config(text="Thank You!")
            self.action_label.config(text="")
            self.status_label.config(text="Data Saved", fg=self.config["accent_color"])
            self.timer_label.config(text="✓", fg=self.config["accent_color"])
            self.phase_label.config(text="DONE", fg=self.config["accent_color"])
            self.progress_bar["value"] = 100
            
            messagebox.showinfo("Done", f"Data saved to:\n{self.subject_dir}")
            
        self.root.after(0, update)
        
    def _play_beep(self, frequency=1000, duration=200):
        if WINSOUND_AVAILABLE:
            try:
                winsound.Beep(frequency, duration)
            except:
                pass
                
    def _toggle_pause(self):
        if not self.is_running:
            return
            
        self.is_paused = not self.is_paused
        
        if self.is_paused:
            self.logger.log_event("PAUSED", {})
            self.status_label.config(text="PAUSED", fg="#ffcc00")
            self.phase_label.config(text="PAUSED", fg="#ffcc00")
        else:
            self.logger.log_event("RESUMED", {})
            
    def _on_escape(self):
        if self.is_running:
            if messagebox.askyesno("Stop?", "Stop the experiment?"):
                self.is_running = False
                self.is_paused = False
                if self.logger:
                    self.logger.log_event("CANCELLED", {})
                    self.logger.save_session_summary(self.current_repetition)
                if self.collector:
                    self.collector.stop()
        else:
            if self.collector:
                self.collector.stop()
            self.root.quit()
            
    def run(self):
        self.root.mainloop()


# ==================== MAIN ====================

def main():
    print("=" * 60)
    print("  Gesture Recording - Cue-Guided Motor Task")
    print("=" * 60)
    print(f"\n{len(GESTURES)} gestures × {CONFIG['actions_per_gesture']} actions × {CONFIG['num_repetitions']} reps")
    print(f"Action: {CONFIG['action_duration']}s | Relax between: {CONFIG['relax_between_actions']}s")
    print(f"\nData Structure:")
    print(f"  recordings/")
    print(f"    [subject_id]_[timestamp]/")
    print(f"      repetitions/   <- .npz files")
    print(f"      events/        <- .json files")
    print(f"      session_info.json")
    print(f"\nControls:")
    print(f"  ENTER = Start | SPACE = Pause | W = Toggle Waveform | ESC = Exit")
    print("=" * 60)
    
    app = GestureRecorderApp(CONFIG)
    app.run()


if __name__ == "__main__":
    main()