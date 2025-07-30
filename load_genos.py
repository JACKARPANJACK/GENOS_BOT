# Fully integrated Genos Assistant VFX system with manual transform, fade, and persistent save

import os
import sys
import json
import random
import torch
import pygame
import psutil
import pyttsx3
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
from peft import PeftModel
# PySide6 Core and GUI essentials
from PySide6.QtCore import Qt, QTimer, QEvent, QPointF
from PySide6.QtGui import QMovie, QColor, QPen
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QPushButton, QLabel, QVBoxLayout, QWidget,
    QMessageBox, QSlider, QGraphicsOpacityEffect,
    QGraphicsView, QGraphicsScene, QGraphicsProxyWidget, QGraphicsRectItem,
    QHBoxLayout, QSpinBox, QGraphicsItem
)


# -------- Resource path for PyInstaller compatibility --------

def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# Model and token setup
model_name = "google/gemma-2b-it"
token_file = os.path.join(os.path.dirname(__file__), resource_path("token.txt"))
AMBIENT_DIR = resource_path("assets/music/ambient/")
ambient_tracks = []
current_track_index = 0

if os.path.exists(AMBIENT_DIR):
    ambient_tracks = [os.path.join(AMBIENT_DIR, f) for f in os.listdir(AMBIENT_DIR) if f.endswith((".mp3", ".wav"))]
    random.shuffle(ambient_tracks)
else:
    print(f"[WARN] Ambient music folder not found: {AMBIENT_DIR}")
    fallback = resource_path("assets/music/fallback.mp3")
    if os.path.exists(fallback):
        ambient_tracks = [fallback]
        print("[INFO] Using fallback ambient track.")

GENOS_EMOTE_SETS = {
    "default": {
        "neutral": resource_path("assets/default/genos_idle.gif"),
        "angry": resource_path("assets/default/genos_angry.gif"),
        "vengeful": resource_path("assets/default/genos_vengeful.gif"),
        "happy": resource_path("assets/default/genos_happy.gif"),
        "goofy": resource_path("assets/default/genos_goofy.gif"),
        "defensive": resource_path("assets/default/genos_defensive.gif"),
        "blush": resource_path("assets/default/genos_blush.gif")
    },
    "eh": {
        "neutral": resource_path("assets/eh/genos_eh_idle.gif"),
        "angry": resource_path("assets/eh/genos_eh_angry.gif"),
        "vengeful": resource_path("assets/eh/genos_eh_vengeful.gif"),
        "happy": resource_path("assets/eh/genos_eh_happy.gif"),
        "goofy": resource_path("assets/eh/genos_eh_goofy.gif"),
        "defensive": resource_path("assets/eh/genos_eh_defensive.gif"),
        "blush": resource_path("assets/eh/genos_eh_blush.gif")
    },
    "weak": {
        "weak1": resource_path("assets/weak/genos_weak1.gif"),
        "weak2": resource_path("assets/weak/genos_weak2.gif"),
        "weak3": resource_path("assets/weak/genos_weak3.gif")
    }
}

# Load token
with open(token_file, "r") as f:
    hf_token = f.read().strip()
login(token=hf_token)

# Load model
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quant_config,
    trust_remote_code=True,
    token=hf_token
)
adapter_path = os.path.join(os.path.dirname(__file__), resource_path("genos_lora_adapter"))
model = PeftModel.from_pretrained(base_model, adapter_path)
device = 0 if torch.cuda.is_available() else -1

# Init TTS
try:
    engine = pyttsx3.init()
    engine.setProperty("rate", 170)
    engine.setProperty("voice", "english-us")
except Exception as e:
    engine = None
    print(f"Warning: TTS engine failed to initialize: {e}")

# Init Music
pygame.mixer.init()

# Emotion SFX Mapping (supports multiple variations)
EMOTION_SFX_MAP = {
    "angry": [
        resource_path("assets/sfx/angry1.mp3"),
        resource_path("assets/sfx/angry2.mp3")
    ],
    "happy": [
        resource_path("assets/sfx/happy1.mp3"),
        resource_path("assets/sfx/happy2.mp3"),
        resource_path("assets/sfx/happy3.mp3"),
        resource_path("assets/sfx/happy4.mp3"),
        resource_path("assets/sfx/happy5.mp3"),
        resource_path("assets/sfx/happy7.mp3"),
        resource_path("assets/sfx/happy6.mp3")
    ],
    "vengeful": [
        resource_path("assets/sfx/vengeful1.mp3"),
        resource_path("assets/sfx/vengeful2.mp3"),
        resource_path("assets/sfx/vengeful3.mp3")
    ],
    "goofy": [
        resource_path("assets/sfx/goofy1.mp3"),
        resource_path("assets/sfx/goofy2.mp3")
    ],
    "defensive": [
        resource_path("assets/sfx/defensive1.mp3"),
        resource_path("assets/sfx/defensive2.mp3"),
        resource_path("assets/sfx/defensive3.mp3"),
        resource_path("assets/sfx/defensive4.mp3")
    ],
    "blush": [
        resource_path("assets/sfx/blush1.mp3"),
        resource_path("assets/sfx/blush2.mp3")
    ],
    "neutral": [
        resource_path("assets/sfx/neutral1.mp3"),
        resource_path("assets/sfx/neutral2.mp3"),
        resource_path("assets/sfx/neutral3.mp3"),
        resource_path("assets/sfx/neutral4.mp3"),
        resource_path("assets/sfx/neutral5.mp3"),
        resource_path("assets/sfx/neutral6.mp3"),
        resource_path("assets/sfx/neutral7.mp3"),
        resource_path("assets/sfx/neutral8.mp3")
    ]
}

def play_ambient_music():
    global current_track_index, ambient_tracks

    if not ambient_tracks:
        print("No ambient tracks found.")
        return

    track = ambient_tracks[current_track_index]
    pygame.mixer.music.load(track)
    pygame.mixer.music.set_volume(0.3)
    pygame.mixer.music.play()

    # Extract name & announce
    track_name = os.path.basename(track).replace("_", " ").replace("-", " ").split(".")[0].title()
    speak(f"Now playing: {track_name}")


def speak(text):
    if engine:
        engine.say(text)
        engine.runAndWait()

def generate_text(prompt, model, tokenizer, max_new_tokens=100):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.95,
        temperature=0.7,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def detect_emotion(text):
    text = text.lower()
    if any(word in text for word in ["kill", "revenge", "destroy"]):
        return "vengeful"
    elif any(word in text for word in ["attack", "angry", "furious"]):
        return "angry"
    elif any(word in text for word in ["lol", "haha", "funny", "joke"]):
        return "goofy"
    elif any(word in text for word in ["love", "happy", "grateful", "thanks"]):
        return "happy"
    elif any(word in text for word in ["defend", "protect", "shield"]):
        return "defensive"
    elif any(word in text for word in ["cute", "beautiful", "demure"]):
        return "blush"
    else:
        return "neutral"

def get_battery_status():
    battery = psutil.sensors_battery()
    return battery.percent if battery else 100

# In class GenosChat(QMainWindow):
class GenosChat(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Genos Kun")
        self.resize(1280, 720)

        # ===== Scene and View for VFX Layers =====
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setAlignment(Qt.AlignCenter)

        # ===== Avatar Layer =====
        self.avatar_label = QLabel()
        self.avatar_label.setAlignment(Qt.AlignCenter)
        self.movie = QMovie(GENOS_EMOTE_SETS["default"]["neutral"])
        self.avatar_label.setMovie(self.movie)
        self.movie.start()

        self.proxy_avatar = QGraphicsProxyWidget()
        self.proxy_avatar.setWidget(self.avatar_label)
        self.proxy_avatar.setZValue(0)
        self.scene.addItem(self.proxy_avatar)

        # ===== VFX Management =====
        self.vfx_proxies = []              # (proxy, gif_name)
        self.active_proxy = None           # currently selected overlay
        self.active_rect_item = None       # bounding box
        self.current_emotion = "neutral"   # current emotion

        # ===== Undo/Redo Stacks =====
        self.undo_stack = []
        self.redo_stack = []

        # ===== Control Panel for Direct Edits =====
        self.panel = QWidget()
        self.panel_layout = QHBoxLayout()

        self.x_spin = QSpinBox()
        self.x_spin.setRange(0, 4000)
        self.x_spin.setPrefix("X: ")
        self.x_spin.valueChanged.connect(self.update_x)

        self.y_spin = QSpinBox()
        self.y_spin.setRange(0, 4000)
        self.y_spin.setPrefix("Y: ")
        self.y_spin.valueChanged.connect(self.update_y)

        self.w_spin = QSpinBox()
        self.w_spin.setRange(10, 4000)
        self.w_spin.setPrefix("W: ")
        self.w_spin.valueChanged.connect(self.update_w)

        self.h_spin = QSpinBox()
        self.h_spin.setRange(10, 4000)
        self.h_spin.setPrefix("H: ")
        self.h_spin.valueChanged.connect(self.update_h)
        
        self.rotation_slider = QSlider(Qt.Horizontal)
        self.rotation_slider.setRange(-180, 180)
        self.rotation_slider.valueChanged.connect(self.apply_rotation)

        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(80)
        self.opacity_slider.valueChanged.connect(self.apply_opacity)

        self.undo_button = QPushButton("Undo")
        self.redo_button = QPushButton("Redo")
        self.undo_button.clicked.connect(self.undo)
        self.redo_button.clicked.connect(self.redo)

        self.up_button = QPushButton("Move Up")
        self.down_button = QPushButton("Move Down")
        self.up_button.clicked.connect(self.move_layer_up)
        self.down_button.clicked.connect(self.move_layer_down)

        self.x_spin.setToolTip("X position of the selected VFX")
        self.y_spin.setToolTip("Y position of the selected VFX")
        self.w_spin.setToolTip("Width of the selected VFX")
        self.h_spin.setToolTip("Height of the selected VFX")
        self.rotation_slider.setToolTip("Rotate selected VFX (-180° to 180°)")
        self.opacity_slider.setToolTip("Opacity of selected VFX (0-100%)")

        self.x_spin.valueChanged.connect(self.update_x)
        self.y_spin.valueChanged.connect(self.update_y)
        self.w_spin.valueChanged.connect(self.update_w)
        self.h_spin.valueChanged.connect(self.update_h)

        self.grid_size = 10
        
        for widget in [
            self.x_spin, self.y_spin, self.w_spin, self.h_spin,
            self.rotation_slider, self.opacity_slider,
            self.undo_button, self.redo_button,
            self.up_button, self.down_button
        ]:
            self.panel_layout.addWidget(widget)

        self.panel.setLayout(self.panel_layout)
        self.panel.setFixedHeight(50)

        # ===== Text Input/Output for Chat =====
        self.input_box = QTextEdit()
        self.input_box.setPlaceholderText("Ask Genos something...")
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_prompt)

        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)

        # ===== Layout Assembly =====
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.view)
        main_layout.addWidget(self.panel)
        main_layout.addWidget(self.input_box)
        main_layout.addWidget(self.send_button)
        main_layout.addWidget(self.output_box)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # ===== Music Timer and Ambient Music =====
        self.music_timer = QTimer(self)
        self.music_timer.timeout.connect(self.check_music_end)
        self.music_timer.start(1000)
        play_ambient_music()

        # ===== Load VFX Config =====
        self.effects_config_file = "effects_config.json"
        self.effects_config = self.load_effects_config()

        # ===== Transformation States =====
        self.current_mode = "base"
        self.transform_sets = {
            "base": resource_path("assets/default/genos_idle.gif"),
            "combat": resource_path("assets/eh/genos_combat.gif")
        }

        # ===== Battery Warning =====
        self.low_battery_warned = False

        # ===== Snap-to-grid =====
        self.grid_size = 10

        # ===== Idle Quotes =====
        self.idle_quotes = [
            "Remaining on standby. Awaiting orders.",
            "Systems nominal. All threats neutralized.",
            "Sometimes I wonder what it would have been like... to remain human.",
            "I will not let the past define my limits.",
            "Core temperature: stable. Armor integrity: optimal.",
            "Nanofiber weave holding at 99.8% efficiency.",
            "Targeting sensors recalibrated. Response time: 0.014 seconds.",
            "All functions within standard deviation. Power levels: sufficient.",
            "Vocal output check: functional.",
            "I wonder... is this what peace feels like?"
        ]
        self.idle_timer = QTimer(self)
        self.idle_timer.timeout.connect(self.play_idle_quote)
        self.idle_timer.start(90000)  # Every 1.5 minutes

        # ===== Battery Check Timer =====
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_battery)
        self.timer.start(60000)  # Every 60 seconds

    def send_prompt(self):
        prompt = self.input_box.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Empty Prompt", "Please enter a prompt.")
            return

        self.output_box.append(f"You: {prompt}")
        self.output_box.append("Genos: Thinking...")
        QApplication.processEvents()

        try:
            result = generate_text(prompt, model, tokenizer)
            combined_text = prompt + " " + result
            emotion = detect_emotion(combined_text)
            self.update_emote(emotion)
            self.output_box.append(f"Genos: {result}")
            speak(result)
            self.play_emotion_sfx(emotion)
            self.apply_vfx(emotion)

            # ✅ Trigger transformation by keyword
            lower_prompt = prompt.lower()
            if "combat mode" in lower_prompt or "engage" in lower_prompt or "combat" in lower_prompt:
                self.transform_to("combat")
                self.current_mode = "combat"
            elif "standby" in lower_prompt or "power down" in lower_prompt or "reboot" in lower_prompt:
                self.transform_to("base")
                self.current_mode = "base"
            # 🔊 Music/ambient keyword triggers
            elif "next song" in lower_prompt or "skip track" in lower_prompt:
                self.next_track()

            elif "calm" in lower_prompt or "relax" in lower_prompt:
                for i, path in enumerate(ambient_tracks):
                    if "calm" in path or "ambient" in path:
                        current_track_index = i
                        play_ambient_music()
                        break

            elif "intense" in lower_prompt or "combat theme" in lower_prompt:
                for i, path in enumerate(ambient_tracks):
                    if "combat" in path or "intense" in path:
                        current_track_index = i
                        play_ambient_music()
                        break

        except Exception as e:
            self.output_box.append(f"Error: {str(e)}")
        
    def update_emote(self, emotion):
        battery = get_battery_status()
        if battery < 20:
            emote_path = random.choice(list(GENOS_EMOTE_SETS["weak"].values()))
        else:
            # Choose emote set based on current mode
            if self.current_mode == "base":
                emote_set = GENOS_EMOTE_SETS["default"]
            elif self.current_mode == "combat":
                emote_set = GENOS_EMOTE_SETS["eh"]
            #elif self.current_mode == "burning_core":
            #    emote_set = GENOS_EMOTE_SETS["eh"]  # You can make a new set if needed
            else:
                emote_set = GENOS_EMOTE_SETS["default"]  # fallback
            
            emote_path = emote_set.get(emotion, emote_set["neutral"])

        self.movie.stop()
        self.movie = QMovie(emote_path)
        self.avatar_label.setMovie(self.movie)
        self.movie.start()

    def play_emotion_sfx(self, emotion):
        sound_choices = EMOTION_SFX_MAP.get(emotion)
        if sound_choices:
            sfx_path = random.choice(sound_choices)
            if os.path.exists(sfx_path):
                pygame.mixer.Sound(sfx_path).play()

    def apply_vfx(self, state_name):
        self.current_emotion = state_name
        self.load_vfx_layers(state_name)

    def play_idle_quote(self):
        if engine and self.isActiveWindow():
            quote = random.choice(self.idle_quotes)
            speak(quote)
            self.output_box.append(f"Genos (idle): {quote}")

            # Randomly trigger emotion visuals + sound
            emotion = random.choice(["neutral", "happy", "vengeful", "defensive", "blush", "goofy", "angry"])
            self.update_emote(emotion)
            self.play_emotion_sfx(emotion)
            self.apply_vfx(emotion)
            
    def opacity_slider_changed(self, value):
        opacity = value / 100.0
        self.vfx_opacity_effect.setOpacity(opacity)
        self.effects_config["opacity"] = opacity
        self.save_effects_config()

    def load_expression_vfx_after_transform(self):
        # fallback to neutral or current emotion
        emotion = "neutral"
        self.apply_vfx(emotion)
        
    def _end_transform_vfx_and_load_expression(self):
        # Clear transform VFX
        self.vfx_label.clear()
        self.output_box.append("✨ Transform VFX ended. Returning to expression VFX.")

        # Load current expression VFX
        emotion = getattr(self, 'current_emotion', 'neutral')
        self.apply_vfx(emotion)


    def check_battery(self):
        battery = get_battery_status()
        if battery < 20 and not self.low_battery_warned:
            sfx_path = os.path.join("assets", "sfx", "low_battery.mp3")
            if os.path.exists(sfx_path):
                pygame.mixer.Sound(sfx_path).play()
            self.low_battery_warned = True
            
    def check_music_end(self):
        global current_track_index, ambient_tracks
        if not pygame.mixer.music.get_busy():
            current_track_index = (current_track_index + 1) % len(ambient_tracks)
            play_ambient_music()

    def next_track(self):
        global current_track_index, ambient_tracks
        if not ambient_tracks:
            return
        current_track_index = (current_track_index + 1) % len(ambient_tracks)
        play_ambient_music()
        
    def load_effects_config(self):
        if os.path.exists(self.effects_config_file):
            with open(self.effects_config_file, "r") as f:
                return json.load(f)

    def save_effects_config(self):
        with open(self.effects_config_file, "w") as f:
            json.dump(self.effects_config, f, indent=4)

    def update_rotation(self, value):
        if self.active_proxy:
            self.active_proxy.setRotation(value)
            label, gif_name = next((p.widget(), g) for p, g in self.vfx_proxies if p == self.active_proxy)
            screen_w = self.width()
            screen_h = self.height()
            cfg = self.effects_config["states"].setdefault(self.active_state, {}).setdefault(gif_name, {
                "position_percent": [self.active_proxy.pos().x() / screen_w, self.active_proxy.pos().y() / screen_h],
                "size_percent": [label.width() / screen_w, label.height() / screen_h],
                "rotation": 0,
                "opacity": 0.8
            })
            cfg["rotation"] = value
            self.save_effects_config()

    def select_proxy(self, proxy, rect_item):
        if hasattr(self, 'active_rect_item') and self.active_rect_item:
            self.active_rect_item.hide()

        self.active_proxy = proxy
        self.active_rect_item = rect_item
        rect_item.show()

        # Defensive: only update controls if the widget exists
        widget = proxy.widget()
        if widget:
            self.x_spin.setValue(int(proxy.pos().x()))
            self.y_spin.setValue(int(proxy.pos().y()))
            self.w_spin.setValue(int(widget.width()))
            self.h_spin.setValue(int(widget.height()))
            self.rotation_slider.setValue(int(proxy.rotation()))
            ge = widget.graphicsEffect()
            opacity = ge.opacity() if ge else 1.0
            self.opacity_slider.setValue(int(opacity * 100))
        else:
            print("[WARN] select_proxy: proxy.widget() is None!")


    def snap_value(self, value):
        return self.grid_size * round(value / self.grid_size)
            
    def add_resize_handles(self, proxy):
        grip_size = 12
        corners = [(0, 0), (1, 0), (0, 1), (1, 1)]
        for cx, cy in corners:
            handle = QGraphicsRectItem(0, 0, grip_size, grip_size, parent=proxy)
            handle.setBrush(QColor("cyan"))
            handle.setZValue(2)

            def resize_proxy(event, h=handle, p=proxy, cx=cx, cy=cy):
                self.push_undo(self.capture_state())
                pos = h.pos() + event.pos()
                new_w = self.snap_value(pos.x()) if cx else p.widget().width()
                new_h = self.snap_value(pos.y()) if cy else p.widget().height()
                p.widget().resize(new_w, new_h)
                self.w_spin.setValue(int(new_w))
                self.h_spin.setValue(int(new_h))
                self.save_vfx_state(p, self.get_gif_name_by_proxy(p))
                event.accept()

            handle.mouseMoveEvent = resize_proxy

    def get_gif_name_by_proxy(self, proxy):
        for p, g in self.vfx_proxies:
            if p == proxy:
                return g
        return ""
    
    def wheelEvent(self, event):
        if QApplication.keyboardModifiers() == Qt.AltModifier and self.active_proxy:
            self.push_undo(self.capture_state())
            delta_angle = event.angleDelta().y() / 8
            new_angle = self.active_proxy.rotation() + delta_angle
            self.active_proxy.setRotation(new_angle)
            self.rotation_slider.setValue(int(new_angle))
            self.save_active_proxy_state()
            
    def load_vfx_layers(self, state_name):
        for proxy, _ in self.vfx_proxies:
            self.scene.removeItem(proxy)
        self.vfx_proxies.clear()

        screen_w = self.width()
        screen_h = self.height()

        state_data = self.effects_config.get("states", {}).get(state_name, {})
        for gif_name, cfg in state_data.items():
            label = QLabel()
            label.setAlignment(Qt.AlignCenter)
            label.installEventFilter(self)
            label.setAttribute(Qt.WA_TranslucentBackground)
            label.setStyleSheet("background: transparent")

            movie_path = os.path.join("assets", "vfx", gif_name)
            if not os.path.exists(movie_path):
                continue

            movie = QMovie(movie_path)
            label.setMovie(movie)
            movie.start()

            pos_percent = cfg.get("position_percent", [0.1, 0.1])
            size_percent = cfg.get("size_percent", [0.3, 0.3])
            rotation = cfg.get("rotation", 0)

            x = pos_percent[0] * screen_w
            y = pos_percent[1] * screen_h
            w = size_percent[0] * screen_w
            h = size_percent[1] * screen_h

            label.resize(int(w), int(h))

            proxy = QGraphicsProxyWidget()
            proxy.setWidget(label)  # ✅ CRITICAL!
            proxy.setPos(QPointF(x, y))
            proxy.setRotation(rotation)

            opacity_effect = QGraphicsOpacityEffect()
            opacity_effect.setOpacity(cfg.get("opacity", 0.8))
            label.setGraphicsEffect(opacity_effect)

            proxy.setFlag(QGraphicsItem.ItemIsMovable, True)
            proxy.setFlag(QGraphicsItem.ItemIsSelectable, True)
            proxy.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)

            rect_item = QGraphicsRectItem(proxy.boundingRect(), proxy)
            rect_item.setPen(QPen(QColor("lime"), 2, Qt.DashLine))
            rect_item.hide()

            proxy.mousePressEvent = lambda event, p=proxy, r=rect_item: self.select_proxy(p, r)

            self.scene.addItem(proxy)
            self.vfx_proxies.append((proxy, gif_name))

    def eventFilter(self, obj, event):
        if isinstance(obj, QLabel) and event.type() in (QEvent.MouseButtonPress, QEvent.MouseMove, QEvent.MouseButtonRelease):
            if event.type() == QEvent.MouseButtonPress:
                self.dragging_vfx = True
                self.drag_offset = event.globalPosition().toPoint() - obj.mapToGlobal(obj.rect().topLeft())
                return True

            elif event.type() == QEvent.MouseMove and self.dragging_vfx:
                new_pos = event.globalPosition().toPoint() - self.drag_offset
                scene_pos = self.view.mapToScene(self.view.mapFromGlobal(new_pos))
                proxy = next((p for p, _ in self.vfx_proxies if p.widget() == obj), None)
                if proxy:
                    proxy.setPos(scene_pos)
                    self.x_spin.setValue(int(scene_pos.x()))
                    self.y_spin.setValue(int(scene_pos.y()))
                    self.save_active_proxy_state()
                return True

            elif event.type() == QEvent.MouseButtonRelease:
                self.dragging_vfx = False
                self.save_active_proxy_state()
                return True

        return super().eventFilter(obj, event)
        
    def transform_to(self, mode):
        if mode in self.transform_sets:
            self.current_mode = mode
            gif_path = self.transform_sets[mode]
            if os.path.exists(gif_path):
                self.movie.stop()
                self.movie = QMovie(gif_path)
                self.avatar_label.setMovie(self.movie)
                self.movie.start()

                # Play transform sound
                pygame.mixer.Sound(resource_path("assets/sfx/transform.mp3")).play()

                # Apply transform VFX
                self.apply_vfx(mode)

                # Schedule revert
                QTimer.singleShot(4000, self.end_transform_and_resume_expression)

    def end_transform_and_resume_expression(self):
        self.current_mode = "base"
        self.update_emote(self.current_emotion)
        self.apply_vfx(self.current_emotion)

    def resume_expression_vfx(self):
        # Revert avatar to idle
        idle_gif = GENOS_EMOTE_SETS["default"]["neutral"]
        if os.path.exists(idle_gif):
            self.movie.stop()
            self.movie = QMovie(idle_gif)
            self.avatar_label.setMovie(self.movie)
            self.movie.start()
            self.output_box.append("🔄 Genos has returned to IDLE mode.")

        # Reapply previous emotion overlays
        emotion = getattr(self, 'previous_emotion', 'neutral')
        self.apply_vfx(emotion)
        
    def push_undo(self, state):
        self.undo_stack.append(state.copy())
        self.redo_stack.clear()

    def undo(self):
        if self.undo_stack:
            state = self.undo_stack.pop()
            self.redo_stack.append(self.capture_state())
            self.restore_state(state)

    def redo(self):
        if self.redo_stack:
            state = self.redo_stack.pop()
            self.undo_stack.append(self.capture_state())
            self.restore_state(state)
    
    def update_x(self, value):
        if self.active_proxy:
            self.push_undo(self.capture_state())
            self.active_proxy.setX(value)
            self.save_active_proxy_state()

    def update_y(self, value):
        if self.active_proxy:
            self.push_undo(self.capture_state())
            self.active_proxy.setY(value)
            self.save_active_proxy_state()

    def update_w(self, value):
        if self.active_proxy:
            self.push_undo(self.capture_state())
            self.active_proxy.widget().resize(value, self.active_proxy.widget().height())
            self.save_active_proxy_state()

    def update_h(self, value):
        if self.active_proxy:
            self.push_undo(self.capture_state())
            self.active_proxy.widget().resize(self.active_proxy.widget().width(), value)
            self.save_active_proxy_state()

    def apply_rotation(self, value):
        if self.active_proxy:
            self.push_undo(self.capture_state())
            self.active_proxy.setRotation(value)
            self.save_active_proxy_state()

    def apply_opacity(self, value):
        if self.active_proxy:
            self.push_undo(self.capture_state())
            opacity = value / 100.0
            self.active_proxy.widget().graphicsEffect().setOpacity(opacity)
            self.save_active_proxy_state()

    def move_layer_up(self):
        if self.active_proxy:
            self.push_undo(self.capture_state())
            z = self.active_proxy.zValue()
            self.active_proxy.setZValue(z + 1)

    def move_layer_down(self):
        if self.active_proxy:
            self.push_undo(self.capture_state())
            z = self.active_proxy.zValue()
            self.active_proxy.setZValue(z - 1)

    def push_undo(self, state):
        self.undo_stack.append(state)
        self.redo_stack.clear()

    def undo(self):
        if self.undo_stack:
            prev = self.undo_stack.pop()
            self.redo_stack.append(self.capture_state())
            self.restore_state(prev)

    def redo(self):
        if self.redo_stack:
            next = self.redo_stack.pop()
            self.undo_stack.append(self.capture_state())
            self.restore_state(next)

    def capture_state(self):
        state = []
        for proxy, gif_name in self.vfx_proxies:
            widget = proxy.widget()
            if widget:
                ge = widget.graphicsEffect()
                opacity = ge.opacity() if ge else 1.0
                state.append({
                    "gif": gif_name,
                    "pos": [proxy.pos().x(), proxy.pos().y()],
                    "size": [widget.width(), widget.height()],
                    "rotation": proxy.rotation(),
                    "opacity": opacity
                })
            else:
                print(f"[WARN] capture_state: Skipping proxy for {gif_name} — widget is None.")
        return state

    def restore_state(self, state):
        for item in state:
            for proxy, gif_name in self.vfx_proxies:
                if gif_name == item["gif"]:
                    proxy.setPos(QPointF(item["pos"][0], item["pos"][1]))
                    proxy.widget().resize(item["size"][0], item["size"][1])
                    proxy.setRotation(item["rotation"])
                    proxy.widget().graphicsEffect().setOpacity(item["opacity"])
                    self.save_vfx_state(proxy, gif_name)

    def save_active_proxy_state(self):
        if self.active_proxy:
            for proxy, gif_name in self.vfx_proxies:
                if proxy == self.active_proxy:
                    self.save_vfx_state(proxy, gif_name)

    def save_vfx_state(self, proxy, gif_name):
        screen_w, screen_h = self.width(), self.height()
        widget = proxy.widget()
        ge = widget.graphicsEffect() if widget else None
        opacity = ge.opacity() if ge else 1.0

        cfg = self.effects_config["states"].setdefault(self.current_emotion, {}).setdefault(gif_name, {})
        cfg["position_percent"] = [
            proxy.pos().x() / screen_w,
            proxy.pos().y() / screen_h
        ]
        cfg["size_percent"] = [
            widget.width() / screen_w if widget else 0.1,
            widget.height() / screen_h if widget else 0.1
        ]
        cfg["rotation"] = proxy.rotation()
        cfg["opacity"] = opacity

        self.save_effects_config()




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GenosChat()
    window.show()
    sys.exit(app.exec())


# ✅ Updated version with VFX fixes, drag, resize, rotate, opacity, and percent scaling improvements.