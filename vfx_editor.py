import os, json, shutil
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QFileDialog, QSlider,
    QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QGraphicsView, QGraphicsScene,
    QGraphicsProxyWidget, QGraphicsRectItem, QComboBox, QGraphicsItem
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QMovie, QColor, QPen, QKeyEvent, QWheelEvent

ASSETS_DIR = "assets/vfx"
EXPRESSIONS_DIR = "assets/default"
CONFIG_PATH = "effects_config.json"
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

class VFXEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Genos VFX Editor")
        self.resize(SCREEN_WIDTH, SCREEN_HEIGHT + 150)

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.setCentralWidget(QWidget())
        layout = QVBoxLayout(self.centralWidget())

        self.state_selector = QComboBox()
        self.state_selector.addItems(["neutral", "angry", "happy", "vengeful", "defensive", "goofy", "blush", "combat", "base"])
        self.state_selector.currentTextChanged.connect(self.load_state)

        self.add_button = QPushButton("Add VFX(s)")
        self.add_button.clicked.connect(self.add_vfx)

        self.delete_button = QPushButton("Delete Selected")
        self.delete_button.clicked.connect(self.delete_selected)

        self.save_button = QPushButton("Save Config")
        self.save_button.clicked.connect(self.save_config)

        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(80)
        self.opacity_slider.valueChanged.connect(self.update_opacity)

        self.rotation_slider = QSlider(Qt.Horizontal)
        self.rotation_slider.setRange(-180, 180)
        self.rotation_slider.setValue(0)
        self.rotation_slider.valueChanged.connect(self.update_rotation)

        top = QHBoxLayout()
        top.addWidget(self.state_selector)
        top.addWidget(self.add_button)
        top.addWidget(self.delete_button)
        top.addWidget(self.save_button)
        layout.addLayout(top)
        layout.addWidget(self.view)

        sliders = QHBoxLayout()
        sliders.addWidget(QLabel("Opacity"))
        sliders.addWidget(self.opacity_slider)
        sliders.addWidget(QLabel("Rotation"))
        sliders.addWidget(self.rotation_slider)
        layout.addLayout(sliders)

        self.current_state = "neutral"
        self.config = self.load_config()
        self.vfx_proxies = []
        self.active_proxy = None
        self.highlight_rect = None

        self.load_state("neutral")

    def load_config(self):
        return json.load(open(CONFIG_PATH)) if os.path.exists(CONFIG_PATH) else {"states": {}}

    def save_config(self):
        out = {}
        for proxy, name in self.vfx_proxies:
            w = proxy.widget()
            out[name] = {
                "position_percent": [proxy.pos().x()/SCREEN_WIDTH, proxy.pos().y()/SCREEN_HEIGHT],
                "size_percent": [w.width()/SCREEN_WIDTH, w.height()/SCREEN_HEIGHT],
                "rotation": proxy.rotation(),
                "opacity": proxy.opacity()
            }
        self.config["states"][self.current_state] = out
        with open(CONFIG_PATH, "w") as f:
            json.dump(self.config, f, indent=4)
        print("✔ Saved config")

    def load_state(self, state):
        self.scene.clear()

        # ✅ Recreate background_label after scene.clear()
        self.background_label = QLabel()
        self.background_label.setScaledContents(True)
        self.background_label.setGeometry(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
        self.scene.addWidget(self.background_label)

        self.vfx_proxies.clear()
        self.active_proxy = None
        self.current_state = state

        # Load background expression GIF
        bg_path = os.path.join(EXPRESSIONS_DIR, f"{state}.gif")
        if os.path.exists(bg_path):
            bg_movie = QMovie(bg_path)
            bg_movie.setCacheMode(QMovie.CacheAll)
            bg_movie.start()
            self.background_label.setMovie(bg_movie)

        # Load overlays
        state_data = self.config.get("states", {}).get(state, {})
        for name, cfg in state_data.items():
            self.spawn_vfx(name, cfg)

    def spawn_vfx(self, gif, cfg):
        path = os.path.join(ASSETS_DIR, gif)
        if not os.path.exists(path): return
        label = QLabel()
        label.setStyleSheet("background: transparent")
        label.setScaledContents(True)
        movie = QMovie(path)
        movie.setCacheMode(QMovie.CacheAll)
        movie.start()
        label.setMovie(movie)
        w, h = cfg.get("size_percent", [0.2, 0.2])
        label.resize(w*SCREEN_WIDTH, h*SCREEN_HEIGHT)

        proxy = QGraphicsProxyWidget()
        proxy.setWidget(label)
        proxy.setPos(cfg.get("position_percent", [0.1, 0.1])[0]*SCREEN_WIDTH,
                     cfg.get("position_percent", [0.1, 0.1])[1]*SCREEN_HEIGHT)
        proxy.setRotation(cfg.get("rotation", 0))
        proxy.setOpacity(cfg.get("opacity", 0.8))
        proxy.setFlag(QGraphicsItem.ItemIsMovable, True)
        proxy.setFlag(QGraphicsItem.ItemIsSelectable, True)
        proxy.setAcceptedMouseButtons(Qt.LeftButton)
        proxy.setAcceptHoverEvents(True)

        proxy.mousePressEvent = lambda e, p=proxy: self.select_proxy(p)
        proxy.wheelEvent = self.scroll_resize
        proxy.mouseMoveEvent = lambda e, p=proxy: self.drag_with_snap(e, p)

        self.scene.addItem(proxy)
        self.vfx_proxies.append((proxy, gif))

    def add_vfx(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select GIFs", ASSETS_DIR, "GIF Files (*.gif)")
        for path in paths:
            gif_name = os.path.basename(path)
            dest_path = os.path.join(ASSETS_DIR, gif_name)
            if not os.path.exists(dest_path):
                os.makedirs(ASSETS_DIR, exist_ok=True)
                shutil.copy(path, dest_path)
            cfg = {
                "position_percent": [0.1, 0.1],
                "size_percent": [0.3, 0.3],
                "rotation": 0,
                "opacity": 0.8
            }
            self.spawn_vfx(gif_name, cfg)

    def select_proxy(self, proxy):
        self.active_proxy = proxy
        self.opacity_slider.setValue(int(proxy.opacity()*100))
        self.rotation_slider.setValue(int(proxy.rotation()))
        if self.highlight_rect:
            self.scene.removeItem(self.highlight_rect)
        self.highlight_rect = QGraphicsRectItem(proxy.boundingRect(), proxy)
        self.highlight_rect.setPen(QPen(QColor("cyan"), 2, Qt.DashLine))

    def delete_selected(self):
        if self.active_proxy:
            for i, (p, name) in enumerate(self.vfx_proxies):
                if p == self.active_proxy:
                    self.scene.removeItem(p)
                    del self.vfx_proxies[i]
                    self.active_proxy = None
                    break
            if self.highlight_rect:
                self.scene.removeItem(self.highlight_rect)
                self.highlight_rect = None

    def keyPressEvent(self, e: QKeyEvent):
        if e.key() == Qt.Key_Delete:
            self.delete_selected()

    def update_opacity(self, val):
        if self.active_proxy:
            self.active_proxy.setOpacity(val / 100)

    def update_rotation(self, val):
        if self.active_proxy:
            self.active_proxy.setRotation(val)

    def scroll_resize(self, e: QWheelEvent):
        if not self.active_proxy: return
        delta = e.angleDelta().y() / 120
        scale = 1 + (delta * 0.05)
        w = self.active_proxy.widget().width() * scale
        h = self.active_proxy.widget().height() * scale
        self.active_proxy.widget().resize(w, h)
        self.active_proxy.update()

    def drag_with_snap(self, e, proxy):
        new_pos = proxy.pos() + e.pos() - e.lastPos()
        if QApplication.keyboardModifiers() & Qt.ControlModifier:
            new_pos.setX(round(new_pos.x() / 10) * 10)
            new_pos.setY(round(new_pos.y() / 10) * 10)
        proxy.setPos(new_pos)
        if self.highlight_rect:
            self.highlight_rect.setRect(proxy.boundingRect())

if __name__ == "__main__":
    app = QApplication([])
    editor = VFXEditor()
    editor.show()
    app.exec()