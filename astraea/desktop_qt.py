"""Astraea's native Qt desktop workbench."""

import asyncio
import os
import sys
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFont, QKeyEvent
from PySide6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QFormLayout, QFrame, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QPushButton, QScrollArea, QSizePolicy,
    QTextEdit, QToolButton, QVBoxLayout, QWidget,
)
from langchain_core.messages import SystemMessage

from astraea.agent import build_model
from astraea.config import PROJECT_DIR, Settings
from astraea.context import ContextManager, PROFILE_PREFIX, estimate_tokens, usage_tokens
from astraea.memory import MemoryManager, read_profile_markdown
from astraea.mcp_tools import invoke_with_mcp


STYLE = """
QWidget { background: #0C111B; color: #DDE5F1; font-family: 'Segoe UI'; font-size: 13px; }
QLabel { background: transparent; }
QMainWindow { background: #0C111B; }
QFrame#sidebar { background: #111A28; border-right: 1px solid #263345; }
QLabel#brand { color: #F2E5C9; font-size: 23px; font-weight: 700; }
QLabel#muted { color: #8C9BAF; }
QLabel#section { color: #8595A9; font-size: 11px; font-weight: 700; letter-spacing: 2px; }
QLabel#title { color: #F1F4FA; font-size: 18px; font-weight: 650; }
QLabel#hero { color: #EEE5D3; font-size: 31px; font-weight: 650; }
QPushButton { background: transparent; border: 1px solid #35445A; border-radius: 10px; padding: 10px 14px; text-align: left; }
QPushButton:hover { background: #202D40; border-color: #546781; }
QPushButton#primary { background: #DDBD83; color: #17202D; border: 0; font-weight: 700; text-align: center; }
QPushButton#primary:hover { background: #EBCF9B; }
QPushButton:disabled { color: #6C7C92; background: #1B2635; }
QLineEdit, QTextEdit { background: #172234; border: 1px solid #34445A; border-radius: 10px; padding: 9px; selection-background-color: #796B54; }
QLineEdit:focus, QTextEdit:focus { border-color: #DDBD83; }
QFrame#composer { background: #172234; border: 1px solid #38485D; border-radius: 14px; }
QTextEdit#input { background: transparent; border: 0; padding: 10px 12px; }
QFrame#assistantBubble { background: #152132; border: 1px solid #29394D; border-radius: 13px; }
QFrame#userBubble { background: #24334A; border: 1px solid #40516A; border-radius: 13px; }
QLabel#role { color: #D9BB84; font-size: 11px; font-weight: 700; letter-spacing: 1px; }
QLabel#body { color: #E3EAF4; line-height: 1.5; }
QScrollArea { border: 0; background: transparent; }
QScrollBar:vertical { background: #101827; width: 8px; }
QScrollBar::handle:vertical { background: #3A4B62; border-radius: 4px; }
QDialog { background: #101827; }
QToolButton { color: #B8C5D8; background: #121D2D; border: 1px solid #34455A; border-radius: 8px; padding: 9px; text-align: left; }
"""


class Composer(QTextEdit):
    submit = Signal()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not (event.modifiers() & Qt.ShiftModifier):
            self.submit.emit()
            return
        super().keyPressEvent(event)


class SettingsDialog(QDialog):
    def __init__(self, parent=None, current: Settings | None = None):
        super().__init__(parent)
        self.setWindowTitle("Astraea 设置")
        self.setMinimumWidth(520)
        self.settings = None
        self.current = current
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 26, 28, 26)
        layout.setSpacing(17)
        heading = QLabel("连接模型")
        heading.setObjectName("title")
        layout.addWidget(heading)
        hint = QLabel("配置只在本次运行中使用。API Key 不写入项目文件。")
        hint.setObjectName("muted")
        layout.addWidget(hint)
        form = QFormLayout()
        form.setSpacing(13)
        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.Password)
        self.key.setPlaceholderText("输入 API Key")
        self.url = QLineEdit(current.base_url if current else "https://api.deepseek.com")
        self.model = QLineEdit(current.model if current else "deepseek-chat")
        self.compactor = QLineEdit(current.compactor_model if current else "")
        self.compactor.setPlaceholderText("留空则使用同一模型")
        self.window = QLineEdit(str(current.context_window if current else 64000))
        self.workspace = QLineEdit(str(current.workspace if current else PROJECT_DIR / "workspace"))
        browse = QPushButton("浏览…")
        browse.clicked.connect(self._browse)
        workspace_row = QWidget()
        row = QHBoxLayout(workspace_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.workspace, 1)
        row.addWidget(browse)
        for label, field in (("API Key", self.key), ("API 地址", self.url), ("模型", self.model),
                             ("摘要模型", self.compactor), ("上下文窗口", self.window), ("工作区", workspace_row)):
            form.addRow(label, field)
        layout.addLayout(form)
        self.error = QLabel("")
        self.error.setStyleSheet("color:#EE9E9E")
        layout.addWidget(self.error)
        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        save = QPushButton("应用设置")
        save.setObjectName("primary")
        save.clicked.connect(self._apply)
        buttons.addWidget(cancel)
        buttons.addWidget(save)
        layout.addLayout(buttons)

    def _browse(self):
        chosen = QFileDialog.getExistingDirectory(self, "选择工作区", self.workspace.text())
        if chosen:
            self.workspace.setText(chosen)

    def _apply(self):
        try:
            key = self.key.text().strip() or (self.current.api_key if self.current else "")
            url = self.url.text().strip()
            model = self.model.text().strip()
            window = int(self.window.text())
            if not key or not url.startswith(("http://", "https://")) or not model:
                raise ValueError("请填写 API Key、HTTP(S) 地址和模型名称。")
            if window < 1000:
                raise ValueError("上下文窗口至少为 1000。")
            workspace = Path(self.workspace.text()).expanduser().resolve()
            workspace.mkdir(parents=True, exist_ok=True)
            self.settings = Settings(key, url, model, window, self.compactor.text().strip() or model, workspace)
            self.accept()
        except (OSError, ValueError) as exc:
            self.error.setText(str(exc))


class TurnWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, model, settings, context, message):
        super().__init__()
        self.model = model
        self.settings = settings
        self.context = context
        self.message = message

    def run(self):
        try:
            request = self.context.prepare(self.message, read_profile_markdown())
            result = asyncio.run(invoke_with_mcp(self.model, self.settings, request))
            fresh = result["messages"][len(request):]
            self.context.record(result["messages"])
            try:
                MemoryManager(self.model).process_turn(self.context.session_id, self.context.history)
            except Exception as exc:
                print(f"[Memory] Could not process turn: {type(exc).__name__}: {exc}")
            events = []
            for item in fresh:
                for call in getattr(item, "tool_calls", []) or []:
                    events.append(f"调用 {call['name']}  {call.get('args', {})}")
                if item.type == "tool":
                    events.append(f"结果 {item.name or 'tool'}  {str(item.content)[:4000]}")
            self.finished.emit((str(result["messages"][-1].content), events))
        except Exception as exc:
            self.failed.emit(str(exc))


class AstraeaWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Astraea Agent")
        self.resize(1120, 760)
        self.setMinimumSize(760, 560)
        self.settings = None
        self.agent = None
        self.context = None
        self.thread = None
        self._build_ui()
        QTimer.singleShot(100, self.open_settings)

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(230)
        rail = QVBoxLayout(sidebar)
        rail.setContentsMargins(20, 25, 20, 20)
        rail.setSpacing(12)
        brand = QLabel("✦  ASTRAEA")
        brand.setObjectName("brand")
        rail.addWidget(brand)
        subtitle = QLabel("AGENT WORKBENCH")
        subtitle.setObjectName("section")
        rail.addWidget(subtitle)
        rail.addSpacing(24)
        new_chat = QPushButton("＋  新对话")
        new_chat.clicked.connect(self.new_chat)
        rail.addWidget(new_chat)
        rail.addSpacing(24)
        section = QLabel("当前工作区")
        section.setObjectName("section")
        rail.addWidget(section)
        self.workspace_label = QLabel("尚未配置")
        self.workspace_label.setObjectName("muted")
        self.workspace_label.setWordWrap(True)
        rail.addWidget(self.workspace_label)
        rail.addStretch()
        self.model_label = QLabel("未连接模型")
        self.model_label.setObjectName("muted")
        rail.addWidget(self.model_label)
        settings_button = QPushButton("⚙  设置")
        settings_button.clicked.connect(self.open_settings)
        rail.addWidget(settings_button)
        outer.addWidget(sidebar)

        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        top = QFrame()
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(30, 19, 30, 18)
        title = QLabel("对话")
        title.setObjectName("title")
        top_layout.addWidget(title)
        top_layout.addStretch()
        self.usage = QLabel("T0  ·  上下文 0%")
        self.usage.setObjectName("muted")
        top_layout.addWidget(self.usage)
        main_layout.addWidget(top)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.timeline = QWidget()
        self.timeline_layout = QVBoxLayout(self.timeline)
        self.timeline_layout.setContentsMargins(35, 28, 35, 24)
        self.timeline_layout.setSpacing(17)
        self.timeline_layout.addStretch()
        self.scroll.setWidget(self.timeline)
        main_layout.addWidget(self.scroll, 1)
        self._welcome()

        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(30, 12, 30, 24)
        bottom_layout.setSpacing(7)
        composer_frame = QFrame()
        composer_frame.setObjectName("composer")
        composer = QHBoxLayout(composer_frame)
        composer.setContentsMargins(7, 6, 7, 6)
        self.input = Composer()
        self.input.setObjectName("input")
        self.input.setPlaceholderText("给 Astraea 一个任务…")
        self.input.setFixedHeight(68)
        self.input.submit.connect(self.send)
        composer.addWidget(self.input, 1)
        self.send_button = QPushButton("发送 ↗")
        self.send_button.setObjectName("primary")
        self.send_button.setFixedWidth(86)
        self.send_button.clicked.connect(self.send)
        composer.addWidget(self.send_button, alignment=Qt.AlignBottom)
        bottom_layout.addWidget(composer_frame)
        self.hint = QLabel("Enter 发送  ·  Shift + Enter 换行  ·  工具操作仅作用于当前工作区")
        self.hint.setObjectName("muted")
        bottom_layout.addWidget(self.hint)
        main_layout.addWidget(bottom)
        outer.addWidget(main, 1)

    def _welcome(self):
        self.welcome = QWidget()
        box = QVBoxLayout(self.welcome)
        box.setAlignment(Qt.AlignCenter)
        star = QLabel("✦")
        star.setStyleSheet("color:#DDBD83;font-size:42px")
        box.addWidget(star, alignment=Qt.AlignCenter)
        hero = QLabel("从一个任务开始")
        hero.setObjectName("hero")
        box.addWidget(hero, alignment=Qt.AlignCenter)
        note = QLabel("让 Astraea 查找、阅读、修改文件，并运行测试。")
        note.setObjectName("muted")
        box.addWidget(note, alignment=Qt.AlignCenter)
        self.timeline_layout.insertWidget(0, self.welcome, 1)

    def open_settings(self):
        if self.thread is not None:
            return
        dialog = SettingsDialog(self, self.settings)
        if dialog.exec() != QDialog.Accepted:
            return
        self.settings = dialog.settings
        os.environ["ASTRAEA_WORKSPACE"] = str(self.settings.workspace)
        model = build_model(self.settings)
        self.agent = model
        compactor = build_model(self.settings, self.settings.compactor_model)
        self.context = ContextManager(compactor, self.settings.context_window)
        self.workspace_label.setText(str(self.settings.workspace))
        self.model_label.setText(self.settings.model)
        self.new_chat()
        self.input.setFocus()

    def new_chat(self):
        if self.thread is not None:
            return
        if self.settings is not None:
            self.context = ContextManager(build_model(self.settings, self.settings.compactor_model), self.settings.context_window)
        while self.timeline_layout.count() > 1:
            item = self.timeline_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._welcome()
        self.usage.setText("T0  ·  上下文 0%")

    def send(self):
        message = self.input.toPlainText().strip()
        if not message or self.agent is None or self.thread is not None:
            if self.agent is None:
                self.open_settings()
            return
        self.input.clear()
        if self.welcome:
            self.welcome.hide()
            self.welcome = None
        self._bubble("你", message, user=True)
        self.hint.setText("Astraea 正在处理…")
        self.send_button.setEnabled(False)
        self.thread = QThread()
        self.worker = TurnWorker(self.agent, self.settings, self.context, message)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._finished)
        self.worker.failed.connect(self._failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self._thread_done)
        self.thread.start()

    def _finished(self, result):
        answer, events = result
        if events:
            self._tool_details(events)
        self._bubble("ASTRAEA", answer)
        profile = read_profile_markdown()
        profile_tokens = estimate_tokens([SystemMessage(content=PROFILE_PREFIX + profile)]) if profile else 0
        percent = 100 * (usage_tokens(self.context.active) + profile_tokens) / self.settings.context_window
        self.usage.setText(f"T{self.context.tier}  ·  上下文 {percent:.1f}%")
        self.hint.setText("Enter 发送  ·  Shift + Enter 换行  ·  工具操作仅作用于当前工作区")

    def _failed(self, error):
        self._bubble("错误", error)
        self.hint.setText("请求失败，请检查设置或重试。")

    def _thread_done(self):
        self.worker.deleteLater()
        self.thread.deleteLater()
        self.thread = None
        self.send_button.setEnabled(True)
        self.input.setFocus()

    def _bubble(self, role, text, user=False):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        bubble = QFrame()
        bubble.setObjectName("userBubble" if user else "assistantBubble")
        bubble.setMaximumWidth(660)
        content = QVBoxLayout(bubble)
        content.setContentsMargins(16, 12, 16, 14)
        content.setSpacing(7)
        name = QLabel(role)
        name.setObjectName("role")
        content.addWidget(name)
        body = QLabel(text)
        body.setObjectName("body")
        body.setWordWrap(True)
        body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        content.addWidget(body)
        if user:
            layout.addStretch()
        layout.addWidget(bubble)
        if not user:
            layout.addStretch()
        self.timeline_layout.insertWidget(self.timeline_layout.count() - 1, row)
        QTimer.singleShot(0, self._scroll_bottom)

    def _tool_details(self, events):
        toggle = QToolButton()
        toggle.setText(f"▸  工具过程  ·  {len(events)} 条记录")
        toggle.setCheckable(True)
        details = QLabel("\n\n".join(events))
        details.setWordWrap(True)
        details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        details.setStyleSheet("color:#AAB9CC; padding:12px 15px; background:#111B2A; border-radius:8px")
        details.setVisible(False)
        toggle.toggled.connect(details.setVisible)
        container = QWidget()
        box = QVBoxLayout(container)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(toggle)
        box.addWidget(details)
        self.timeline_layout.insertWidget(self.timeline_layout.count() - 1, container)

    def _scroll_bottom(self):
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    app.setFont(QFont("Segoe UI", 10))
    window = AstraeaWindow()
    window.show()
    sys.exit(app.exec())
