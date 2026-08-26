# -*- coding: utf-8 -*-

# Copyright © 2024-2026 Datagniel (IfaDW)
# Licensed under the terms of the MIT License

"""Chat panel widget for Ollama assistant."""

# Standard library imports
import logging

# Third party imports
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QFont, QTextCursor
from qtpy.QtWidgets import (
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Spyder imports
from spyder.api.translations import _
from spyder.api.widgets.main_widget import PluginMainWidget

# Local imports
from spyder_ollama.chat_client import OllamaChatClient


logger = logging.getLogger(__name__)


class OllamaChatActions:
    ExplainSelection = "explain_selection"
    AskQuestion = "ask_question"
    ClearChat = "clear_chat"
    StopGeneration = "stop_generation"


class OllamaChatWidget(PluginMainWidget):
    """Main widget for the Ollama assistant panel."""

    ENABLE_SPINNER = True

    # Signal to request selected text from editor
    sig_request_editor_selection = Signal()

    def __init__(self, name, plugin, parent=None):
        super().__init__(name, plugin, parent)

        # Chat client
        self.chat_client = OllamaChatClient(self)
        self.chat_client.sig_token.connect(self._on_token)
        self.chat_client.sig_response_finished.connect(
            self._on_response_finished
        )
        self.chat_client.sig_error.connect(self._on_error)

        self._streaming = False

        # --- Conversation display ---
        self.chat_display = QTextEdit(self)
        self.chat_display.setReadOnly(True)
        self.chat_display.setAcceptRichText(True)
        font = QFont("Monospace", 10)
        font.setStyleHint(QFont.TypeWriter)
        self.chat_display.setFont(font)
        self.chat_display.setPlaceholderText(
            _("Ollama assistant — select code and click 'Explain', "
              "or type a question below.")
        )

        # --- Input area ---
        self.input_edit = QPlainTextEdit(self)
        self.input_edit.setMaximumHeight(100)
        self.input_edit.setPlaceholderText(
            _("Ask a question... (Shift+Enter to send)")
        )
        self.input_edit.installEventFilter(self)

        # --- Buttons ---
        self.send_btn = QPushButton(_("Send"))
        self.send_btn.clicked.connect(self._on_send_clicked)

        self.stop_btn = QPushButton(_("Stop"))
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        self.stop_btn.setEnabled(False)

        btn_layout = QVBoxLayout()
        btn_layout.addWidget(self.send_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addStretch()

        input_row = QHBoxLayout()
        input_row.addWidget(self.input_edit, stretch=1)
        input_row.addLayout(btn_layout)

        input_container = QWidget(self)
        input_container.setLayout(input_row)

        # --- Main layout ---
        splitter = QSplitter(Qt.Vertical, self)
        splitter.addWidget(self.chat_display)
        splitter.addWidget(input_container)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)
        self.setLayout(layout)

    # ---- PluginMainWidget API -------------------------------------------
    def get_title(self):
        return _("Ollama Assistant")

    def get_focus_widget(self):
        return self.input_edit

    def setup(self):
        """Create actions for the toolbar and menu."""
        # Explain selection action
        explain_action = self.create_action(
            OllamaChatActions.ExplainSelection,
            text=_("Explain Selection"),
            icon=self.create_icon("help"),
            triggered=self._on_explain_triggered,
            register_shortcut=True,
        )

        # Clear chat action
        clear_action = self.create_action(
            OllamaChatActions.ClearChat,
            text=_("Clear Chat"),
            icon=self.create_icon("editclear"),
            triggered=self.clear_chat,
        )

        # Add to toolbar
        toolbar = self.get_main_toolbar()
        for item in [explain_action, clear_action]:
            self.add_item_to_toolbar(item, toolbar=toolbar)

        # Add to options menu
        menu = self.get_options_menu()
        for item in [explain_action, clear_action]:
            self.add_item_to_menu(item, menu=menu)

    def update_actions(self):
        pass

    # ---- Event filter for Shift+Enter -----------------------------------
    def eventFilter(self, obj, event):
        """Handle Shift+Enter in the input field."""
        if obj is self.input_edit:
            from qtpy.QtCore import QEvent
            from qtpy.QtGui import QKeyEvent

            if event.type() == QEvent.KeyPress:
                if (
                    event.key() in (Qt.Key_Return, Qt.Key_Enter)
                    and event.modifiers() & Qt.ShiftModifier
                ):
                    self._on_send_clicked()
                    return True
        return super().eventFilter(obj, event)

    # ---- Public API -----------------------------------------------------
    def update_client_config(
        self, model_name: str, base_url: str, temperature: float
    ):
        """Update the chat client configuration."""
        self.chat_client.update_config(model_name, base_url, temperature)

    def explain_code(self, code: str):
        """Explain the given code snippet."""
        if not code.strip():
            self._append_system("No code selected.")
            return

        self._append_user(f"Explain this code:\n```python\n{code}\n```")
        self._start_response()
        self.chat_client.explain_code(code)

    def ask_with_context(self, question: str, code_context: str = ""):
        """Ask a question, optionally with code context."""
        if not question.strip():
            return

        self._append_user(question)
        self._start_response()
        self.chat_client.ask(question, code_context)

    def clear_chat(self):
        """Clear the conversation display."""
        self.chat_display.clear()

    # ---- Private slots --------------------------------------------------
    def _on_explain_triggered(self):
        """Request selected text from the editor plugin."""
        self.sig_request_editor_selection.emit()

    def _on_send_clicked(self):
        """Send the current input text."""
        text = self.input_edit.toPlainText().strip()
        if not text or self._streaming:
            return
        self.input_edit.clear()
        self.ask_with_context(text)

    def _on_stop_clicked(self):
        """Stop the current generation."""
        self.chat_client.stop()
        self._on_response_finished()

    def _on_token(self, token: str):
        """Append a streamed token to the display."""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(token)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    def _on_response_finished(self):
        """Handle end of streaming response."""
        self._streaming = False
        self.send_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._append_raw("\n\n")

    def _on_error(self, error_msg: str):
        """Display an error in the chat."""
        self._append_system(f"Error: {error_msg}")

    # ---- Formatting helpers ---------------------------------------------
    def _start_response(self):
        """Prepare UI for incoming response."""
        self._streaming = True
        self.send_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._append_raw("🤖 ")

    def _append_user(self, text: str):
        """Append a user message to the display."""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(f"\n👤 {text}\n\n")
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    def _append_system(self, text: str):
        """Append a system message to the display."""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(f"\n⚙️ {text}\n\n")
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    def _append_raw(self, text: str):
        """Append raw text to the display."""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.chat_display.setTextCursor(cursor)
