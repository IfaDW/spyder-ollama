# -*- coding: utf-8 -*-

# Copyright © 2024-2026 Datagniel (IfaDW)
# Licensed under the terms of the MIT License

"""Chat panel widget for Ollama assistant."""

# Standard library imports
import logging
import re

# Third party imports
from qtpy.QtCore import QEvent, Qt, Signal
from qtpy.QtGui import QFont, QTextCursor
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
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
from spyder_ollama.client import fetch_ollama_models


logger = logging.getLogger(__name__)


class OllamaChatActions:
    ExplainSelection = "explain_selection"
    GenerateCode = "generate_code"
    ClearChat = "clear_chat"


class OllamaChatWidget(PluginMainWidget):
    """Main widget for the Ollama assistant panel."""

    ENABLE_SPINNER = True

    sig_request_editor_selection = Signal()
    sig_request_code_generation = Signal()
    sig_code_generated = Signal(str)
    sig_model_changed = Signal(str)

    def __init__(self, name, plugin, parent=None):
        super().__init__(name, plugin, parent)

        # Chat client
        self.chat_client = OllamaChatClient(self)
        self.chat_client.sig_token.connect(self._on_token)
        self.chat_client.sig_response_finished.connect(
            self._on_response_finished
        )
        self.chat_client.sig_error.connect(self._on_error)
        self.chat_client.sig_notice.connect(self._append_system)

        self._streaming = False
        self._generating = False
        self._generation_buffer = []

        # --- Model selector row ---
        self.model_combobox = QComboBox(self)
        self.model_combobox.setEditable(False)
        self.model_combobox.currentTextChanged.connect(
            self._on_model_selected
        )

        self.refresh_models_btn = QPushButton(_("Refresh"), self)
        self.refresh_models_btn.clicked.connect(self.populate_models)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel(_("Model:"), self))
        model_row.addWidget(self.model_combobox, stretch=1)
        model_row.addWidget(self.refresh_models_btn)

        model_container = QWidget(self)
        model_container.setLayout(model_row)

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
            _("Ask a question... (Enter to send, Shift+Enter for newline)")
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
        layout.addWidget(model_container)
        layout.addWidget(splitter)
        self.setLayout(layout)

    # ---- PluginMainWidget API -------------------------------------------
    def get_title(self):
        return _("Ollama Assistant")

    def get_focus_widget(self):
        return self.input_edit

    def setup(self):
        """Create actions for the toolbar and menu."""
        explain_action = self.create_action(
            OllamaChatActions.ExplainSelection,
            text=_("Explain Selection"),
            icon=self.create_icon("help"),
            triggered=self._on_explain_triggered,
            register_shortcut=True,
        )

        generate_action = self.create_action(
            OllamaChatActions.GenerateCode,
            text=_("Generate Code from Comment"),
            icon=self.create_icon("run"),
            triggered=self._on_generate_triggered,
            register_shortcut=True,
        )

        clear_action = self.create_action(
            OllamaChatActions.ClearChat,
            text=_("Clear Chat"),
            icon=self.create_icon("editclear"),
            triggered=self.clear_chat,
        )

        toolbar = self.get_main_toolbar()
        for item in [explain_action, generate_action, clear_action]:
            self.add_item_to_toolbar(item, toolbar=toolbar)

        menu = self.get_options_menu()
        for item in [explain_action, generate_action, clear_action]:
            self.add_item_to_menu(item, menu=menu)

    def update_actions(self):
        pass

    # ---- Event filter for Shift+Enter -----------------------------------
    def eventFilter(self, obj, event):
        """Handle Shift+Enter in the input field."""
        if obj is self.input_edit and event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                if event.modifiers() & Qt.ShiftModifier:
                    # Shift+Enter: let Qt insert a newline
                    return False
                self._on_send_clicked()
                return True
        return super().eventFilter(obj, event)

    # ---- Public API -----------------------------------------------------
    def update_client_config(
        self, model_name: str, base_url: str, temperature: float
    ):
        """Update the chat client configuration and the model selector."""
        self.chat_client.update_config(model_name, base_url, temperature)
        self.populate_models(preselect=model_name)

    def populate_models(self, preselect: str = ""):
        """Fill the model selector from the Ollama server."""
        target = preselect or self.model_combobox.currentText()
        models = fetch_ollama_models(self.chat_client.base_url)

        self.model_combobox.blockSignals(True)
        self.model_combobox.clear()
        if models:
            self.model_combobox.addItems(sorted(models))
            if target and target in models:
                self.model_combobox.setCurrentText(target)
        self.model_combobox.blockSignals(False)

        # Sync client with whatever is now selected
        current = self.model_combobox.currentText()
        if current:
            self.chat_client.model_name = current

    def explain_code(self, code: str):
        """Explain the given code snippet."""
        if not code.strip():
            self._append_system("No code selected.")
            return

        self._append_user(f"Explain this code:\n```python\n{code}\n```")
        self._start_response()
        self.chat_client.explain_code(code)

    def generate_from_comment(self, instruction: str, code_context: str = ""):
        """Generate code from a comment and stream it into the panel.

        On finish, sig_code_generated is emitted with the cleaned code
        so the plugin can insert it into the editor.
        """
        if not instruction.strip():
            self._append_system("No comment found at the cursor.")
            return

        self._append_user(f"Generate code for:\n{instruction}")
        self._generating = True
        self._generation_buffer = []
        self._start_response()
        self.chat_client.generate_code(instruction, code_context)

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
    def _on_model_selected(self, model_name: str):
        """User picked a model in the selector."""
        if not model_name:
            return
        self.chat_client.model_name = model_name
        self.sig_model_changed.emit(model_name)

    def _on_explain_triggered(self):
        self.sig_request_editor_selection.emit()

    def _on_generate_triggered(self):
        self.sig_request_code_generation.emit()

    def _on_send_clicked(self):
        text = self.input_edit.toPlainText().strip()
        if not text or self._streaming:
            return
        self.input_edit.clear()
        self.ask_with_context(text)

    def _on_stop_clicked(self):
        self.chat_client.stop()
        self._on_response_finished()

    def _on_token(self, token: str):
        if self._generating:
            self._generation_buffer.append(token)
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(token)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    def _on_response_finished(self):
        self._streaming = False
        self.send_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._append_raw("\n\n")

        if self._generating:
            self._generating = False
            code = self._clean_generated_code(
                "".join(self._generation_buffer)
            )
            self._generation_buffer = []
            if not code.strip():
                return
            try:
                compile(code, "<generated>", "exec")
            except SyntaxError as e:
                self._append_system(
                    "Generated code failed the syntax check "
                    f"(line {e.lineno}: {e.msg}) — NOT inserted into "
                    "the editor. Try again, rephrase the comment, or "
                    "switch to a code model (e.g. qwen2.5-coder)."
                )
                return
            self.sig_code_generated.emit(code)

    @staticmethod
    def _clean_generated_code(text: str) -> str:
        """Extract code from the model response.

        Small models often wrap code in markdown fences and surround
        it with prose despite instructions. If fenced blocks exist
        anywhere in the response, only their contents are kept;
        otherwise the whole text is used with stray fences removed.
        """
        text = text.strip()
        blocks = re.findall(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", text, re.DOTALL)
        if blocks:
            return "\n\n".join(b.strip() for b in blocks).strip()
        lines = [
            ln for ln in text.split("\n")
            if not ln.strip().startswith("```")
        ]
        return "\n".join(lines).strip()

    def _on_error(self, error_msg: str):
        self._append_system(f"Error: {error_msg}")

    # ---- Formatting helpers ---------------------------------------------
    def _start_response(self):
        self._streaming = True
        self.send_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._append_raw("🤖 ")

    def _append_user(self, text: str):
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(f"\n👤 {text}\n\n")
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    def _append_system(self, text: str):
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(f"\n⚙️ {text}\n\n")
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    def _append_raw(self, text: str):
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.chat_display.setTextCursor(cursor)
