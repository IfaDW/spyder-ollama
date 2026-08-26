# -*- coding: utf-8 -*-

# Copyright © 2024-2026 Datagniel (IfaDW)
# Licensed under the terms of the MIT License

"""Spyder dockable plugin for Ollama assistant panel."""

# Standard library imports
import logging

# Spyder imports
from spyder.api.plugins import Plugins, SpyderDockablePlugin
from spyder.api.plugin_registration.decorators import (
    on_plugin_available,
    on_plugin_teardown,
)
from spyder.api.translations import _

# Local imports
from spyder_ollama.widgets.chat_panel import OllamaChatWidget


logger = logging.getLogger(__name__)


class OllamaChatPlugin(SpyderDockablePlugin):
    """Dockable Ollama assistant panel for Spyder."""

    NAME = "ollama_chat"
    REQUIRES = []
    OPTIONAL = [Plugins.Editor, Plugins.Preferences]
    TABIFY = [Plugins.Help]
    WIDGET_CLASS = OllamaChatWidget
    CONF_SECTION = NAME
    CONF_DEFAULTS = [
        (
            NAME,
            {
                "model_name": "",
                "base_url": "http://localhost:11434",
                "temperature": 0.3,
            },
        )
    ]
    CONF_VERSION = "1.1.0"

    # ---- SpyderDockablePlugin API ---------------------------------------
    @staticmethod
    def get_name():
        return _("Ollama Assistant")

    @staticmethod
    def get_description():
        return _("Chat with a local Ollama model to explain code, "
                 "answer questions, and get coding help.")

    @classmethod
    def get_icon(cls):
        return cls.create_icon("help")

    def on_initialize(self):
        """Set up the plugin."""
        widget = self.get_widget()
        widget.sig_request_editor_selection.connect(
            self._explain_editor_selection
        )
        widget.sig_request_code_generation.connect(
            self._generate_from_editor_comment
        )
        widget.sig_code_generated.connect(self._insert_generated_code)
        widget.sig_model_changed.connect(self._persist_model)

        # Apply saved config; empty model_name means: pick the first
        # model the server offers (resolved in the widget/client).
        widget.update_client_config(
            model_name=self.get_conf("model_name"),
            base_url=self.get_conf("base_url"),
            temperature=self.get_conf("temperature"),
        )

    @on_plugin_available(plugin=Plugins.Editor)
    def on_editor_available(self):
        logger.info("Ollama chat: Editor plugin connected.")

    @on_plugin_teardown(plugin=Plugins.Editor)
    def on_editor_teardown(self):
        logger.info("Ollama chat: Editor plugin disconnected.")

    # ---- Private methods ------------------------------------------------
    def _persist_model(self, model_name: str):
        """Save the model the user picked in the panel selector."""
        self.set_conf("model_name", model_name)

    def _generate_from_editor_comment(self):
        """Read the comment at the cursor (or selection) and generate code."""
        self._insert_cursor = None
        try:
            editor_plugin = self.get_plugin(Plugins.Editor)
            editor = (
                editor_plugin.get_current_editor()
                if editor_plugin else None
            )
            if editor is None:
                self.get_widget()._append_system("No active editor.")
                return

            instruction = editor.get_selected_text()
            cursor = editor.textCursor()

            if not instruction:
                # Use the current line; walk upward to collect a
                # contiguous comment block.
                doc = editor.document()
                block = cursor.block()
                comment_lines = []
                b = block
                while b.isValid() and b.text().lstrip().startswith("#"):
                    comment_lines.insert(
                        0, b.text().lstrip().lstrip("#").strip()
                    )
                    b = b.previous()
                instruction = "\n".join(comment_lines)

            if not instruction.strip():
                self.get_widget()._append_system(
                    "Place the cursor on a comment line "
                    "(or select a comment) first."
                )
                return

            # Remember where to insert: end of the current line.
            insert_cursor = editor.textCursor()
            insert_cursor.movePosition(insert_cursor.EndOfLine)
            self._insert_cursor = insert_cursor

            # File content as context (capped to keep prompts small)
            context = editor.toPlainText()
            if len(context) > 6000:
                context = context[-6000:]

            self.get_widget().generate_from_comment(instruction, context)

        except Exception as e:
            logger.warning("Code generation trigger failed: %s", e)
            self.get_widget()._append_system(
                f"Could not read editor: {e}"
            )

    def _insert_generated_code(self, code: str):
        """Insert generated code below the comment line."""
        cursor = getattr(self, "_insert_cursor", None)
        if cursor is None:
            return
        try:
            cursor.insertText("\n" + code + "\n")
            self.get_widget()._append_system("Code inserted into editor.")
        except Exception as e:
            logger.warning("Code insertion failed: %s", e)
            self.get_widget()._append_system(
                f"Could not insert code: {e}"
            )
        finally:
            self._insert_cursor = None

    def _explain_editor_selection(self):
        """Get selected text from editor and explain it."""
        try:
            editor_plugin = self.get_plugin(Plugins.Editor)
            if editor_plugin is None:
                self.get_widget()._append_system(
                    "Editor plugin not available."
                )
                return

            editor = editor_plugin.get_current_editor()
            if editor is None:
                self.get_widget()._append_system("No active editor.")
                return

            selected_text = editor.get_selected_text()
            if not selected_text:
                cursor = editor.textCursor()
                cursor.select(cursor.LineUnderCursor)
                selected_text = cursor.selectedText()

            self.get_widget().explain_code(selected_text)

        except Exception as e:
            logger.warning("Failed to get editor selection: %s", e)
            self.get_widget()._append_system(
                f"Could not access editor: {e}"
            )
