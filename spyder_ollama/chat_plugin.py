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
        ("model_name", "qwen2.5-coder:1.5b"),
        ("base_url", "http://localhost:11434"),
        ("temperature", 0.3),
    ]
    CONF_VERSION = "1.0.0"

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

        # Apply saved config
        widget.update_client_config(
            model_name=self.get_conf("model_name"),
            base_url=self.get_conf("base_url"),
            temperature=self.get_conf("temperature"),
        )

    @on_plugin_available(plugin=Plugins.Editor)
    def on_editor_available(self):
        """Register context menu action in the editor."""
        editor = self.get_plugin(Plugins.Editor)
        # The explain action is available via the toolbar in the chat panel
        # and can also be triggered via keyboard shortcut
        logger.info("Ollama chat: Editor plugin connected.")

    @on_plugin_teardown(plugin=Plugins.Editor)
    def on_editor_teardown(self):
        logger.info("Ollama chat: Editor plugin disconnected.")

    # ---- Private methods ------------------------------------------------
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
                # If nothing selected, use current line
                cursor = editor.textCursor()
                cursor.select(cursor.LineUnderCursor)
                selected_text = cursor.selectedText()

            self.get_widget().explain_code(selected_text)

        except Exception as e:
            logger.warning("Failed to get editor selection: %s", e)
            self.get_widget()._append_system(
                f"Could not access editor: {e}"
            )
