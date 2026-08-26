# -*- coding: utf-8 -*-

# Copyright © 2024-2026 Datagniel (IfaDW)
# Licensed under the terms of the MIT License

"""Ollama completion provider for Spyder."""

# Standard library imports
import logging
import os

# Qt imports
from qtpy.QtCore import Slot

# Local imports
from spyder_ollama.client import OllamaCompletionClient
from spyder_ollama.widgets import OllamaStatusWidget

# Spyder imports
from spyder.api.config.decorators import on_conf_change
from spyder.config.base import running_under_pytest, get_module_data_path
from spyder.plugins.completion.api import SpyderCompletionProvider
from spyder.utils.image_path_manager import IMAGE_PATH_MANAGER


logger = logging.getLogger(__name__)

_IMAGE_PATH_REGISTERED = False


def _register_image_path():
    """Register the plugin image path exactly once.

    The provider can be instantiated more than once by Spyder;
    repeated registration triggers an override warning.
    """
    global _IMAGE_PATH_REGISTERED
    if not _IMAGE_PATH_REGISTERED:
        IMAGE_PATH_MANAGER.add_image_path(
            get_module_data_path("spyder_ollama", relpath="images")
        )
        _IMAGE_PATH_REGISTERED = True


class OllamaProvider(SpyderCompletionProvider):
    COMPLETION_PROVIDER_NAME = "ollama"
    DEFAULT_ORDER = 1
    SLOW = True
    CONF_VERSION = "1.1.0"
    CONF_DEFAULTS = [
        ("suggestions", 3),
        ("model_name", "qwen2.5-coder:1.5b"),
        ("base_url", "http://localhost:11434"),
        ("context_lines_before", 50),
        ("context_lines_after", 10),
        ("temperature", 0.2),
    ]

    SYSTEM_PROMPT = (
        "You are a code completion engine inside a Python IDE. "
        "The input is a code excerpt with the exact cursor position "
        "marked as <|CURSOR|>. Predict ONLY the text to INSERT at "
        "<|CURSOR|>.\n"
        "Rules:\n"
        "- The characters directly before <|CURSOR|> are the prefix the "
        "user is typing. Every suggestion must continue seamlessly from "
        "that prefix. If the prefix ends with 'df.', every suggestion "
        "must be a method or attribute that completes 'df.' — for "
        "example 'head()' or 'columns'.\n"
        "- Never repeat text that already appears before the cursor.\n"
        "- Each suggestion is short: an identifier, a call, or the rest "
        "of the current statement. No full programs.\n"
        "- Respond ONLY with a JSON object of the form "
        '{"suggestions": ["...", "..."]}. '
        "No markdown, no explanation.\n\n"
        "Example input:\n"
        "import pandas as pd\n"
        "df = pd.read_csv('data.csv')\n"
        "df.drop<|CURSOR|>\n"
        "Example output:\n"
        '{"suggestions": ["_duplicates()", "na(subset=[\'col\'])", '
        '"(columns=[\'col\'])"]}'
    )

    def __init__(self, parent, config):
        super().__init__(parent, config)
        _register_image_path()
        self.available_languages = []
        self.client = OllamaCompletionClient(
            parent=None,
            model_name=self.get_conf("model_name"),
            base_url=self.get_conf("base_url"),
            system_prompt=self.SYSTEM_PROMPT,
            num_suggestions=self.get_conf("suggestions"),
            temperature=self.get_conf("temperature"),
            context_lines_before=self.get_conf("context_lines_before"),
            context_lines_after=self.get_conf("context_lines_after"),
        )

        # Signals
        self.client.sig_client_started.connect(
            lambda: self.sig_provider_ready.emit(self.COMPLETION_PROVIDER_NAME)
        )
        self.client.sig_client_error.connect(self.set_status_error)
        self.client.sig_status_response_ready[str].connect(self.set_status)
        self.client.sig_status_response_ready[dict].connect(self.set_status)
        self.client.sig_response_ready.connect(
            lambda _id, resp: self.sig_response_ready.emit(
                self.COMPLETION_PROVIDER_NAME, _id, resp
            )
        )

        # Status bar widget
        self.STATUS_BAR_CLASSES = [self.create_statusbar]
        self.started = False

    # ------------------ SpyderCompletionProvider methods ---------------------
    def get_name(self):
        return "Ollama"

    def send_request(self, language, req_type, req, req_id):
        request = {
            "type": req_type,
            "file": req["file"],
            "id": req_id,
            "msg": req,
        }
        self.client.sig_perform_request.emit(request)

    def start_completion_services_for_language(self, language):
        return self.started

    def start(self):
        if not self.started:
            self.client.start()
            self.started = True

    def shutdown(self):
        if self.started:
            self.client.stop()
            self.started = False

    @Slot(str)
    @Slot(dict)
    def set_status(self, status):
        """Show provider status in the status bar."""
        self.sig_call_statusbar.emit(
            OllamaStatusWidget.ID, "set_value", (status,), {}
        )

    def set_status_error(self, error_message):
        """Show provider error in the status bar."""
        self.sig_call_statusbar.emit(
            OllamaStatusWidget.ID, "set_value", (error_message,), {}
        )

    def file_opened_closed_or_updated(self, filename, _language):
        """Request status for the given file."""
        self.client.sig_perform_status_request.emit(filename)

    @on_conf_change(
        section="completions", option=("enabled_providers", "ollama")
    )
    def on_ollama_enable_changed(self, value):
        self.sig_call_statusbar.emit(
            OllamaStatusWidget.ID, "set_value", (None,), {}
        )

    @on_conf_change
    def update_ollama_configuration(self, config):
        if running_under_pytest():
            if not os.environ.get("SPY_TEST_USE_INTROSPECTION"):
                return
        self.client.update_configuration(
            model_name=self.get_conf("model_name"),
            base_url=self.get_conf("base_url"),
            system_prompt=self.SYSTEM_PROMPT,
            num_suggestions=self.get_conf("suggestions"),
            temperature=self.get_conf("temperature"),
            context_lines_before=self.get_conf("context_lines_before"),
            context_lines_after=self.get_conf("context_lines_after"),
        )

    def create_statusbar(self, parent):
        return OllamaStatusWidget(parent, self)
