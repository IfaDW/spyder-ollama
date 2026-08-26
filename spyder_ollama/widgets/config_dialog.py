# -*- coding: utf-8 -*-
#
# Copyright © 2024-2026 Datagniel (IfaDW)
# Licensed under the terms of the MIT License

"""Configuration dialog for Ollama completion provider."""

# Standard library imports
import logging

# Third party imports
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

# Spyder imports
from spyder.api.translations import _

# Local imports
from spyder_ollama.client import check_ollama_health, fetch_ollama_models

logger = logging.getLogger(__name__)


class OllamaConfigDialog(QDialog):
    def __init__(self, provider, parent=None):
        super().__init__(parent=parent)

        self._provider = provider

        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle(_("Ollama Completion — Configuration"))
        self.setModal(True)
        self.setMinimumWidth(420)

        # --- Base URL ---
        self.url_edit = QLineEdit()
        self.url_edit.setText(provider.get_conf("base_url"))
        self.url_edit.setPlaceholderText("http://localhost:11434")

        self.test_btn = QPushButton(_("Test Connection"))
        self.test_btn.clicked.connect(self._test_connection)

        url_layout = QHBoxLayout()
        url_layout.addWidget(self.url_edit, stretch=1)
        url_layout.addWidget(self.test_btn)

        # --- Model selection ---
        self.model_combobox = QComboBox()
        self.model_combobox.setEditable(True)
        self._populate_models()
        self.model_combobox.setCurrentText(provider.get_conf("model_name"))

        self.refresh_btn = QPushButton(_("Refresh"))
        self.refresh_btn.clicked.connect(self._populate_models)

        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_combobox, stretch=1)
        model_layout.addWidget(self.refresh_btn)

        # --- Suggestions ---
        self.suggestions_spinbox = QSpinBox()
        self.suggestions_spinbox.setRange(1, 10)
        self.suggestions_spinbox.setSingleStep(1)
        self.suggestions_spinbox.setValue(provider.get_conf("suggestions"))

        # --- Temperature ---
        self.temperature_spinbox = QDoubleSpinBox()
        self.temperature_spinbox.setRange(0.0, 2.0)
        self.temperature_spinbox.setSingleStep(0.1)
        self.temperature_spinbox.setDecimals(2)
        self.temperature_spinbox.setValue(provider.get_conf("temperature"))

        # --- Context window ---
        self.ctx_before_spinbox = QSpinBox()
        self.ctx_before_spinbox.setRange(10, 200)
        self.ctx_before_spinbox.setSingleStep(10)
        self.ctx_before_spinbox.setValue(
            provider.get_conf("context_lines_before")
        )

        self.ctx_after_spinbox = QSpinBox()
        self.ctx_after_spinbox.setRange(0, 50)
        self.ctx_after_spinbox.setSingleStep(5)
        self.ctx_after_spinbox.setValue(
            provider.get_conf("context_lines_after")
        )

        # --- Form layout ---
        form_layout = QFormLayout()
        form_layout.addRow(_("Ollama URL:"), url_layout)
        form_layout.addRow(_("Model:"), model_layout)
        form_layout.addRow(_("Suggestions:"), self.suggestions_spinbox)
        form_layout.addRow(_("Temperature:"), self.temperature_spinbox)
        form_layout.addRow(
            _("Context lines before cursor:"), self.ctx_before_spinbox
        )
        form_layout.addRow(
            _("Context lines after cursor:"), self.ctx_after_spinbox
        )

        # --- Buttons ---
        bbox = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal,
            self,
        )
        bbox.accepted.connect(self.accept)
        bbox.rejected.connect(self.reject)

        # --- Main layout ---
        layout = QVBoxLayout()
        layout.addLayout(form_layout)
        layout.addWidget(bbox)
        self.setLayout(layout)

        self.model_combobox.setFocus()

    def _populate_models(self):
        """Fetch models from Ollama and populate the combobox."""
        base_url = self.url_edit.text().strip() or "http://localhost:11434"
        current_text = self.model_combobox.currentText()

        models = fetch_ollama_models(base_url)
        self.model_combobox.clear()

        if models:
            self.model_combobox.addItems(sorted(models))
        else:
            logger.warning("No models found at %s", base_url)

        if current_text:
            self.model_combobox.setCurrentText(current_text)

    def _test_connection(self):
        """Test if Ollama is reachable at the configured URL."""
        base_url = self.url_edit.text().strip() or "http://localhost:11434"
        if check_ollama_health(base_url):
            models = fetch_ollama_models(base_url)
            QMessageBox.information(
                self,
                _("Connection OK"),
                _(
                    "Ollama is reachable.\n"
                    f"{len(models)} model(s) available."
                ),
            )
            self._populate_models()
        else:
            QMessageBox.warning(
                self,
                _("Connection Failed"),
                _(
                    f"Cannot reach Ollama at:\n{base_url}\n\n"
                    "Make sure Ollama is running."
                ),
            )

    def accept(self):
        self._provider.set_conf("base_url", self.url_edit.text().strip())
        self._provider.set_conf(
            "model_name", self.model_combobox.currentText()
        )
        self._provider.set_conf(
            "suggestions", self.suggestions_spinbox.value()
        )
        self._provider.set_conf(
            "temperature", self.temperature_spinbox.value()
        )
        self._provider.set_conf(
            "context_lines_before", self.ctx_before_spinbox.value()
        )
        self._provider.set_conf(
            "context_lines_after", self.ctx_after_spinbox.value()
        )
        super().accept()
