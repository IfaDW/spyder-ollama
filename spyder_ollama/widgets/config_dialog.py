# -*- coding: utf-8 -*-
#
# Copyright © 2024-2026 Datagniel (IfaDW)
# Licensed under the terms of the MIT License

"""Configuration dialog for Ollama completion provider."""

# Standard library imports
import logging

# Third party imports
from qtpy.QtCore import Qt, QObject, QThread, Signal, Slot
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

# Spyder imports
from spyder.api.translations import _

# Local imports
from spyder_ollama.client import (
    check_ollama_health,
    fetch_ollama_models,
    pull_model_stream,
)

logger = logging.getLogger(__name__)


class _PullWorker(QObject):
    """Streams an Ollama model pull in a background thread."""

    sig_progress = Signal(str, int)   # status text, percent (-1 = unknown)
    sig_finished = Signal(bool, str)  # success, message

    def __init__(self, base_url: str, model: str):
        super().__init__()
        self.base_url = base_url
        self.model = model

    @Slot()
    def run(self):
        try:
            for update in pull_model_stream(self.base_url, self.model):
                status = update.get("status", "")
                total = update.get("total")
                completed = update.get("completed")
                if total and completed is not None and total > 0:
                    percent = int(completed * 100 / total)
                else:
                    percent = -1
                self.sig_progress.emit(status, percent)
                if status == "success":
                    self.sig_finished.emit(
                        True, f"Model '{self.model}' pulled."
                    )
                    return
                if "error" in update:
                    self.sig_finished.emit(False, update["error"])
                    return
            # Stream ended without explicit success
            self.sig_finished.emit(True, f"Pull of '{self.model}' finished.")
        except Exception as e:
            self.sig_finished.emit(False, str(e))


class OllamaConfigDialog(QDialog):
    def __init__(self, provider, parent=None):
        super().__init__(parent=parent)

        self._provider = provider
        self._pull_thread = None
        self._pull_worker = None

        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle(_("Ollama Completion — Configuration"))
        self.setModal(True)
        self.setMinimumWidth(460)

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

        # --- Pull model ---
        self.pull_edit = QLineEdit()
        self.pull_edit.setPlaceholderText(
            _("e.g. qwen2.5-coder:1.5b — see ollama.com/library")
        )
        self.pull_btn = QPushButton(_("Pull"))
        self.pull_btn.clicked.connect(self._start_pull)

        pull_layout = QHBoxLayout()
        pull_layout.addWidget(self.pull_edit, stretch=1)
        pull_layout.addWidget(self.pull_btn)

        self.pull_progress = QProgressBar()
        self.pull_progress.setRange(0, 100)
        self.pull_progress.setValue(0)
        self.pull_progress.setVisible(False)

        self.pull_status = QLabel("")
        self.pull_status.setWordWrap(True)
        self.pull_status.setVisible(False)

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
        form_layout.addRow(_("Completion model:"), model_layout)
        form_layout.addRow(_("Pull model:"), pull_layout)
        form_layout.addRow("", self.pull_progress)
        form_layout.addRow("", self.pull_status)
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

    # ---- Model list ------------------------------------------------------
    def _populate_models(self):
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

    # ---- Pull ------------------------------------------------------------
    def _start_pull(self):
        model = self.pull_edit.text().strip()
        if not model:
            return
        if self._pull_thread is not None:
            return  # a pull is already running

        base_url = self.url_edit.text().strip() or "http://localhost:11434"

        self.pull_btn.setEnabled(False)
        self.pull_progress.setVisible(True)
        self.pull_progress.setRange(0, 0)  # busy until first percentage
        self.pull_status.setVisible(True)
        self.pull_status.setText(_("Starting pull..."))

        self._pull_thread = QThread(self)
        self._pull_worker = _PullWorker(base_url, model)
        self._pull_worker.moveToThread(self._pull_thread)
        self._pull_thread.started.connect(self._pull_worker.run)
        self._pull_worker.sig_progress.connect(self._on_pull_progress)
        self._pull_worker.sig_finished.connect(self._on_pull_finished)
        self._pull_thread.start()

    def _on_pull_progress(self, status: str, percent: int):
        if percent >= 0:
            self.pull_progress.setRange(0, 100)
            self.pull_progress.setValue(percent)
        self.pull_status.setText(status)

    def _on_pull_finished(self, success: bool, message: str):
        self.pull_status.setText(message)
        self.pull_progress.setRange(0, 100)
        self.pull_progress.setValue(100 if success else 0)
        self.pull_btn.setEnabled(True)

        if self._pull_thread is not None:
            self._pull_thread.quit()
            self._pull_thread.wait()
            self._pull_thread = None
            self._pull_worker = None

        if success:
            pulled = self.pull_edit.text().strip()
            self.pull_edit.clear()
            self._populate_models()
            if pulled:
                self.model_combobox.setCurrentText(pulled)

    def closeEvent(self, event):
        # Don't leave a pull thread orphaned; the server-side pull
        # continues regardless, we just stop listening.
        if self._pull_thread is not None:
            self._pull_thread.quit()
            self._pull_thread.wait(2000)
        super().closeEvent(event)

    # ---- Accept ----------------------------------------------------------
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
