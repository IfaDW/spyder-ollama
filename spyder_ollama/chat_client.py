# -*- coding: utf-8 -*-

# Copyright © 2024-2026 Datagniel (IfaDW)
# Licensed under the terms of the MIT License

"""Streaming Ollama chat client for the assistant panel."""

# Standard library imports
import logging

# Third party imports
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from qtpy.QtCore import QObject, QThread, Signal, Slot

# Local imports
from spyder_ollama.client import check_ollama_health


logger = logging.getLogger(__name__)


EXPLAIN_SYSTEM_PROMPT = (
    "You are a helpful coding assistant embedded in a Python IDE. "
    "Explain the given code clearly and concisely. "
    "Focus on what the code does, why it might be written that way, "
    "and mention any potential issues or improvements. "
    "Use plain language. Format your response in Markdown."
)

CHAT_SYSTEM_PROMPT = (
    "You are a helpful coding assistant embedded in a Python IDE called Spyder. "
    "Answer questions about Python, data science, and programming concisely. "
    "When showing code, use Markdown code blocks. "
    "Be direct and practical."
)


class _StreamWorker(QObject):
    """Worker that runs LLM streaming in a background thread."""

    sig_token = Signal(str)
    sig_finished = Signal()
    sig_error = Signal(str)

    def __init__(self):
        super().__init__()
        self.llm = None
        self.messages = []

    def configure(self, llm, messages):
        self.llm = llm
        self.messages = messages

    @Slot()
    def run(self):
        """Stream tokens from the LLM."""
        try:
            for chunk in self.llm.stream(self.messages):
                token = chunk.content
                if token:
                    self.sig_token.emit(token)
            self.sig_finished.emit()
        except Exception as e:
            logger.warning("Chat stream error: %s", e)
            self.sig_error.emit(str(e))
            self.sig_finished.emit()


class OllamaChatClient(QObject):
    """Client for chat/explain interactions with Ollama.

    Streams responses token by token via signals.
    """

    sig_token = Signal(str)
    sig_response_finished = Signal()
    sig_error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model_name = "qwen2.5-coder:1.5b"
        self.base_url = "http://localhost:11434"
        self.temperature = 0.3

        self._thread = None
        self._worker = None

    def update_config(self, model_name: str, base_url: str, temperature: float):
        """Update connection parameters."""
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature

    def _build_llm(self):
        return ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=self.temperature,
        )

    def is_available(self) -> bool:
        """Check if Ollama is reachable."""
        return check_ollama_health(self.base_url)

    def explain_code(self, code: str):
        """Send a code explanation request (streaming)."""
        messages = [
            ("system", EXPLAIN_SYSTEM_PROMPT),
            ("human", f"Explain this code:\n\n```python\n{code}\n```"),
        ]
        self._start_stream(messages)

    def ask(self, question: str, code_context: str = ""):
        """Send a freeform question, optionally with code context."""
        human_msg = question
        if code_context:
            human_msg = (
                f"Here is the code I'm working with:\n\n"
                f"```python\n{code_context}\n```\n\n{question}"
            )
        messages = [
            ("system", CHAT_SYSTEM_PROMPT),
            ("human", human_msg),
        ]
        self._start_stream(messages)

    def _start_stream(self, messages):
        """Start streaming in a background thread."""
        # Clean up previous thread if running
        self.stop()

        if not self.is_available():
            self.sig_error.emit(
                f"Ollama not reachable at {self.base_url}"
            )
            self.sig_response_finished.emit()
            return

        try:
            llm = self._build_llm()
        except Exception as e:
            self.sig_error.emit(f"Failed to create LLM: {e}")
            self.sig_response_finished.emit()
            return

        self._thread = QThread()
        self._worker = _StreamWorker()
        self._worker.configure(llm, messages)
        self._worker.moveToThread(self._thread)

        # Connect signals
        self._thread.started.connect(self._worker.run)
        self._worker.sig_token.connect(self.sig_token)
        self._worker.sig_error.connect(self.sig_error)
        self._worker.sig_finished.connect(self._on_finished)

        self._thread.start()

    def _on_finished(self):
        """Clean up after streaming completes."""
        self.sig_response_finished.emit()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
            self._worker = None

    def stop(self):
        """Stop any running stream."""
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()
            self._thread = None
            self._worker = None
