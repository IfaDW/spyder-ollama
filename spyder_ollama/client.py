# -*- coding: utf-8 -*-

# Copyright © 2024-2026 Datagniel (IfaDW)
# Licensed under the terms of the MIT License

"""Ollama completion client using LangChain LCEL."""

# Standard library imports
import json
import logging
import urllib.request
import urllib.error

# Third party imports
from langchain_ollama import ChatOllama
from qtpy.QtCore import QObject, QThread, Signal, QMutex, QMutexLocker, Slot

# Spyder imports
from spyder.plugins.completion.api import (
    CompletionRequestTypes,
    CompletionItemKind,
)


logger = logging.getLogger(__name__)

PROVIDER_LABEL = "Ollama"
ICON_SCALE = 1


def check_ollama_health(base_url: str, timeout: float = 3.0) -> bool:
    """Check if Ollama is reachable."""
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def fetch_ollama_models(base_url: str, timeout: float = 5.0) -> list[str]:
    """Fetch list of locally available model names from Ollama."""
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [m["name"] for m in data.get("models", [])]
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        return []


def pull_model_stream(base_url: str, model: str):
    """Pull a model via Ollama's API, yielding progress dicts.

    Yields dicts like {"status": "...", "total": int, "completed": int}.
    Raises URLError/OSError on connection problems.
    """
    payload = json.dumps({"name": model}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/pull",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def extract_cursor_context(
    full_text: str,
    cursor_line: int | None,
    cursor_col: int | None = None,
    lines_before: int = 50,
    lines_after: int = 10,
) -> str:
    """Extract a window of code around the cursor and insert a marker.

    cursor_line / cursor_col are 0-based (LSP convention). If either
    is unavailable or out of range, the marker falls back to the end
    of the line / end of the text.
    """
    lines = full_text.splitlines()
    total = len(lines)

    if total == 0:
        return "<|CURSOR|>"

    if cursor_line is None or not (0 <= cursor_line < total):
        cursor_line = total - 1
        cursor_col = None

    line_text = lines[cursor_line]
    if cursor_col is None or not (0 <= cursor_col <= len(line_text)):
        cursor_col = len(line_text)

    marked = line_text[:cursor_col] + "<|CURSOR|>" + line_text[cursor_col:]

    start = max(0, cursor_line - lines_before)
    end = min(total, cursor_line + 1 + lines_after)

    parts = lines[start:cursor_line] + [marked] + lines[cursor_line + 1 : end]
    return "\n".join(parts)


def parse_suggestions(raw_text: str) -> list[str]:
    """Robustly parse LLM output into a list of suggestion strings.

    Tries multiple strategies:
    1. Direct JSON parse
    2. Extract JSON object from surrounding text
    3. Extract JSON array from surrounding text
    4. Return raw text as single suggestion
    """
    text = raw_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Strategy 1: direct JSON object
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "suggestions" in obj:
            return [s for s in obj["suggestions"] if isinstance(s, str) and s]
        if isinstance(obj, list):
            return [s for s in obj if isinstance(s, str) and s]
    except (json.JSONDecodeError, TypeError):
        pass

    # Strategy 2: find JSON object in text
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            obj = json.loads(text[brace_start : brace_end + 1])
            if isinstance(obj, dict) and "suggestions" in obj:
                return [
                    s for s in obj["suggestions"] if isinstance(s, str) and s
                ]
        except (json.JSONDecodeError, TypeError):
            pass

    # Strategy 3: find JSON array in text
    bracket_start = text.find("[")
    bracket_end = text.rfind("]")
    if bracket_start != -1 and bracket_end > bracket_start:
        try:
            arr = json.loads(text[bracket_start : bracket_end + 1])
            if isinstance(arr, list):
                return [s for s in arr if isinstance(s, str) and s]
        except (json.JSONDecodeError, TypeError):
            pass

    # Strategy 4: return non-empty text as single suggestion
    if text and not text.startswith("{"):
        return [text]

    return []


class OllamaCompletionClient(QObject):
    sig_response_ready = Signal(int, dict)
    sig_client_started = Signal()
    sig_client_error = Signal(str)
    sig_perform_request = Signal(dict)
    sig_perform_status_request = Signal(str)
    sig_status_response_ready = Signal((str,), (dict,))

    def __init__(
        self,
        parent,
        model_name: str,
        base_url: str,
        system_prompt: str,
        num_suggestions: int = 3,
        temperature: float = 0.2,
        context_lines_before: int = 50,
        context_lines_after: int = 10,
    ):
        QObject.__init__(self, parent)
        self.mutex = QMutex()
        self.opened_files: dict[str, str] = {}
        self.thread_started = False
        self.thread = QThread(None)
        self.moveToThread(self.thread)
        self.thread.started.connect(self._on_thread_started)
        self.sig_perform_request.connect(self.handle_msg)
        self.sig_perform_status_request.connect(self.get_status)

        # Configuration
        self.model_name = model_name
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.num_suggestions = num_suggestions
        self.temperature = temperature
        self.context_lines_before = context_lines_before
        self.context_lines_after = context_lines_after

        # LangChain objects (created in start())
        self.llm = None

    def _build_llm(self):
        """Build the LLM.

        No prompt template on purpose: the system prompt contains
        literal JSON braces, which ChatPromptTemplate would treat as
        template variables (KeyError on every invoke).
        """
        return ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=self.temperature,
            format="json",
            num_predict=512,
        )

    def start(self):
        if not self.thread_started:
            self.thread.start()

        logger.info(
            "Starting Ollama completion client (model=%s, url=%s)",
            self.model_name,
            self.base_url,
        )

        if not check_ollama_health(self.base_url):
            msg = f"Ollama not reachable at {self.base_url}"
            logger.warning(msg)
            self.sig_client_error.emit(msg)
            return

        try:
            self.llm = self._build_llm()
            self.sig_client_started.emit()
            logger.info("Ollama completion client ready.")
        except Exception as e:
            logger.exception("Failed to initialize Ollama LLM")
            self.sig_client_error.emit(f"Init error: {e}")

    def _on_thread_started(self):
        self.thread_started = True

    def stop(self):
        if self.thread_started:
            logger.info("Stopping Ollama completion client.")
            self.thread.quit()
            self.thread.wait()
            self.thread_started = False

    def update_configuration(
        self,
        model_name: str,
        base_url: str,
        system_prompt: str,
        num_suggestions: int,
        temperature: float,
        context_lines_before: int,
        context_lines_after: int,
    ):
        """Rebuild the chain with new configuration."""
        self.stop()
        self.model_name = model_name
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.num_suggestions = num_suggestions
        self.temperature = temperature
        self.context_lines_before = context_lines_before
        self.context_lines_after = context_lines_after
        self.start()

    def get_status(self, filename):
        """Emit current model name as status."""
        status = self.model_name
        if self.llm is None:
            status = f"{self.model_name} (not connected)"
        self.sig_status_response_ready[str].emit(status)

    def _invoke_chain(self, code_context: str) -> list[str]:
        """Invoke the LLM with system + human message, parse suggestions."""
        if self.llm is None:
            return []

        with QMutexLocker(self.mutex):
            try:
                raw_response = self.llm.invoke(
                    [
                        ("system", self.system_prompt),
                        ("human", code_context),
                    ]
                ).content
                logger.debug("Raw LLM response: %s", raw_response)
                suggestions = parse_suggestions(raw_response)
                return suggestions[: self.num_suggestions]
            except Exception as e:
                logger.warning("Chain invocation failed: %s", e)
                self.sig_client_error.emit("No suggestions available")
                return []

    @Slot(dict)
    def handle_msg(self, message):
        """Handle a completion request from Spyder."""
        msg_type = message["type"]
        _id = message["id"]
        msg = message["msg"]

        logger.debug("Request type=%s id=%s", msg_type, _id)

        if msg_type == CompletionRequestTypes.DOCUMENT_DID_OPEN:
            self.opened_files[msg["file"]] = msg["text"]

        elif msg_type == CompletionRequestTypes.DOCUMENT_DID_CHANGE:
            self.opened_files[msg["file"]] = msg["text"]

        elif msg_type == CompletionRequestTypes.DOCUMENT_COMPLETION:
            full_text = self.opened_files.get(msg["file"], "")
            if not full_text.strip():
                self.sig_response_ready.emit(_id, {"params": []})
                return

            cursor_line = msg.get("line")
            cursor_col = msg.get("column", msg.get("character"))
            logger.debug(
                "Completion request keys=%s line=%s col=%s",
                sorted(msg.keys()), cursor_line, cursor_col,
            )
            context = extract_cursor_context(
                full_text,
                cursor_line,
                cursor_col,
                self.context_lines_before,
                self.context_lines_after,
            )

            suggestions = self._invoke_chain(context)

            spyder_completions = []
            for i, suggestion in enumerate(suggestions):
                entry = {
                    "kind": CompletionItemKind.TEXT,
                    "label": suggestion,
                    "insertText": suggestion,
                    "filterText": "",
                    "sortText": (0, i),
                    "documentation": suggestion,
                    "provider": PROVIDER_LABEL,
                    "icon": ("ollama", ICON_SCALE),
                }
                spyder_completions.append(entry)

            self.sig_response_ready.emit(_id, {"params": spyder_completions})
