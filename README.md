# spyder-ollama

Local AI code completion and assistant for [Spyder 6+](https://www.spyder-ide.org/) via [Ollama](https://ollama.com/).

Two features in one package:

1. **Inline completions** — suggests code as you type, powered by a local model
2. **Ollama Assistant panel** — explain selected code, ask questions, get coding help — all streaming, all local

No API keys, no cloud, fully private.

## Requirements

- **Spyder** ≥ 6.0.0a3
- **Ollama** running locally (default: `http://localhost:11434`)
- A code-capable model, e.g.:
  ```bash
  ollama pull qwen2.5-coder:1.5b
  ```

## Installation

```bash
pip install git+https://github.com/IfaDW/spyder-ollama.git
```

## Feature 1: Inline Completions

Works automatically. Click the **Ollama** status bar widget → **Configure Ollama provider** to adjust settings.

| Setting | Default | Description |
|---|---|---|
| Ollama URL | `http://localhost:11434` | Base URL of your Ollama instance |
| Model | `qwen2.5-coder:1.5b` | Any locally available model |
| Suggestions | 3 | Number of completions (1–10) |
| Temperature | 0.2 | Lower = more deterministic |
| Context before cursor | 50 lines | Code context sent before cursor |
| Context after cursor | 10 lines | Code context sent after cursor |

## Feature 2: Ollama Assistant Panel

Open via **View → Panes → Ollama Assistant**.

**Explain code:** Select code in the editor → click **Explain Selection** in the assistant toolbar (or use the keyboard shortcut). The assistant streams a plain-language explanation.

**Ask questions:** Type any question in the input field and press **Shift+Enter** to send. The assistant responds with streaming output.

**Stop generation:** Click **Stop** to interrupt a long response.

The assistant panel shares the same Ollama connection. Model and URL are configured via the completion provider settings.

## How it works

**Completions:** On trigger, the plugin extracts a code window around your cursor, inserts a `<|CURSOR|>` marker, and sends it to Ollama with `format="json"` for structured suggestions.

**Assistant:** Selected code or questions are sent to Ollama with a chat-oriented system prompt. Responses stream token by token into the panel.

## Origin

This project started as a fork of [spyder-ide/langchain-provider](https://github.com/spyder-ide/langchain-provider) but has been rewritten from scratch with a different architecture.

## License

[MIT](LICENSE)
