# AI Video Editor

An AI-powered video editing tool that uses [Ollama](https://ollama.com/) LLMs and FFmpeg to perform video editing tasks through natural language. Describe what you want done and the agent will select and execute the right FFmpeg operations automatically.

## Features

- **Natural Language Editing** — describe edits in plain English; the agent plans and executes FFmpeg commands
- **35+ Video Operations** — trim, resize, crop, rotate, flip, pad, change FPS/codec/speed, reverse, create GIFs, extract frames, and more
- **Color & Effects** — brightness, contrast, saturation, gamma, color temperature, vignette, blur, sharpen, grayscale, sepia
- **Audio** — extract audio, adjust volume, fade in/out, normalize loudness
- **Captions & Subtitles** — generate captions from audio (Whisper) or screen content (LLM vision), burn subtitles into video
- **Idle Frame Detection** — detect and speed up frozen/static segments in screen recordings
- **Text & Watermark Overlays** — add text or image watermarks at configurable positions
- **Fade Transitions** — video and audio fade in/out
- **Format Conversion** — convert between video formats with configurable codecs and bitrates
- **Interactive & Single-Shot Modes** — REPL loop for iterative editing or one-off commands via CLI

## Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.com/)** — must be installed and running locally (default `http://localhost:11434`). Pull at least one model before using the tool:
  ```bash
  ollama pull qwen3.6:latest
  ```
- **FFmpeg** — if FFmpeg is already on your system PATH it will be used automatically. Otherwise the bundled [`static-ffmpeg`](https://pypi.org/project/static-ffmpeg/) package downloads a portable copy on first run, so no manual install is needed.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/HariharanElancheliyan/ai-video-editor.git
cd ai-video-editor
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install the package

**Core install** — includes video editing, LLM agent, and FFmpeg support:

```bash
pip install -e .
```

**With audio captions (Whisper)** — adds speech-to-text captioning via OpenAI Whisper:

```bash
pip install -e ".[audio]"
```

**All extras:**

```bash
pip install -e ".[all]"
```

> **GPU acceleration (optional):**
> - NVIDIA — install CUDA-enabled PyTorch (`pip install torch` with CUDA) for faster Whisper inference.
> - AMD / Intel on Windows — install `torch-directml` for DirectML acceleration.
> - Alternatively, install `faster-whisper` instead of `openai-whisper` for CTranslate2-based inference that supports both NVIDIA and CPU int8.

### 4. Verify the installation

```bash
ai-video-editor --help
```

## Configuration

Settings are loaded from environment variables (or a `.env` file):

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen3.6:latest` | Default LLM model |
| `OLLAMA_VISION_MODEL` | `qwen3.6:latest` | Vision model for screen captions |

## Usage

### Interactive Mode

```bash
python -m ai_video_editor
```

You'll enter a REPL where you can type requests like:

```
You: trim input.mp4 to the first 30 seconds and save as output.mp4
You: resize video.mp4 to 1280x720
You: speed up idle segments in recording.mp4
You: add captions to demo.mp4
```

Commands: `/reset` to clear history, `/exit` to quit.


### CLI Options

| Flag | Description |
|---|---|
| `-p`, `--prompt` | Initial prompt (interactive mode continues after) |
| `-m`, `--model` | Override the Ollama model |

## Project Structure

```
ai_video_editor/
├── __main__.py              # CLI entry point (Typer app)
├── agents/
│   └── video_editor_agent.py # LLM agent loop with tool calling
├── config/
│   └── settings.py           # Environment-based configuration
├── core/
│   ├── ollama_client.py      # Async Ollama API client
│   └── types.py              # Pydantic models (Message, ToolDefinition, etc.)
├── tools/
│   ├── video_ops.py          # FFmpeg-backed video operations (35+ tools)
│   ├── caption_ops.py        # Caption generation & embedding tool
│   └── file_ops.py           # File/folder read, delete (recycle bin), copy, move
└── utils/
    ├── captions.py           # Caption generation (Whisper + LLM vision)
    └── ffmpeg_utils.py       # FFmpeg/ffprobe wrapper
```
