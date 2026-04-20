import logging
import os
import shutil
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


def _resolve_binary(name: str) -> str:
    """Find a binary on PATH; otherwise fall back to the static-ffmpeg bundled copy."""
    found = shutil.which(name)
    if found:
        return found
    try:
        from static_ffmpeg import add_paths
        add_paths()
        found = shutil.which(name)
        if found:
            return found
    except Exception:
        pass
    return name


class Settings(BaseModel):
    # Provider selection
    llm_provider: Literal["ollama", "openai", "google"] = os.getenv("LLM_PROVIDER", "ollama")  # type: ignore[assignment]

    # Ollama settings
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3.6:latest")
    ollama_vision_model: str = os.getenv("OLLAMA_VISION_MODEL", "qwen3.6:latest")

    # OpenAI settings
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    openai_vision_model: str = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")

    # Google GenAI settings
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    google_model: str = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
    google_vision_model: str = os.getenv("GOOGLE_VISION_MODEL", "gemini-2.5-flash")

    # General settings
    output_dir: Path = Path("./output")
    temp_dir: Path = Path("./temp")
    ffmpeg_path: str = _resolve_binary("ffmpeg")
    ffprobe_path: str = _resolve_binary("ffprobe")
    allow_system_drive_folder_access: bool = os.getenv("ALLOW_SYSTEM_DRIVE_FOLDER_ACCESS", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    @property
    def default_model(self) -> str:
        if self.llm_provider == "openai":
            return self.openai_model
        if self.llm_provider == "google":
            return self.google_model
        return self.ollama_model

    @property
    def vision_model(self) -> str:
        if self.llm_provider == "openai":
            return self.openai_vision_model
        if self.llm_provider == "google":
            return self.google_vision_model
        return self.ollama_vision_model


def setup_logging(level: str | None = None) -> None:
    """Configure logging for the application."""
    log_level = (level or Settings().log_level).upper()
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "ai_video_editor.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(getattr(logging, log_level, logging.INFO))
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))
    root_logger.addHandler(file_handler)