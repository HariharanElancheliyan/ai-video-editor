import logging
import os
import shutil
from pathlib import Path
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
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    default_model: str = os.getenv("OLLAMA_MODEL", "qwen3.6:latest")
    vision_model: str = os.getenv("OLLAMA_VISION_MODEL", "qwen3.6:latest")
    output_dir: Path = Path("./output")
    temp_dir: Path = Path("./temp")
    ffmpeg_path: str = _resolve_binary("ffmpeg")
    ffprobe_path: str = _resolve_binary("ffprobe")
    allow_system_drive_folder_access: bool = os.getenv("ALLOW_SYSTEM_DRIVE_FOLDER_ACCESS", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


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