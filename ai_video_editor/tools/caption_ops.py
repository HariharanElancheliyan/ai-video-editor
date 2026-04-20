import logging
from pathlib import Path
from typing import Any

from ..config.settings import Settings
from ..utils.captions import CaptionGenerator, has_whisper_support

logger = logging.getLogger(__name__)


class CaptionTool:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.captioner = CaptionGenerator(
            self.settings.ffmpeg_path, self.settings.ffprobe_path
        )
        logger.info("CaptionTool initialized")

    def generate_screen_captions(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        language: str = "en",
        fps: int = 1,
        vision_model: str | None = None,
    ) -> dict[str, Any]:
        logger.debug("generate_screen_captions: %s -> %s", input_path, output_path)
        vision = vision_model or self.settings.vision_model or None
        return self.captioner.screen_capture_to_text(
            input_path, output_path, language=language, fps=fps, vision_model=vision
        )

    def generate_audio_captions(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        model: str = "base",
        language: str | None = None,
    ) -> dict[str, Any]:
        logger.debug("generate_audio_captions: %s -> %s", input_path, output_path)
        if not has_whisper_support():
            return {"error": "Whisper not available. Install: pip install ai-video-editor[audio]"}
        return self.captioner.audio_to_text(input_path, output_path, model, language)

    def generate_captions(
        self,
        input_path: str | Path,
        output_path: str | Path,
        prefer: str = "audio",
        fps: int = 1,
        whisper_model: str = "base",
    ) -> dict[str, Any]:
        return self.captioner.screen_and_audio_captions(
            input_path, output_path, prefer, fps, whisper_model
        )

    def embed_captions(
        self,
        input_path: str | Path,
        output_path: str | Path,
        caption_source: str | Path,
        create_if_missing: bool = False,
    ) -> dict[str, Any]:
        caption_path = Path(caption_source)
        if not caption_path.exists() and create_if_missing:
            temp_srt = Path(str(input_path) + ".srt")
            self.generate_captions(input_path, temp_srt, prefer="audio")
            caption_path = temp_srt

        if not caption_path.exists():
            return {"error": "Caption file not found"}

        result = self.captioner.embed_subtitles(input_path, output_path, caption_path)
        return {"success": result.returncode == 0, "output": str(output_path)}
