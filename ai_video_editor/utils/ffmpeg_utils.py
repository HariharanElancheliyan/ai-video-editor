import logging
import subprocess
import json
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def escape_filter_path(path: str | Path) -> str:
    """Escape a file path for use inside an ffmpeg filter string (e.g. subtitles=...)."""
    s = str(path).replace("\\", "/")
    s = s.replace("'", r"\'")
    s = s.replace(":", r"\:")
    return s


class FFmpegUtils:
    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe"):
        self.ffmpeg = ffmpeg_path
        self.ffprobe = ffprobe_path

    def probe(self, file_path: str | Path) -> dict[str, Any]:
        cmd = [
            self.ffprobe,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(file_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not result.stdout.strip():
            return {"error": f"ffprobe failed (code {result.returncode}): {(result.stderr or '').strip()[:500]}"}
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse ffprobe output: {e}"}

    def get_duration(self, file_path: str | Path) -> float:
        info = self.probe(file_path)
        return float(info.get("format", {}).get("duration", 0))

    def get_resolution(self, file_path: str | Path) -> tuple[int, int]:
        info = self.probe(file_path)
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                return stream.get("width", 0), stream.get("height", 0)
        return 0, 0

    def get_fps(self, file_path: str | Path) -> float:
        info = self.probe(file_path)
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                fps_str = stream.get("r_frame_rate", "0/1")
                if "/" in fps_str:
                    num, den = map(int, fps_str.split("/"))
                    return num / den if den else 0
                return float(fps_str)
        return 0

    def get_codec(self, file_path: str | Path) -> str:
        info = self.probe(file_path)
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                return stream.get("codec_name", "unknown")
        return "unknown"

    def run(self, args: list[str], capture_output: bool = True) -> subprocess.CompletedProcess:
        cmd = [self.ffmpeg, "-y"] + args
        logger.debug("FFmpeg command: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=capture_output, text=True)
        if result.returncode != 0:
            logger.error("FFmpeg failed (code %d): %s", result.returncode, (result.stderr or "")[:300])
        return result