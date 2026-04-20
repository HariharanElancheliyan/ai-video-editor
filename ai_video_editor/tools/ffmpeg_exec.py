import logging
import shlex
from typing import Any

from ..config.settings import Settings
from ..utils.ffmpeg_utils import FFmpegUtils

logger = logging.getLogger(__name__)


class FFmpegExecTool:
    """Generic wrapper that lets the AI agent execute arbitrary ffmpeg commands."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.ffmpeg = FFmpegUtils(self.settings.ffmpeg_path, self.settings.ffprobe_path)
        logger.info("FFmpegExecTool initialized (ffmpeg=%s)", self.settings.ffmpeg_path)

    def run_commands(self, commands: list[str]) -> dict[str, Any]:
        """Execute one or more ffmpeg commands sequentially.

        Each command string should contain only the ffmpeg arguments
        (e.g. ``"-i input.mp4 -vf scale=1280:720 output.mp4"``).
        The ``ffmpeg -y`` prefix is added automatically.

        Returns a summary dict with per-command results and overall success.
        """
        results: list[dict[str, Any]] = []
        all_success = True

        for idx, cmd in enumerate(commands):
            logger.info("FFmpegExec: running command %d/%d: %s", idx + 1, len(commands), cmd[:200])
            try:
                args = shlex.split(cmd)
            except ValueError as e:
                logger.error("FFmpegExec: failed to parse command %d: %s", idx + 1, e)
                results.append({"command": cmd, "success": False, "error": f"Failed to parse command: {e}"})
                all_success = False
                continue

            result = self.ffmpeg.run(args)
            success = result.returncode == 0
            if not success:
                all_success = False
            results.append({
                "command": cmd,
                "success": success,
                "returncode": result.returncode,
                "stdout": (result.stdout or "")[-500:],
                "stderr": (result.stderr or "")[-500:],
            })

        return {"success": all_success, "results": results}
