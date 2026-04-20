import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable
from ..core.ollama_client import OllamaClient
from ..core.types import Message, ToolDefinition
from ..tools.video_ops import VideoTool
from ..tools.file_ops import FileTool
from ..tools.ffmpeg_exec import FFmpegExecTool
from ..config.settings import Settings

logger = logging.getLogger(__name__)


class VideoEditorAgent:
    def _register_tools(self):
        # Map schema tool names (exposed to the LLM) -> VideoTool method names
        name_map = {
            "get_video_info": "get_info",
            "trim_video": "trim",
            "resize_video": "resize",
            "change_fps": "change_fps",
            "change_codec": "change_codec",
            "extract_audio": "extract_audio",
            "concatenate_videos": "concatenate",
            "add_subtitle": "add_subtitle",
            "add_watermark": "add_watermark",
            "change_speed": "change_speed",
            "reverse_video": "reverse",
            "create_gif": "create_gif",
            "extract_frames": "extract_frames",
            "adjust_brightness": "adjust_brightness",
            "adjust_contrast": "adjust_contrast",
            "color_correct": "color_correct",
            "vignette": "vignette",
            "blur": "blur",
            "sharpen": "sharpen",
            "rotate": "rotate",
            "flip": "flip",
            "crop": "crop",
            "pad": "pad",
            "fade_in_out": "fade_in_out",
            "add_text": "add_text",
            "set_volume": "set_volume",
            "add_fade_audio": "add_fade_audio",
            "normalize_audio": "normalize_audio",
            "convert_format": "convert_format",
            "to_grayscale": "to_grayscale",
            "sepia": "sepia",
            "detect_idle_segments": "detect_idle_segments",
            "speed_idle_frames": "speed_idle_frames",
            "generate_screen_captions": "generate_screen_captions",
            "generate_audio_captions": "generate_audio_captions",
            "generate_captions": "generate_captions",
            "embed_captions": "embed_captions",
        }
        for schema_name, method_name in name_map.items():
            method = getattr(self.video_tool, method_name, None)
            if method is not None:
                self._tool_registry[schema_name] = method
            # also allow calling by the raw method name
            if method is not None and method_name not in self._tool_registry:
                self._tool_registry[method_name] = method

        # File tool mappings
        file_name_map = {
            "read_file": "read_file",
            "read_folder": "read_folder",
            "file_info": "file_info",
            "delete_file": "delete_file",
            "delete_folder": "delete_folder",
            "create_directory": "create_directory",
            "move_file": "move",
            "copy_file": "copy",
            "file_exists": "exists",
            "rename_file": "rename",
        }
        for schema_name, method_name in file_name_map.items():
            method = getattr(self.file_tool, method_name, None)
            if method is not None:
                self._tool_registry[schema_name] = method
            if method is not None and method_name not in self._tool_registry:
                self._tool_registry[method_name] = method

        # FFmpeg exec tool
        self._tool_registry["run_ffmpeg_commands"] = self.ffmpeg_exec_tool.run_commands

    def get_tools(self) -> list[ToolDefinition]:
        tools = []
        tool_descriptions = {
            "get_video_info": "Get information about a video file including duration, resolution, codec, FPS",
            "trim_video": "Trim a video to a specific start time and duration",
            "resize_video": "Resize a video to specific width and height",
            "change_fps": "Change the frames per second of a video",
            "change_codec": "Change the video codec (e.g., to h264)",
            "extract_audio": "Extract audio from a video file",
            "concatenate_videos": "Concatenate multiple videos together",
            "add_subtitle": "Add subtitles to a video",
            "add_watermark": "Add a watermark image to a video",
            "change_speed": "Change the playback speed of a video",
            "reverse_video": "Reverse a video (play backwards)",
            "create_gif": "Create an animated GIF from a video",
            "extract_frames": "Extract frames from a video as images",
            "adjust_brightness": "Adjust the brightness of a video",
            "adjust_contrast": "Adjust the contrast of a video",
            "color_correct": "Adjust color (saturation, gamma, temperature, tint)",
            "vignette": "Add vignette effect to video",
            "blur": "Apply blur effect to video",
            "sharpen": "Sharpen the video",
            "rotate": "Rotate the video by degrees",
            "flip": "Flip video horizontally or vertically",
            "crop": "Crop video to specific region",
            "pad": "Add padding/border to video",
            "fade_in_out": "Add fade in/out transitions",
            "add_text": "Add text overlay on video",
            "set_volume": "Adjust audio volume",
            "add_fade_audio": "Fade audio in/out",
            "normalize_audio": "Normalize audio loudness",
            "convert_format": "Convert to different format",
            "to_grayscale": "Convert video to grayscale",
            "sepia": "Apply sepia color effect",
            "detect_idle_segments": "Detect idle/frozen segments in a video. Returns a list of time ranges where the screen is static (e.g. '10.0s - 17.0s idle for 7.0s'). Useful for identifying dead air in screen recordings.",
            "speed_idle_frames": "Speed up idle/frozen segments in a video. Detects static frames and speeds them up by a given factor while keeping active segments at normal speed.",
            "generate_screen_captions": "Generate captions from video frames using LLM vision model",
            "generate_audio_captions": "Generate captions from audio speech (Whisper)",
            "generate_captions": "Auto-generate captions (prefer audio, fallback to vision-based screen captions)",
            "embed_captions": "Burn captions into video as subtitles",
            "read_file": "Read a file's content and metadata (path, size, modified date)",
            "read_folder": "List contents of a directory with metadata. Supports recursive listing and glob patterns",
            "file_info": "Get metadata (size, dates, type) for a file or directory without reading content",
            "delete_file": "Delete a file by sending it to the recycle bin (recoverable). Blocks system drive deletion unless allowed in settings",
            "delete_folder": "Delete a folder by sending it to the recycle bin (recoverable). Blocks system drive deletion unless allowed in settings",
            "create_directory": "Create a new directory (and parent directories)",
            "move_file": "Move or rename a file or folder",
            "copy_file": "Copy a file or folder to a new location",
            "file_exists": "Check whether a path exists and its type (file or directory)",
            "rename_file": "Rename a file or folder in-place (same directory, new name)",
            "run_ffmpeg_commands": "Execute one or more raw ffmpeg commands sequentially. Each command string should contain only the ffmpeg arguments (the 'ffmpeg -y' prefix is added automatically). Use this for any ffmpeg operation not covered by the other specialized tools.",
        }
        
        for name, params in {
            "get_video_info": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]},
            "trim_video": {"type": "object", "properties": {"input_path": {"type": "string"}, "output_path": {"type": "string"}, "start_time": {"type": "number"}, "duration": {"type": "number"}}, "required": ["input_path", "output_path"]},
            "resize_video": {"type": "object", "properties": {"input_path": {"type": "string"}, "output_path": {"type": "string"}, "width": {"type": "integer"}, "height": {"type": "integer"}}, "required": ["input_path", "output_path", "width", "height"]},
            "change_fps": {"type": "object", "properties": {"input_path": {"type": "string"}, "output_path": {"type": "string"}, "fps": {"type": "number"}}, "required": ["input_path", "output_path", "fps"]},
            "change_codec": {"type": "object", "properties": {"input_path": {"type": "string"}, "output_path": {"type": "string"}, "codec": {"type": "string"}, "crf": {"type": "integer"}}, "required": ["input_path", "output_path"]},
            "extract_audio": {"type": "object", "properties": {"input_path": {"type": "string"}, "output_path": {"type": "string"}, "format": {"type": "string"}}, "required": ["input_path", "output_path"]},
            "concatenate_videos": {"type": "object", "properties": {"input_paths": {"type": "array", "items": {"type": "string"}}, "output_path": {"type": "string"}}, "required": ["input_paths", "output_path"]},
            "add_subtitle": {"type": "object", "properties": {"input_path": {"type": "string"}, "output_path": {"type": "string"}, "subtitle_path": {"type": "string"}}, "required": ["input_path", "output_path", "subtitle_path"]},
            "add_watermark": {"type": "object", "properties": {"input_path": {"type": "string"}, "output_path": {"type": "string"}, "watermark_path": {"type": "string"}, "position": {"type": "string"}}, "required": ["input_path", "output_path", "watermark_path"]},
            "change_speed": {"type": "object", "properties": {"input_path": {"type": "string"}, "output_path": {"type": "string"}, "speed": {"type": "number"}}, "required": ["input_path", "output_path"]},
            "reverse_video": {"type": "object", "properties": {"input_path": {"type": "string"}, "output_path": {"type": "string"}}, "required": ["input_path", "output_path"]},
            "create_gif": {"type": "object", "properties": {"input_path": {"type": "string"}, "output_path": {"type": "string"}, "fps": {"type": "integer"}, "width": {"type": "integer"}}, "required": ["input_path", "output_path"]},
            "extract_frames": {"type": "object", "properties": {"input_path": {"type": "string"}, "output_dir": {"type": "string"}, "fps": {"type": "integer"}}, "required": ["input_path", "output_dir"]},
            "adjust_brightness": {"type": "object", "properties": {"input_path": {"type": "string"}, "output_path": {"type": "string"}, "brightness": {"type": "number"}}, "required": ["input_path", "output_path"]},
            "adjust_contrast": {"type": "object", "properties": {"input_path": {"type": "string"}, "output_path": {"type": "string"}, "contrast": {"type": "number"}}, "required": ["input_path", "output_path"]},
            "color_correct": {"type": "object", "properties": {"input_path": {"type": "string"}, "output_path": {"type": "string"}, "saturation": {"type": "number"}, "gamma": {"type": "number"}, "temperature": {"type": "number"}}, "required": ["input_path", "output_path"]},
            "vignette": {"type": "object", "properties": {"input_path": {"type": "string"}, "output_path": {"type": "string"}, "angle": {"type": "number"}}, "required": ["input_path", "output_path"]},
            "blur": {"type": "object", "properties": {"input_path": {"type": "string"}, "output_path": {"type": "string"}, "radius": {"type": "integer"}}, "required": ["input_path", "output_path"]},
            "sharpen": {"type": "object", "properties": {"input_path": {"type": "string"}, "output_path": {"type": "string"}, "amount": {"type": "number"}}, "required": ["input_path", "output_path"]},
            "rotate": {"type": "object", "properties": {"input_path": {"type": "string"}, "output_path": {"type": "string"}, "angle": {"type": "number"}}, "required": ["input_path", "output_path"]},
            "flip": {"type": "object", "properties": {"input_path": {"type": "string"}, "output_path": {"type": "string"}, "direction": {"type": "string"}}, "required": ["input_path", "output_path"]},
            "crop": {"type": "object", "properties": {"input_path": {"type": "string"}, "output_path": {"type": "string"}, "x": {"type": "integer"}, "y": {"type": "integer"}, "width": {"type": "integer"}, "height": {"type": "integer"}}, "required": ["input_path", "output_path"]},
            "pad": {"type": "object", "properties": {"input_path": {"type": "string"}, "output_path": {"type": "string"}, "width": {"type": "integer"}, "height": {"type": "integer"}}, "required": ["input_path", "output_path", "width", "height"]},
            "fade_in_out": {"type": "object", "properties": {"input_path": {"type": "string"}, "output_path": {"type": "string"}, "fade_in": {"type": "number"}, "fade_out": {"type": "number"}}, "required": ["input_path", "output_path"]},
            "add_text": {"type": "object", "properties": {"input_path": {"type": "string"}, "output_path": {"type": "string"}, "text": {"type": "string"}, "font_size": {"type": "integer"}, "font_color": {"type": "string"}}, "required": ["input_path", "output_path", "text"]},
            "set_volume": {"type": "object", "properties": {"input_path": {"type": "string"}, "output_path": {"type": "string"}, "volume": {"type": "number"}}, "required": ["input_path", "output_path"]},
            "add_fade_audio": {"type": "object", "properties": {"input_path": {"type": "string"}, "output_path": {"type": "string"}, "fade_in": {"type": "number"}, "fade_out": {"type": "number"}}, "required": ["input_path", "output_path"]},
            "normalize_audio": {"type": "object", "properties": {"input_path": {"type": "string"}, "output_path": {"type": "string"}, "loudness": {"type": "number"}}, "required": ["input_path", "output_path"]},
            "convert_format": {"type": "object", "properties": {"input_path": {"type": "string"}, "output_path": {"type": "string"}, "format": {"type": "string"}}, "required": ["input_path", "output_path"]},
            "to_grayscale": {"type": "object", "properties": {"input_path": {"type": "string"}, "output_path": {"type": "string"}}, "required": ["input_path", "output_path"]},
            "sepia": {"type": "object", "properties": {"input_path": {"type": "string"}, "output_path": {"type": "string"}}, "required": ["input_path", "output_path"]},
            "detect_idle_segments": {"type": "object", "properties": {"input_path": {"type": "string"}, "noise_threshold": {"type": "number", "description": "Pixel noise threshold for freeze detection (default 0.003, lower = stricter)"}, "min_duration": {"type": "number", "description": "Minimum freeze duration in seconds to report (default 2.0)"}}, "required": ["input_path"]},
            "speed_idle_frames": {"type": "object", "properties": {"input_path": {"type": "string"}, "output_path": {"type": "string"}, "speed": {"type": "number", "description": "Speed multiplier for idle segments (e.g. 6.0 = 6x faster)"}, "noise_threshold": {"type": "number", "description": "Pixel noise threshold for freeze detection (default 0.003)"}, "min_duration": {"type": "number", "description": "Minimum freeze duration in seconds (default 2.0)"}}, "required": ["input_path", "output_path"]},
            "generate_screen_captions": {"type": "object", "properties": {"input_path": {"type": "string"}, "output_path": {"type": "string"}, "language": {"type": "string"}}, "required": ["input_path", "output_path"]},
            "generate_audio_captions": {"type": "object", "properties": {"input_path": {"type": "string"}, "output_path": {"type": "string"}, "model": {"type": "string"}}, "required": ["input_path", "output_path"]},
            "generate_captions": {"type": "object", "properties": {"input_path": {"type": "string"}, "output_path": {"type": "string"}, "prefer": {"type": "string"}}, "required": ["input_path", "output_path"]},
            "embed_captions": {"type": "object", "properties": {"input_path": {"type": "string"}, "output_path": {"type": "string"}, "caption_source": {"type": "string"}}, "required": ["input_path", "output_path"]},
            "read_file": {"type": "object", "properties": {"file_path": {"type": "string"}, "encoding": {"type": "string"}, "max_bytes": {"type": "integer"}}, "required": ["file_path"]},
            "read_folder": {"type": "object", "properties": {"folder_path": {"type": "string"}, "recursive": {"type": "boolean"}, "pattern": {"type": "string"}}, "required": ["folder_path"]},
            "file_info": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]},
            "delete_file": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]},
            "delete_folder": {"type": "object", "properties": {"folder_path": {"type": "string"}}, "required": ["folder_path"]},
            "create_directory": {"type": "object", "properties": {"folder_path": {"type": "string"}, "exist_ok": {"type": "boolean"}}, "required": ["folder_path"]},
            "move_file": {"type": "object", "properties": {"source": {"type": "string"}, "destination": {"type": "string"}}, "required": ["source", "destination"]},
            "copy_file": {"type": "object", "properties": {"source": {"type": "string"}, "destination": {"type": "string"}}, "required": ["source", "destination"]},
            "file_exists": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            "rename_file": {"type": "object", "properties": {"file_path": {"type": "string"}, "new_name": {"type": "string"}}, "required": ["file_path", "new_name"]},
            "run_ffmpeg_commands": {"type": "object", "properties": {"commands": {"type": "array", "items": {"type": "string"}, "description": "List of ffmpeg argument strings to execute sequentially. Do NOT include 'ffmpeg' or '-y' — they are added automatically. Example: [\"-i input.mp4 -vf scale=1280:720 output.mp4\"]"}}, "required": ["commands"]},
        }.items():
            tools.append(ToolDefinition(
                type="function",
                function={
                    "name": name,
                    "description": tool_descriptions.get(name, ""),
                    "parameters": params
                }
            ))
        return tools

    SYSTEM_PROMPT = """You are an autonomous video editing assistant. Use the available tools to fulfill the user's request step by step.

Workflow rules:
- Think about what needs to happen, then call the appropriate tool(s).
- After each tool result, decide whether more tool calls are required.
- When the user's task is fully complete, reply in plain text WITHOUT any tool calls and clearly summarize what was done (include output file paths).
- If something fails, explain the failure and either retry with corrected arguments or ask the user for clarification.
- Prefer calling get_video_info first when you need details about an unknown input file.
- Always use absolute or workspace-relative paths exactly as provided by the user.

Available tools include video probing, trimming, resizing, fps/codec changes, audio extraction, concatenation, watermarks, speed/reverse, gif/frame extraction, color/brightness/contrast/vignette/blur/sharpen, rotate/flip/crop/pad, fades, text overlay, audio volume/fade/normalize, format conversion, grayscale/sepia, idle frame detection and speed-up, caption generation/embedding, file operations (read, list, delete to recycle bin, copy, move, create directory), and a generic run_ffmpeg_commands tool for executing any raw ffmpeg commands not covered by other tools."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.ollama = OllamaClient(settings=self.settings)
        self.video_tool = VideoTool(self.settings)
        self.file_tool = FileTool(self.settings)
        self.ffmpeg_exec_tool = FFmpegExecTool(self.settings)
        self._tool_registry: dict[str, Callable] = {}
        self._register_tools()
        self.messages: list[Message] = [Message(role="system", content=self.SYSTEM_PROMPT)]
        self.max_iterations: int = 15
        logger.info("VideoEditorAgent initialized (model=%s, max_iterations=%d)", self.settings.default_model, self.max_iterations)

    def reset(self) -> None:
        """Clear conversation history (keeps system prompt)."""
        self.messages = [Message(role="system", content=self.SYSTEM_PROMPT)]

    async def process_prompt(
        self,
        prompt: str,
        on_tool_call: Callable[[str, dict, dict], None] | None = None,
        on_assistant: Callable[[str], None] | None = None,
    ) -> str:
        """Run an agent loop: chat -> execute tool calls -> feed results back -> repeat
        until the model produces a final assistant message with no tool calls.
        Returns the final assistant text."""
        self.messages.append(Message(role="user", content=prompt))
        tools = self.get_tools()
        logger.info("Processing prompt: %s", prompt[:100])

        for iteration in range(self.max_iterations):
            logger.debug("Agent iteration %d/%d", iteration + 1, self.max_iterations)
            response = await self.ollama.chat(self.messages, tools=tools)
            msg = response.message
            tool_calls = getattr(msg, "tool_calls", None) or []

            # Persist the assistant turn (with any tool_calls) into history.
            self.messages.append(Message(
                role="assistant",
                content=msg.content or "",
                tool_calls=tool_calls or None,
            ))

            if not tool_calls:
                final = msg.content or ""
                logger.info("Agent finished with final response (length=%d)", len(final))
                if on_assistant:
                    on_assistant(final)
                return final

            for tc in tool_calls:
                fn = tc.function if isinstance(tc.function, dict) else {}
                tool_name = fn.get("name")
                raw_args = fn.get("arguments", {})
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        args = self._parse_arguments(raw_args)
                else:
                    args = dict(raw_args) if raw_args else {}

                result = self._execute_tool(tool_name, args)
                if on_tool_call:
                    on_tool_call(tool_name or "", args, result)

                self.messages.append(Message(
                    role="tool",
                    content=json.dumps(result, default=str),
                    tool_call_id=tool_name,
                ))

        warning = f"[agent] Stopped after {self.max_iterations} iterations without a final answer."
        logger.warning(warning)
        if on_assistant:
            on_assistant(warning)
        return warning

    def _execute_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        logger.info("Executing tool: %s", tool_name)
        logger.debug("Tool args: %s", json.dumps(args, default=str))
        if tool_name in self._tool_registry:
            try:
                result = self._tool_registry[tool_name](**args)
                if hasattr(result, "returncode"):
                    tool_result = {
                        "success": result.returncode == 0,
                        "returncode": result.returncode,
                        "stdout": (result.stdout or "")[-500:] if hasattr(result, "stdout") else "",
                        "stderr": (result.stderr or "")[-500:] if hasattr(result, "stderr") else "",
                    }
                else:
                    tool_result = {"success": True, "output": result}
                logger.debug("Tool %s result: success=%s", tool_name, tool_result.get("success"))
                return tool_result
            except Exception as e:
                logger.error("Tool %s failed: %s", tool_name, e)
                return {"success": False, "error": str(e)}
        logger.warning("Tool not found: %s", tool_name)
        return {"success": False, "error": f"Tool {tool_name} not found"}

    def _parse_arguments(self, arg_str: str) -> dict[str, Any]:
        args = {}
        patterns = {
            "input_path": r"input[:\s]*([^\s]+)",
            "output_path": r"output[:\s]*([^\s]+)",
            "file_path": r"file[:\s]*([^\s]+)",
            "start_time": r"start[_\s]*time[:\s]*(\d+\.?\d*)",
            "duration": r"duration[:\s]*(\d+\.?\d*)",
            "width": r"width[:\s]*(\d+)",
            "height": r"height[:\s]*(\d+)",
            "fps": r"fps[:\s]*(\d+\.?\d*)",
            "speed": r"speed[:\s]*(\d+\.?\d*)",
            "codec": r"codec[:\s]*([^\s]+)",
            "crf": r"crf[:\s]*(\d+)",
            "format": r"format[:\s]*([^\s]+)",
            "brightness": r"brightness[:\s]*(-?\d+\.?\d*)",
            "contrast": r"contrast[:\s]*(\d+\.?\d*)"
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, arg_str, re.IGNORECASE)
            if match:
                value = match.group(1)
                if key in ["width", "height", "fps", "duration", "start_time", "crf"]:
                    value = float(value) if "." in value else int(value)
                args[key] = value
        return args