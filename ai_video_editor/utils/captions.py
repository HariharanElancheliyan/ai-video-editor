import base64
import subprocess
import re
from pathlib import Path
from typing import Any

# Prefer faster-whisper (CTranslate2). Fall back to openai-whisper.
FASTER_WHISPER_AVAILABLE = False
FASTER_WHISPER_IMPORT_ERROR: str | None = None
try:
    from faster_whisper import WhisperModel as _FasterWhisperModel  # type: ignore
    FASTER_WHISPER_AVAILABLE = True
except Exception as _e:  # noqa: BLE001
    FASTER_WHISPER_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"

WHISPER_IMPORT_ERROR: str | None = None
try:
    import whisper  # type: ignore
    WHISPER_AVAILABLE = True
except Exception as _e:  # noqa: BLE001
    WHISPER_AVAILABLE = False
    WHISPER_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"


def _detect_torch_device() -> str:
    """Return best available torch device string: 'cuda', 'dml' (AMD/Intel on Windows), or 'cpu'."""
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    try:
        import torch_directml  # type: ignore  # noqa: F401
        return "dml"
    except Exception:
        pass
    return "cpu"


def _detect_ct2_device() -> tuple[str, str]:
    """Return (device, compute_type) for faster-whisper / CTranslate2."""
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def _text_similarity(a: str, b: str) -> float:
    """Return 0.0-1.0 similarity ratio between two strings (word-level Jaccard)."""
    if not a or not b:
        return 0.0
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


class _LLMCaptioner:
    """Generates captions for video frames using an Ollama vision model.

    Sends each frame image directly to a vision-capable LLM to produce a
    concise one-line caption. If Ollama is unreachable, methods return empty
    strings.
    """

    _VISION_SYSTEM = (
        "You are a precise video captioner. You see a single frame "
        "from a video and must produce ONE short, SPECIFIC caption "
        "describing exactly what is visible.\n\n"
        "CRITICAL RULES:\n"
        "- Output ONLY the caption text. No quotes, no prefixes, no explanations.\n"
        "- Maximum 15 words / 120 characters, single line.\n"
        "- Be SPECIFIC: mention actual UI elements, app names, filenames, text, "
        "buttons, dialogs, objects, people, or scenes visible.\n"
        "- Focus on what makes THIS frame unique compared to a typical frame.\n"
        "- If a PREVIOUS CAPTION is provided, describe what CHANGED.\n"
        "- If the image is blank or unclear, reply with an empty string."
    )

    def __init__(self, vision_model: str | None = None, host: str | None = None):
        self.ok = False
        self._client = None
        self._vision_model = vision_model
        try:
            from ..config.settings import Settings
            import ollama  # type: ignore

            settings = Settings()
            self._vision_model = vision_model or settings.vision_model or settings.default_model
            self._client = ollama.Client(host=host or settings.ollama_base_url)
            # Verify the model is available
            self._client.show(self._vision_model)
            self.ok = True
        except Exception:
            self.ok = False

    def _parse_response(self, resp: object, max_chars: int) -> str:
        """Extract and clean caption text from an Ollama response."""
        msg = resp.get("message") if isinstance(resp, dict) else getattr(resp, "message", None)
        content = ""
        if isinstance(msg, dict):
            content = msg.get("content", "") or ""
        elif msg is not None:
            content = getattr(msg, "content", "") or ""
        content = content.strip().strip('"').strip("'")
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        first = next((ln.strip() for ln in content.splitlines() if ln.strip()), "")
        if len(first) > max_chars:
            first = first[:max_chars].rstrip()
        return first

    def describe_frame(self, frame_path: str, max_chars: int = 120,
                       prev_caption: str = "", timestamp: str = "") -> str:
        """Use the vision model to describe a frame image directly."""
        if not self.ok or self._client is None:
            return ""
        try:
            with open(frame_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")

            user_parts = []
            if prev_caption:
                user_parts.append(f"PREVIOUS CAPTION: {prev_caption}")
            if timestamp:
                user_parts.append(f"TIMESTAMP: {timestamp}")
            user_parts.append("Describe what is shown in this frame.")
            user_content = "\n".join(user_parts)

            resp = self._client.chat(
                model=self._vision_model,
                messages=[
                    {"role": "system", "content": self._VISION_SYSTEM},
                    {"role": "user", "content": user_content, "images": [img_b64]},
                ],
                options={"temperature": 0.3, "num_predict": 60},
            )
            return self._parse_response(resp, max_chars)
        except Exception:
            return ""


class CaptionGenerator:
    _whisper_cache: dict = {}

    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe"):
        self.ffmpeg = ffmpeg_path
        self.ffprobe = ffprobe_path

    def extract_frames(self, input_path, output_dir, fps: int = 1) -> list[str]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [self.ffmpeg, "-y", "-i", str(input_path), "-vf", f"fps={fps}", str(output_dir / "frame_%04d.png")]
        subprocess.run(cmd, capture_output=True)
        return sorted([str(f) for f in output_dir.glob("frame_*.png")])

    def screen_capture_to_text(
        self,
        input_path,
        output_path=None,
        language: str = "en",
        fps: int = 1,
        summary_model: str | None = None,
        ollama_host: str | None = None,
        max_caption_chars: int = 120,
        vision_model: str | None = None,
    ):
        """Extract frames and generate captions using an LLM vision model.

        Each frame is sent directly to the vision model to produce a concise
        one-line description. Duplicate/similar captions are automatically
        de-duplicated.
        """
        captioner = _LLMCaptioner(
            vision_model=vision_model or summary_model,
            host=ollama_host,
        )
        if not captioner.ok:
            return {"error": "Vision model not available. Ensure Ollama is running and a vision model is configured."}

        temp_dir = Path("temp_caption_frames")
        frames = self.extract_frames(input_path, temp_dir, fps)
        if not frames:
            return {"error": "No frames extracted"}

        captions = []
        prev_caption = ""
        for i, frame in enumerate(frames):
            try:
                start, end = self._frame_to_timestamp(fps, i)
                timestamp = start.split(",")[0]  # HH:MM:SS without ms

                caption_text = captioner.describe_frame(
                    frame, max_chars=max_caption_chars,
                    prev_caption=prev_caption, timestamp=timestamp,
                )

                if not caption_text:
                    continue

                # Skip if new caption is too similar to previous
                if prev_caption and _text_similarity(caption_text, prev_caption) > 0.75:
                    continue
                prev_caption = caption_text

                captions.append({
                    "start": start,
                    "end": end,
                    "text": caption_text,
                })
            except Exception:
                continue

        for f in frames:
            Path(f).unlink(missing_ok=True)
        try:
            temp_dir.rmdir()
        except OSError:
            pass

        if output_path:
            self._save_srt(captions, output_path)

        return {
            "captions": captions,
            "count": len(captions),
            "output": str(output_path) if output_path else None,
        }

    def audio_to_text(self, input_path, output_path=None, model: str = "base", language=None):
        """Transcribe audio. Uses faster-whisper (CUDA or CPU int8) when available,
        otherwise falls back to openai-whisper (CUDA / DirectML / CPU)."""
        # ---- Preferred path: faster-whisper ----
        if FASTER_WHISPER_AVAILABLE:
            try:
                device, compute_type = _detect_ct2_device()
                cache_key = (model, device, compute_type)
                model_obj = self._whisper_cache.get(cache_key)
                if model_obj is None:
                    model_obj = _FasterWhisperModel(model, device=device, compute_type=compute_type)
                    self._whisper_cache[cache_key] = model_obj
                seg_iter, info = model_obj.transcribe(str(input_path), language=language)
                segments = []
                full_text_parts = []
                for seg in seg_iter:
                    segments.append({
                        "start": self._format_timestamp(seg.start),
                        "end": self._format_timestamp(seg.end),
                        "text": seg.text.strip(),
                    })
                    full_text_parts.append(seg.text)
                srt_output = output_path or str(Path(str(input_path).replace(Path(input_path).suffix, ".srt")))
                self._save_srt(segments, srt_output)
                return {
                    "text": "".join(full_text_parts).strip(),
                    "segments": segments,
                    "count": len(segments),
                    "output": srt_output,
                    "backend": f"faster-whisper/{device}/{compute_type}",
                }
            except Exception as e:
                faster_err = str(e)
        else:
            faster_err = FASTER_WHISPER_IMPORT_ERROR

        # ---- Fallback: openai-whisper ----
        if not WHISPER_AVAILABLE:
            return {
                "error": (
                    "No Whisper backend available. Install one of:\n"
                    "  pip install faster-whisper   (recommended; AMD+NVIDIA)\n"
                    "  pip install openai-whisper"
                ),
                "faster_whisper_error": faster_err,
                "whisper_error": WHISPER_IMPORT_ERROR,
            }
        try:
            device = _detect_torch_device()
            load_device: Any = device
            if device == "dml":
                try:
                    import torch_directml  # type: ignore
                    load_device = torch_directml.device()
                except Exception:
                    load_device = "cpu"
                    device = "cpu"
            cache_key = ("openai", model, str(device))
            model_obj = self._whisper_cache.get(cache_key)
            if model_obj is None:
                model_obj = whisper.load_model(model, device=load_device)
                self._whisper_cache[cache_key] = model_obj
            use_fp16 = device == "cuda"
            result = model_obj.transcribe(str(input_path), language=language, fp16=use_fp16)
            segments = []
            for seg in result.get("segments", []):
                segments.append({
                    "start": self._format_timestamp(seg["start"]),
                    "end": self._format_timestamp(seg["end"]),
                    "text": seg["text"].strip(),
                })
            srt_output = output_path or str(Path(str(input_path).replace(Path(input_path).suffix, ".srt")))
            self._save_srt(segments, srt_output)
            return {
                "text": result["text"],
                "segments": segments,
                "count": len(segments),
                "output": srt_output,
                "backend": f"openai-whisper/{device}",
            }
        except Exception as e:
            return {"error": str(e), "faster_whisper_error": faster_err}

    def screen_and_audio_captions(self, input_path, output_path, prefer: str = "audio", fps: int = 1, whisper_model: str = "base"):
        audio_ok = FASTER_WHISPER_AVAILABLE or WHISPER_AVAILABLE
        if prefer == "audio" and audio_ok:
            return self.audio_to_text(input_path, output_path, whisper_model)
        elif prefer == "screen":
            return self.screen_capture_to_text(input_path, output_path, fps=fps)
        elif audio_ok:
            return self.audio_to_text(input_path, output_path, whisper_model)
        return self.screen_capture_to_text(input_path, output_path, fps=fps)

    def embed_subtitles(self, input_path, output_path, subtitle_path, font: str = "Arial", font_size: int = 24):
        from .ffmpeg_utils import escape_filter_path
        escaped = escape_filter_path(subtitle_path)
        cmd = [self.ffmpeg, "-y", "-i", str(input_path), "-vf", f"subtitles='{escaped}'", str(output_path)]
        return subprocess.run(cmd, capture_output=True, text=True)

    def _frame_to_timestamp(self, fps: int, frame_num: int):
        start_sec = frame_num / fps
        end_sec = (frame_num + 1) / fps
        return self._format_timestamp(start_sec), self._format_timestamp(end_sec)

    def _format_timestamp(self, seconds: float) -> str:
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"

    def _save_srt(self, captions: list, output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            for i, cap in enumerate(captions, 1):
                f.write(f"{i}\n")
                f.write(f"{cap['start']} --> {cap['end']}\n")
                f.write(f"{cap['text']}\n\n")


def has_whisper_support() -> bool:
    return FASTER_WHISPER_AVAILABLE or WHISPER_AVAILABLE


def active_device() -> dict:
    """Diagnostics: which device will be used for captions."""
    ct2_device, ct2_compute = _detect_ct2_device()
    return {
        "torch_device": _detect_torch_device(),
        "faster_whisper": {
            "available": FASTER_WHISPER_AVAILABLE,
            "device": ct2_device,
            "compute_type": ct2_compute,
            "error": FASTER_WHISPER_IMPORT_ERROR,
        },
        "openai_whisper": {
            "available": WHISPER_AVAILABLE,
            "error": WHISPER_IMPORT_ERROR,
        },
    }
