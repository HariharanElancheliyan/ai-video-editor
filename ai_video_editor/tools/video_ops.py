import logging
from pathlib import Path
from typing import Any
import re
import subprocess
import tempfile
from ..utils.ffmpeg_utils import FFmpegUtils, escape_filter_path
from ..utils.captions import (
    CaptionGenerator,
    has_whisper_support,
)
from ..config.settings import Settings

logger = logging.getLogger(__name__)


class VideoTool:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.ffmpeg = FFmpegUtils(self.settings.ffmpeg_path, self.settings.ffprobe_path)
        logger.info("VideoTool initialized (ffmpeg=%s)", self.settings.ffmpeg_path)

    def get_info(self, file_path: str | Path) -> dict[str, Any]:
        logger.debug("get_info: %s", file_path)
        return self.ffmpeg.probe(file_path)

    def trim(
        self,
        input_path: str | Path,
        output_path: str | Path,
        start_time: float = 0,
        duration: float | None = None
    ) -> dict[str, Any]:
        logger.debug("trim: %s -> %s (start=%.2f, duration=%s)", input_path, output_path, start_time, duration)
        args = ["-ss", str(start_time)]
        if duration:
            args.extend(["-t", str(duration)])
        args.extend(["-i", str(input_path), "-c", "copy", str(output_path)])
        return self.ffmpeg.run(args)

    def resize(
        self,
        input_path: str | Path,
        output_path: str | Path,
        width: int,
        height: int
    ) -> dict[str, Any]:
        args = [
            "-i", str(input_path),
            "-vf", f"scale={width}:{height}",
            "-c:a", "copy",
            str(output_path)
        ]
        return self.ffmpeg.run(args)

    def change_fps(
        self,
        input_path: str | Path,
        output_path: str | Path,
        fps: float
    ) -> dict[str, Any]:
        args = [
            "-i", str(input_path),
            "-r", str(fps),
            "-c:a", "copy",
            str(output_path)
        ]
        return self.ffmpeg.run(args)

    def change_codec(
        self,
        input_path: str | Path,
        output_path: str | Path,
        codec: str = "libx264",
        crf: int = 23,
        preset: str = "medium"
    ) -> dict[str, Any]:
        args = [
            "-i", str(input_path),
            "-c:v", codec,
            "-crf", str(crf),
            "-preset", preset,
            "-c:a", "aac",
            "-b:a", "128k",
            str(output_path)
        ]
        return self.ffmpeg.run(args)

    def extract_audio(
        self,
        input_path: str | Path,
        output_path: str | Path,
        format: str = "mp3",
        bitrate: str = "192k"
    ) -> dict[str, Any]:
        args = [
            "-i", str(input_path),
            "-vn",
            "-ab", bitrate,
            "-f", format,
            str(output_path)
        ]
        return self.ffmpeg.run(args)

    def concatenate(
        self,
        input_paths: list[str | Path],
        output_path: str | Path,
        method: str = "concat"
    ) -> dict[str, Any]:
        if method == "concat":
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                for p in input_paths:
                    f.write(f"file '{p}'\n")
                list_file = Path(f.name)
            try:
                args = ["-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(output_path)]
                result = self.ffmpeg.run(args)
            finally:
                list_file.unlink(missing_ok=True)
            return result
        return {"error": "Unsupported concat method"}

    def add_subtitle(
        self,
        input_path: str | Path,
        output_path: str | Path,
        subtitle_path: str | Path
    ) -> dict[str, Any]:
        escaped = escape_filter_path(subtitle_path)
        args = [
            "-i", str(input_path),
            "-vf", f"subtitles='{escaped}'",
            "-c:a", "copy",
            str(output_path)
        ]
        return self.ffmpeg.run(args)

    def add_watermark(
        self,
        input_path: str | Path,
        output_path: str | Path,
        watermark_path: str | Path,
        position: str = "top-right",
        opacity: float = 1.0
    ) -> dict[str, Any]:
        overlay_pos = {
            "top-left": "10:10",
            "top-right": "W-w-10:10",
            "bottom-left": "10:H-h-10",
            "bottom-right": "W-w-10:H-h-10",
        }.get(position, "W-w-10:10")
        
        args = [
            "-i", str(input_path),
            "-i", str(watermark_path),
            "-filter_complex", f"overlay={overlay_pos}:format=auto",
            str(output_path)
        ]
        return self.ffmpeg.run(args)

    def change_speed(
        self,
        input_path: str | Path,
        output_path: str | Path,
        speed: float = 1.0
    ) -> dict[str, Any]:
        if speed <= 0:
            return {"error": "Speed must be positive"}
        # Build atempo chain: ffmpeg atempo only supports [0.5, 100.0],
        # and for accuracy values outside [0.5, 2.0] should be chained.
        atempo_parts = []
        remaining = speed
        while remaining > 2.0:
            atempo_parts.append("atempo=2.0")
            remaining /= 2.0
        while remaining < 0.5:
            atempo_parts.append("atempo=0.5")
            remaining /= 0.5
        atempo_parts.append(f"atempo={remaining}")
        args = [
            "-i", str(input_path),
            "-vf", f"setpts={1/speed}*PTS",
            "-af", ",".join(atempo_parts),
            str(output_path)
        ]
        return self.ffmpeg.run(args)

    def reverse(self, input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
        args = [
            "-i", str(input_path),
            "-vf", "reverse",
            "-af", "areverse",
            str(output_path)
        ]
        return self.ffmpeg.run(args)

    def extract_frames(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        fps: int = 1,
        start_time: float = 0,
        duration: float | None = None
    ) -> dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        args = []
        if start_time > 0:
            args.extend(["-ss", str(start_time)])
        args.extend(["-i", str(input_path)])
        if duration is not None:
            args.extend(["-t", str(duration)])
        args.extend([
            "-vf", f"fps={fps}",
            str(output_dir / "frame_%04d.png")
        ])
        return self.ffmpeg.run(args)

    def create_gif(
        self,
        input_path: str | Path,
        output_path: str | Path,
        fps: int = 15,
        width: int = 480,
        start_time: float = 0,
        duration: float | None = None
    ) -> dict[str, Any]:
        args = [
            "-i", str(input_path),
            "-ss", str(start_time),
        ]
        if duration:
            args.extend(["-t", str(duration)])
        args.extend([
            "-vf", f"fps={fps},scale={width}:-1:flags=lanczos",
            "-f", "gif",
            str(output_path)
        ])
        return self.ffmpeg.run(args)

    def stabilize(
        self,
        input_path: str | Path,
        output_path: str | Path
    ) -> dict[str, Any]:
        args = [
            "-i", str(input_path),
            "-vf", "deshake",
            "-c:a", "copy",
            str(output_path)
        ]
        return self.ffmpeg.run(args)

    def adjust_brightness(
        self,
        input_path: str | Path,
        output_path: str | Path,
        brightness: float = 0
    ) -> dict[str, Any]:
        args = [
            "-i", str(input_path),
            "-vf", f"eq=brightness={brightness}",
            "-c:a", "copy",
            str(output_path)
        ]
        return self.ffmpeg.run(args)

    def adjust_contrast(
        self,
        input_path: str | Path,
        output_path: str | Path,
        contrast: float = 1.0
    ) -> dict[str, Any]:
        args = [
            "-i", str(input_path),
            "-vf", f"eq=contrast={contrast}",
            "-c:a", "copy",
            str(output_path)
        ]
        return self.ffmpeg.run(args)

    def color_correct(
        self,
        input_path: str | Path,
        output_path: str | Path,
        saturation: float = 1.0,
        gamma: float = 1.0,
        temperature: float = 0,
        tint: float = 0
    ) -> dict[str, Any]:
        filters = [f"eq=saturation={saturation}:gamma={gamma}"]
        if temperature != 0:
            filters.append(f"colortemperature=temperature={6500 + temperature * 100}")
        if tint != 0:
            filters.append(f"colorbalance=rs={tint}:gs={tint}:bs={tint}")
        args = [
            "-i", str(input_path),
            "-vf", ",".join(filters),
            "-c:a", "copy",
            str(output_path)
        ]
        return self.ffmpeg.run(args)

    def vignette(
        self,
        input_path: str | Path,
        output_path: str | Path,
        angle: float = 0.5,
        smoothness: float = 0.5
    ) -> dict[str, Any]:
        args = [
            "-i", str(input_path),
            "-vf", f"vignette=angle={angle}:mode=forward",
            "-c:a", "copy",
            str(output_path)
        ]
        return self.ffmpeg.run(args)

    def blur(
        self,
        input_path: str | Path,
        output_path: str | Path,
        radius: int = 5,
        power: int = 2
    ) -> dict[str, Any]:
        args = [
            "-i", str(input_path),
            "-vf", f"boxblur={radius}:{power}",
            "-c:a", "copy",
            str(output_path)
        ]
        return self.ffmpeg.run(args)

    def sharpen(
        self,
        input_path: str | Path,
        output_path: str | Path,
        amount: float = 1.0,
        radius: float = 1.0
    ) -> dict[str, Any]:
        args = [
            "-i", str(input_path),
            "-vf", f"unsharp=5:5:{amount}:5:5:{radius}",
            "-c:a", "copy",
            str(output_path)
        ]
        return self.ffmpeg.run(args)

    def rotate(
        self,
        input_path: str | Path,
        output_path: str | Path,
        angle: float = 90,
        fill_color: str = "black"
    ) -> dict[str, Any]:
        args = [
            "-i", str(input_path),
            "-vf", f"rotate={angle}*PI/180:fillcolor={fill_color}",
            "-c:a", "copy",
            str(output_path)
        ]
        return self.ffmpeg.run(args)

    def flip(
        self,
        input_path: str | Path,
        output_path: str | Path,
        direction: str = "horizontal"
    ) -> dict[str, Any]:
        vf = "hflip" if direction == "horizontal" else "vflip"
        args = [
            "-i", str(input_path),
            "-vf", vf,
            "-c:a", "copy",
            str(output_path)
        ]
        return self.ffmpeg.run(args)

    def crop(
        self,
        input_path: str | Path,
        output_path: str | Path,
        x: int = 0,
        y: int = 0,
        width: int | None = None,
        height: int | None = None
    ) -> dict[str, Any]:
        if not width or not height:
            return {"error": "width and height are required for crop"}
        crop_str = f"{width}:{height}:{x}:{y}"
        args = [
            "-i", str(input_path),
            "-vf", f"crop={crop_str}",
            "-c:a", "copy",
            str(output_path)
        ]
        return self.ffmpeg.run(args)

    def pad(
        self,
        input_path: str | Path,
        output_path: str | Path,
        width: int,
        height: int,
        x: int = 0,
        y: int = 0,
        color: str = "black"
    ) -> dict[str, Any]:
        args = [
            "-i", str(input_path),
            "-vf", f"pad={width}:{height}:{x}:{y}:color={color}",
            "-c:a", "copy",
            str(output_path)
        ]
        return self.ffmpeg.run(args)

    def fade_in_out(
        self,
        input_path: str | Path,
        output_path: str | Path,
        fade_in: float = 0,
        fade_out: float = 0,
        color: str = "black"
    ) -> dict[str, Any]:
        filters = []
        if fade_in > 0:
            filters.append(f"fade=t=in:st=0:d={fade_in}")
        if fade_out > 0:
            duration = self.ffmpeg.get_duration(input_path)
            filters.append(f"fade=t=out:st={duration - fade_out}:d={fade_out}")
        
        if not filters:
            return {"error": "No fade specified"}
        
        args = [
            "-i", str(input_path),
            "-vf", ",".join(filters),
            "-c:a", "copy",
            str(output_path)
        ]
        return self.ffmpeg.run(args)

    def add_text(
        self,
        input_path: str | Path,
        output_path: str | Path,
        text: str,
        font: str = "Arial",
        font_size: int = 24,
        font_color: str = "white",
        x: str = "(w-text_w)/2",
        y: str = "(h-text_h)/2",
        shadow: int = 0
    ) -> dict[str, Any]:
        import platform
        shadow_str = f":shadowx={shadow}:shadowy={shadow}" if shadow > 0 else ""
        escaped_text = text.replace("'", r"\'").replace(":", r"\:")
        if platform.system() == "Windows":
            font_spec = f"font='{font}'"
        else:
            font_spec = "fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        args = [
            "-i", str(input_path),
            "-vf", f"drawtext=text='{escaped_text}':{font_spec}:fontsize={font_size}:fontcolor={font_color}:x={x}:y={y}{shadow_str}",
            "-c:a", "copy",
            str(output_path)
        ]
        return self.ffmpeg.run(args)

    def set_volume(
        self,
        input_path: str | Path,
        output_path: str | Path,
        volume: float = 1.0,
        volume_db: float | None = None
    ) -> dict[str, Any]:
        if volume_db is not None:
            args = ["-i", str(input_path), "-af", f"volume={volume_db}dB", str(output_path)]
        else:
            args = ["-i", str(input_path), "-af", f"volume={volume}", str(output_path)]
        return self.ffmpeg.run(args)

    def add_fade_audio(
        self,
        input_path: str | Path,
        output_path: str | Path,
        fade_in: float = 0,
        fade_out: float = 0
    ) -> dict[str, Any]:
        af = []
        if fade_in > 0:
            af.append(f"afade=t=in:st=0:d={fade_in}")
        if fade_out > 0:
            duration = self.ffmpeg.get_duration(input_path)
            af.append(f"afade=t=out:st={duration - fade_out}:d={fade_out}")
        
        if not af:
            return {"error": "No fade specified"}
        
        args = ["-i", str(input_path), "-af", ",".join(af), str(output_path)]
        return self.ffmpeg.run(args)

    def normalize_audio(
        self,
        input_path: str | Path,
        output_path: str | Path,
        loudness: float = -24.0,
        peak: float = -1.0
    ) -> dict[str, Any]:
        args = [
            "-i", str(input_path),
            "-af", f"loudnorm=I={loudness}:TP={peak}:LRA=11",
            str(output_path)
        ]
        return self.ffmpeg.run(args)

    def convert_format(
        self,
        input_path: str | Path,
        output_path: str | Path,
        format: str = "mp4",
        video_codec: str = "libx264",
        audio_codec: str = "aac",
        video_bitrate: str = "2M",
        audio_bitrate: str = "192k"
    ) -> dict[str, Any]:
        args = [
            "-i", str(input_path),
            "-c:v", video_codec,
            "-b:v", video_bitrate,
            "-c:a", audio_codec,
            "-b:a", audio_bitrate,
            str(output_path)
        ]
        return self.ffmpeg.run(args)

    def generate_screen_captions(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        language: str = "eng",
        fps: int = 1
    ) -> dict[str, Any]:
        captioner = CaptionGenerator(self.settings.ffmpeg_path, self.settings.ffprobe_path)
        return captioner.screen_capture_to_text(input_path, output_path, language, fps)

    def generate_audio_captions(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        model: str = "base",
        language: str | None = None
    ) -> dict[str, Any]:
        if not has_whisper_support():
            return {"error": "Whisper not available. Install: pip install ai-video-editor[audio]"}
        
        captioner = CaptionGenerator(self.settings.ffmpeg_path, self.settings.ffprobe_path)
        return captioner.audio_to_text(input_path, output_path, model, language)

    def generate_captions(
        self,
        input_path: str | Path,
        output_path: str | Path,
        prefer: str = "audio",
        fps: int = 1,
        whisper_model: str = "base"
    ) -> dict[str, Any]:
        captioner = CaptionGenerator(self.settings.ffmpeg_path, self.settings.ffprobe_path)
        return captioner.screen_and_audio_captions(
            input_path, output_path, prefer, fps, whisper_model
        )

    def embed_captions(
        self,
        input_path: str | Path,
        output_path: str | Path,
        caption_source: str | Path,
        create_if_missing: bool = False
    ) -> dict[str, Any]:
        caption_path = Path(caption_source)
        if not caption_path.exists() and create_if_missing:
            temp_srt = Path(str(input_path) + ".srt")
            self.generate_captions(input_path, temp_srt, prefer="audio")
            caption_path = temp_srt
        
        if not caption_path.exists():
            return {"error": "Caption file not found"}
        
        captioner = CaptionGenerator(self.settings.ffmpeg_path, self.settings.ffprobe_path)
        result = captioner.embed_subtitles(input_path, output_path, caption_path)
        
        return {"success": result.returncode == 0, "output": str(output_path)}

    def to_grayscale(
        self,
        input_path: str | Path,
        output_path: str | Path
    ) -> dict[str, Any]:
        args = [
            "-i", str(input_path),
            "-vf", "hue=s=0",
            "-c:a", "copy",
            str(output_path)
        ]
        return self.ffmpeg.run(args)

    def sepia(
        self,
        input_path: str | Path,
        output_path: str | Path
    ) -> dict[str, Any]:
        args = [
            "-i", str(input_path),
            "-vf", "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131",
            "-c:a", "copy",
            str(output_path)
        ]
        return self.ffmpeg.run(args)

    def detect_idle_segments(
        self,
        input_path: str | Path,
        noise_threshold: float = 0.003,
        min_duration: float = 2.0,
    ) -> dict[str, Any]:
        """Detect idle/frozen segments in a video using ffmpeg freezedetect.

        Returns a list of segments with start/end times and durations,
        formatted for easy consumption by the editing agent.
        """
        args = [
            "-i", str(input_path),
            "-vf", f"freezedetect=n={noise_threshold}:d={min_duration}",
            "-f", "null", "-",
        ]
        result = self.ffmpeg.run(args)
        stderr = result.stderr or ""

        # Parse freeze_start / freeze_end / freeze_duration from stderr
        starts = [float(m) for m in re.findall(r"freeze_start:\s*([\d.]+)", stderr)]
        ends = [float(m) for m in re.findall(r"freeze_end:\s*([\d.]+)", stderr)]
        durations = [float(m) for m in re.findall(r"freeze_duration:\s*([\d.]+)", stderr)]

        # Get total video duration for context
        total_duration = self.ffmpeg.get_duration(input_path)

        segments = []
        for i in range(min(len(starts), len(ends))):
            segments.append({
                "index": i + 1,
                "start": round(starts[i], 2),
                "end": round(ends[i], 2),
                "duration": round(durations[i], 2) if i < len(durations) else round(ends[i] - starts[i], 2),
                "description": f"{round(starts[i], 1)}s - {round(ends[i], 1)}s (idle for {round(ends[i] - starts[i], 1)}s)",
            })

        total_idle = sum(s["duration"] for s in segments)
        return {
            "success": True,
            "video_duration": round(total_duration, 2),
            "idle_segment_count": len(segments),
            "total_idle_time": round(total_idle, 2),
            "idle_percentage": round((total_idle / total_duration) * 100, 1) if total_duration > 0 else 0,
            "segments": segments,
            "summary": (
                f"Found {len(segments)} idle segment(s) totalling {round(total_idle, 1)}s "
                f"out of {round(total_duration, 1)}s ({round((total_idle / total_duration) * 100, 1)}% idle)"
                if total_duration > 0 else "Could not determine video duration"
            ),
        }

    def speed_idle_frames(
        self,
        input_path: str | Path,
        output_path: str | Path,
        speed: float = 6.0,
        noise_threshold: float = 0.003,
        min_duration: float = 2.0,
    ) -> dict[str, Any]:
        """Detect idle segments and speed them up by the given factor.

        Splits the video into normal and idle parts, re-encodes all parts
        uniformly, speeds up the idle parts, then concatenates everything.
        """
        if speed <= 0:
            return {"success": False, "error": "Speed must be positive"}

        # Step 1: detect idle segments
        detection = self.detect_idle_segments(input_path, noise_threshold, min_duration)
        segments = detection.get("segments", [])
        if not segments:
            # Actually copy the file so output_path exists
            import shutil
            shutil.copy2(str(input_path), str(output_path))
            return {
                "success": True,
                "message": "No idle segments detected — output is a copy of the input.",
                "output": str(output_path),
            }

        total_duration = detection["video_duration"]

        # Check if input has an audio stream
        probe = self.ffmpeg.probe(input_path)
        has_audio = any(
            s.get("codec_type") == "audio"
            for s in probe.get("streams", [])
        )

        # Step 2: build a list of (start, end, is_idle) intervals covering the full video
        intervals: list[tuple[float, float, bool]] = []
        cursor = 0.0
        for seg in segments:
            if seg["start"] > cursor:
                intervals.append((cursor, seg["start"], False))
            intervals.append((seg["start"], seg["end"], True))
            cursor = seg["end"]
        if cursor < total_duration:
            intervals.append((cursor, total_duration, False))

        # Step 3: split each interval into a temp file
        tmp_dir = tempfile.mkdtemp(prefix="idle_speed_")
        part_files: list[str] = []
        try:
            for idx, (start, end, is_idle) in enumerate(intervals):
                dur = end - start
                if dur <= 0:
                    continue
                part_path = str(Path(tmp_dir) / f"part_{idx:04d}.mp4")

                if is_idle and speed != 1.0:
                    # Extract + speed up in one pass (avoids keyframe cut issues)
                    setpts = f"setpts={1.0 / speed}*PTS"
                    speed_args = [
                        "-ss", str(start),
                        "-t", str(dur),
                        "-i", str(input_path),
                        "-vf", setpts,
                    ]

                    if has_audio:
                        # Build atempo chain for values outside [0.5, 2.0]
                        atempo_filters = []
                        remaining = speed
                        while remaining > 2.0:
                            atempo_filters.append("atempo=2.0")
                            remaining /= 2.0
                        while remaining < 0.5:
                            atempo_filters.append("atempo=0.5")
                            remaining /= 0.5
                        atempo_filters.append(f"atempo={remaining:.4f}")
                        speed_args.extend(["-af", ",".join(atempo_filters)])
                    else:
                        speed_args.extend(["-an"])

                    speed_args.extend([
                        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
                    ])
                    if has_audio:
                        speed_args.extend(["-c:a", "aac", "-b:a", "128k"])
                    speed_args.append(part_path)

                    r = self.ffmpeg.run(speed_args)
                    if r.returncode != 0:
                        return {"success": False, "error": f"Failed to speed up segment {idx}: {(r.stderr or '')[-300:]}"}
                else:
                    # Extract normal segment — re-encode for uniform concat
                    extract_args = [
                        "-ss", str(start),
                        "-t", str(dur),
                        "-i", str(input_path),
                        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
                    ]
                    if has_audio:
                        extract_args.extend(["-c:a", "aac", "-b:a", "128k"])
                    else:
                        extract_args.append("-an")
                    extract_args.append(part_path)

                    r = self.ffmpeg.run(extract_args)
                    if r.returncode != 0:
                        return {"success": False, "error": f"Failed to extract segment {idx}: {(r.stderr or '')[-300:]}"}

                part_files.append(part_path)

            # Step 4: concatenate all parts
            concat_list = Path(tmp_dir) / "concat.txt"
            with open(concat_list, "w", encoding="utf-8") as f:
                for pf in part_files:
                    # Use forward slashes for ffmpeg concat compatibility on Windows
                    f.write(f"file '{pf.replace(chr(92), '/')}'\n")

            concat_args = [
                "-f", "concat", "-safe", "0",
                "-i", str(concat_list),
                "-c", "copy",
                str(output_path),
            ]
            r = self.ffmpeg.run(concat_args)
            if r.returncode != 0:
                return {"success": False, "error": f"Failed to concatenate: {(r.stderr or '')[-300:]}"}

        finally:
            # Cleanup temp files
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

        idle_time_saved = sum(
            (s["duration"] - s["duration"] / speed) for s in segments
        )
        return {
            "success": True,
            "output": str(output_path),
            "original_duration": round(total_duration, 2),
            "estimated_new_duration": round(total_duration - idle_time_saved, 2),
            "time_saved": round(idle_time_saved, 2),
            "idle_segments_processed": len(segments),
            "speed_factor": speed,
        }