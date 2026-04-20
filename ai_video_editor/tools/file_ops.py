import logging
import os
import shutil
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

from send2trash import send2trash

from ..config.settings import Settings

logger = logging.getLogger(__name__)


class FileTool:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        logger.info("FileTool initialized")

    # ── helpers ──────────────────────────────────────────────────────────

    def _is_system_drive(self, path: Path) -> bool:
        """Check if *path* resides on the system drive (C:\\ on Windows)."""
        if platform.system() != "Windows":
            return False
        resolved = path.resolve()
        return resolved.drive.upper() == "C:"

    def _check_system_drive_access(self, path: Path, operation: str = "Operation") -> str | None:
        """Return an error string if *path* is on the system drive and access is blocked, else None."""
        resolved = path.resolve()
        if self._is_system_drive(resolved) and not self.settings.allow_system_drive_folder_access:
            return (
                f"{operation} blocked: {resolved} is on the system drive (C:\\). "
                "Set ALLOW_SYSTEM_DRIVE_FOLDER_ACCESS=true in .env or settings to override."
            )
        return None

    # ── read operations ─────────────────────────────────────────────────

    def read_file(
        self,
        file_path: str | Path,
        encoding: str = "utf-8",
        max_bytes: int | None = None,
    ) -> dict[str, Any]:
        """Read a file and return its content along with basic metadata."""
        p = Path(file_path).resolve()
        logger.debug("read_file: %s", p)
        err = self._check_system_drive_access(p, "Read")
        if err:
            return {"error": err}
        if not p.exists():
            return {"error": f"File not found: {p}"}
        if not p.is_file():
            return {"error": f"Not a file: {p}"}
        try:
            stat = p.stat()
            info: dict[str, Any] = {
                "path": str(p),
                "name": p.name,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "extension": p.suffix,
            }
            if max_bytes and stat.st_size > max_bytes:
                content = p.read_bytes()[:max_bytes].decode(encoding, errors="replace")
                info["truncated"] = True
            else:
                content = p.read_text(encoding=encoding, errors="replace")
                info["truncated"] = False
            info["content"] = content
            return info
        except Exception as e:
            return {"error": str(e)}

    def read_folder(
        self,
        folder_path: str | Path,
        recursive: bool = False,
        pattern: str | None = None,
    ) -> dict[str, Any]:
        """List contents of a directory with metadata for each entry."""
        p = Path(folder_path).resolve()
        err = self._check_system_drive_access(p, "Read folder")
        if err:
            return {"error": err}
        if not p.exists():
            return {"error": f"Folder not found: {p}"}
        if not p.is_dir():
            return {"error": f"Not a directory: {p}"}
        try:
            entries: list[dict[str, Any]] = []
            iterator = p.rglob(pattern or "*") if recursive else p.glob(pattern or "*")
            for item in sorted(iterator):
                stat = item.stat()
                entries.append({
                    "name": item.name,
                    "path": str(item),
                    "type": "directory" if item.is_dir() else "file",
                    "size": stat.st_size if item.is_file() else None,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })
            return {"path": str(p), "count": len(entries), "entries": entries}
        except Exception as e:
            return {"error": str(e)}

    def file_info(self, file_path: str | Path) -> dict[str, Any]:
        """Return metadata for a file or directory without reading content."""
        p = Path(file_path).resolve()
        err = self._check_system_drive_access(p, "File info")
        if err:
            return {"error": err}
        if not p.exists():
            return {"error": f"Path not found: {p}"}
        try:
            stat = p.stat()
            return {
                "path": str(p),
                "name": p.name,
                "type": "directory" if p.is_dir() else "file",
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "extension": p.suffix if p.is_file() else None,
                "exists": True,
            }
        except Exception as e:
            return {"error": str(e)}

    # ── delete operations ───────────────────────────────────────────────

    def delete_file(self, file_path: str | Path) -> dict[str, Any]:
        """Move a file to the recycle bin (recoverable).

        Blocks access on the system drive unless ``allow_system_drive_folder_access``
        is enabled in settings.
        """
        p = Path(file_path).resolve()
        logger.debug("delete_file: %s", p)
        err = self._check_system_drive_access(p, "Delete")
        if err:
            return {"error": err}
        if not p.exists():
            return {"error": f"Path does not exist: {p}"}
        if not p.is_file():
            return {"error": f"Not a file: {p}"}
        try:
            send2trash(str(p))
            return {"success": True, "deleted": str(p), "recoverable": True}
        except Exception as e:
            return {"error": str(e)}

    def delete_folder(self, folder_path: str | Path) -> dict[str, Any]:
        """Move a folder to the recycle bin (recoverable).

        Blocks access on the system drive unless ``allow_system_drive_folder_access``
        is enabled in settings.
        """
        p = Path(folder_path).resolve()
        logger.debug("delete_folder: %s", p)
        err = self._check_system_drive_access(p, "Delete")
        if err:
            return {"error": err}
        if not p.exists():
            return {"error": f"Path does not exist: {p}"}
        if not p.is_dir():
            return {"error": f"Not a directory: {p}"}
        try:
            send2trash(str(p))
            return {"success": True, "deleted": str(p), "recoverable": True}
        except Exception as e:
            return {"error": str(e)}

    # ── write / move operations ─────────────────────────────────────────

    def create_directory(
        self, folder_path: str | Path, exist_ok: bool = True
    ) -> dict[str, Any]:
        """Create a directory (and parents)."""
        p = Path(folder_path).resolve()
        err = self._check_system_drive_access(p, "Create directory")
        if err:
            return {"error": err}
        try:
            p.mkdir(parents=True, exist_ok=exist_ok)
            return {"success": True, "path": str(p)}
        except Exception as e:
            return {"error": str(e)}

    def move(
        self,
        source: str | Path,
        destination: str | Path,
    ) -> dict[str, Any]:
        """Move or rename a file / folder."""
        src = Path(source).resolve()
        dst = Path(destination).resolve()
        logger.debug("move: %s -> %s", src, dst)
        for label, p in [("Source", src), ("Destination", dst)]:
            err = self._check_system_drive_access(p, f"Move ({label.lower()})")
            if err:
                return {"error": err}
        if not src.exists():
            return {"error": f"Source not found: {src}"}
        try:
            shutil.move(str(src), str(dst))
            return {"success": True, "source": str(src), "destination": str(dst)}
        except Exception as e:
            return {"error": str(e)}

    def copy(
        self,
        source: str | Path,
        destination: str | Path,
    ) -> dict[str, Any]:
        """Copy a file or folder."""
        src = Path(source).resolve()
        dst = Path(destination).resolve()
        for label, p in [("Source", src), ("Destination", dst)]:
            err = self._check_system_drive_access(p, f"Copy ({label.lower()})")
            if err:
                return {"error": err}
        if not src.exists():
            return {"error": f"Source not found: {src}"}
        try:
            if src.is_dir():
                shutil.copytree(str(src), str(dst))
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(dst))
            return {"success": True, "source": str(src), "destination": str(dst)}
        except Exception as e:
            return {"error": str(e)}

    def exists(self, path: str | Path) -> dict[str, Any]:
        """Check whether a path exists and what type it is."""
        p = Path(path).resolve()
        err = self._check_system_drive_access(p, "Exists check")
        if err:
            return {"error": err}
        return {
            "path": str(p),
            "exists": p.exists(),
            "type": "directory" if p.is_dir() else ("file" if p.is_file() else None),
        }

    def rename(
        self,
        file_path: str | Path,
        new_name: str,
    ) -> dict[str, Any]:
        """Rename a file or folder in-place (keeps it in the same directory)."""
        src = Path(file_path).resolve()
        err = self._check_system_drive_access(src, "Rename")
        if err:
            return {"error": err}
        if not src.exists():
            return {"error": f"Path not found: {src}"}
        dst = src.parent / new_name
        if dst.exists():
            return {"error": f"Destination already exists: {dst}"}
        try:
            src.rename(dst)
            return {"success": True, "old_path": str(src), "new_path": str(dst)}
        except Exception as e:
            return {"error": str(e)}
