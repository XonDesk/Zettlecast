"""
Files that live on the worker machine, as the primary refers to them.

With two machines the primary keeps the notes and the worker keeps the media (podcast audio,
screenshots). A file on the worker is "desktop:<path>": relative to the worker's storage folder
when it is inside it ("desktop:podcasts/Show/episode.mp3"), else absolute
("desktop:C:/Users/me/OneDrive/Screenshots/shot.png"). The primary fetches such files from the
worker's file server when it is on.
"""

from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Optional

from .config import settings

PREFIX = "desktop:"


def is_worker_ref(value) -> bool:
    return isinstance(value, str) and value.startswith(PREFIX)


def strip(value: str) -> str:
    """The path part of a worker reference (anything else unchanged)."""
    return value[len(PREFIX):] if is_worker_ref(value) else value


def relative_to(path: Path, root: Path) -> Optional[str]:
    """path relative to root, with forward slashes (None if it is outside root)."""
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return None


def to_worker_ref(path: Path, root: Optional[Path] = None) -> str:
    """How the primary should refer to a file on this (worker) machine."""
    root = root or settings.storage_path
    rel = relative_to(path, root)
    return PREFIX + (rel if rel is not None else Path(path).resolve().as_posix())


def from_worker_ref(ref: str, root: Optional[Path] = None) -> Path:
    """The local file a worker reference names (on the worker)."""
    root = root or settings.storage_path
    value = strip(ref)
    if PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute():
        return Path(value)
    if ".." in PurePosixPath(value).parts:
        raise ValueError(f"Not a path inside storage: {ref}")
    return Path(root) / value


def name(ref: str) -> str:
    """File name of a path or worker reference (either slash)."""
    return PureWindowsPath(strip(ref)).name
