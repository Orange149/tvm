"""Small, hardware-independent artifact primitives."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

PROJECT = Path(__file__).resolve().parents[2]
REPO = PROJECT.parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    data = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
        for directory in (path.parent, *path.parent.parents):
            sync_directory(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sync_tree(path):
    """Make returned evidence durable before acknowledging remote deletion."""
    path = Path(path)
    for item in path.rglob("*"):
        if item.is_file():
            with item.open("rb") as stream:
                os.fsync(stream.fileno())
    for directory in sorted((p for p in path.rglob("*") if p.is_dir()),
                            key=lambda p: len(p.parts), reverse=True):
        sync_directory(directory)
    for directory in (path, *path.parents):
        sync_directory(directory)


def relative_file(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError("path escapes artifact root: " + str(relative))
    return path


def inventory(root, exclude=()):
    return {str(p.relative_to(root)): {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(Path(root).rglob("*")) if p.is_file()
            and str(p.relative_to(root)) not in exclude}


def verify_inventory(root, files):
    for name, expected in files.items():
        p = relative_file(root, name)
        if not p.is_file() or p.stat().st_size != expected["bytes"] or sha256(p) != expected["sha256"]:
            raise ValueError("missing or changed artifact: " + name)


@contextmanager
def locked(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another controller owns " + str(directory)) from error
        yield


def event(directory, kind, **fields):
    with (Path(directory) / "events.jsonl").open("a") as stream:
        stream.write(canonical({"utc": now(), "event": kind, **fields}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
