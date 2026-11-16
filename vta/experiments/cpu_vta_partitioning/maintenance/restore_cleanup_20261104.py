"""Restore individually verified files from the local 2026-11-04 cleanup.

Default is a dry run. No network or board access; existing changed files are never overwritten.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tarfile
import tempfile


REPO = Path(__file__).resolve().parents[4]
DEFAULT_MANIFEST = REPO / "vta/experiments/cpu_vta_partitioning/artifacts/cleanup_20261104/manifest.json"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inside(value):
    path = REPO / value
    if Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError("unsafe relative path: " + value)
    path.resolve().relative_to(REPO)
    return path


def install(stream, record):
    target = inside(record["path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as out:
        temporary = Path(out.name)
        try:
            shutil.copyfileobj(stream, out)
            out.flush()
            if temporary.stat().st_size != record["bytes"] or sha256(temporary) != record["sha256"]:
                raise ValueError("recovery content mismatch: " + record["path"])
            os.chmod(temporary, record.get("mode", stat.S_IRUSR | stat.S_IWUSR))
            if "mtime_ns" in record:
                os.utime(temporary, ns=(record["mtime_ns"], record["mtime_ns"]))
            # Atomic no-clobber install, including if a file appeared after preflight.
            os.link(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--category", action="append", help="repeat to select categories")
    parser.add_argument("--path-prefix", default="", help="restore only matching original paths")
    parser.add_argument("--restore", action="store_true", help="write files; default only lists them")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    records = [r for r in manifest["removed"]
               if (not args.category or r["category"] in args.category)
               and r["path"].startswith(args.path_prefix)]
    pending = []
    for record in records:
        target = inside(record["path"])
        if target.exists() or target.is_symlink():
            if target.is_symlink() or not target.is_file() or sha256(target) != record["sha256"]:
                raise ValueError("existing changed file: " + record["path"])
            continue
        pending.append(record)
    print(json.dumps({"restore": args.restore, "files": len(pending),
                      "bytes": sum(r["bytes"] for r in pending),
                      "categories": sorted({r["category"] for r in pending})}))
    if not args.restore:
        for r in pending:
            print(r["path"])
        return
    archives = defaultdict(list)
    for record in pending:
        if record["recovery"]["type"] == "retained_file":
            source = inside(record["recovery"]["path"])
            with source.open("rb") as stream:
                install(stream, record)
        else:
            archives[record["recovery"]["archive"]].append(record)
    for name, group in archives.items():
        archive = inside(name)
        expected = {r["recovery"]["archive_sha256"] for r in group}
        if expected != {sha256(archive)}:
            raise ValueError("recovery archive changed: " + name)
        members = defaultdict(list)
        for record in group:
            members[record["recovery"]["member"]].append(record)
        with tarfile.open(archive, "r|gz") as tar:
            for entry in tar:
                wanted = members.pop(entry.name, [])
                if not wanted:
                    continue
                if not entry.isfile():
                    raise ValueError("recovery member is not a regular file")
                # These cleanup archives have one destination per member.
                if len(wanted) != 1:
                    raise ValueError("duplicate recovery member")
                with tar.extractfile(entry) as stream:
                    install(stream, wanted[0])
        if members:
            raise ValueError("missing recovery members: " + str(list(members)[:3]))
    print("Restored and SHA256-verified", len(pending), "files.")


if __name__ == "__main__":
    main()
