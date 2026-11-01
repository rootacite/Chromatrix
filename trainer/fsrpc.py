"""Filesystem operations Chromatrix used to do itself. Torch-free.

Handlers here are called from `api.dispatch`. Paths are allowlisted against
`config.toml` (train-data folders, repo `config.toml` / `configs/`, `output_dir`).
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import time
import tomllib
import uuid
from pathlib import Path
from typing import Any, Optional, Sequence

from PIL import Image

try:
    from config import resolve_train_data_entries
except ImportError:
    from trainer.config import resolve_train_data_entries

try:
    from runs import validate_output_name
except ImportError:
    from trainer.runs import validate_output_name

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MASK_SIDECAR_SUFFIX = ".mask.png"
TRASH_DEFAULT = Path("/tmp/axlranko/trash")
TRASH_MANIFEST_NAME = "trash.json"
TRASH_MANIFEST_VERSION = 1
TRASH_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
PROFILE_DIR_NAME = "configs"
PROFILE_EXT = ".toml"
PROFILE_MAX_NAME = 64
PROFILE_ILLEGAL = set('/\\:*?"<>|')
PROMPT_MATRIX_NAME = "input_matrix.txt"
PROMPT_PROFILE_DIR_NAME = "prompt_profiles"
PROMPT_PROFILE_EXT = ".json"
SHUFFLE_TEMP_PREFIX = "axl-shuffle-"
SHUFFLE_STEM_PAD = 4
SHUFFLE_FIRST = "0001"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def config_path() -> Path:
    return _repo_root() / "config.toml"


def profile_dir() -> Path:
    return _repo_root() / PROFILE_DIR_NAME


def trash_dir() -> Path:
    raw = os.environ.get("AXL_TRASH_DIR")
    if raw:
        return Path(raw)
    return TRASH_DEFAULT


def is_mask_sidecar(name: str) -> bool:
    return name.lower().endswith(MASK_SIDECAR_SUFFIX)


def mask_path_for(image: Path) -> Path:
    return image.with_name(image.stem + MASK_SIDECAR_SUFFIX)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def train_data_roots(cfg: dict[str, Any]) -> list[Path]:
    entries = resolve_train_data_entries(cfg)
    return [Path(entry.path).expanduser().resolve() for entry in entries]


def output_root(cfg: dict[str, Any]) -> Path:
    return Path(str(cfg.get("output_dir") or "./output")).expanduser().resolve()


def require_dataset_dir(directory: Any, cfg: dict[str, Any]) -> Path:
    if not directory:
        raise ValueError("missing directory")
    path = Path(str(directory)).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"not a directory: {path}")
    roots = train_data_roots(cfg)
    if path not in roots:
        raise ValueError(f"not a configured train_data directory: {path}")
    return path


def require_stem(stem: Any) -> str:
    value = str(stem or "").strip()
    if not value:
        raise ValueError("missing stem")
    if "/" in value or "\\" in value or value in {".", ".."} or ".." in value:
        raise ValueError(f"invalid stem: {stem}")
    return value


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def blob_path_allowed(path: Path, cfg: dict[str, Any], extra_roots: Sequence[Path] = ()) -> bool:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return False
    if is_mask_sidecar(resolved.name):
        return False
    if resolved.suffix.lower() not in IMAGE_EXTENSIONS:
        return False
    if resolved.parent in train_data_roots(cfg):
        return True
    if _is_under(resolved, output_root(cfg)):
        return True
    # Review-mode Trash keeps the original image bytes for its "Trashed" previews.
    if _is_under(resolved, trash_dir()):
        return True
    # The Automation page's own output (settings.json's output_dir, plus the automation tree).
    return any(_is_under(resolved, root) for root in extra_roots)


def checkpoint_source_allowed(path: Path, cfg: dict[str, Any]) -> bool:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return False
    if resolved.suffix.lower() != ".safetensors":
        return False
    return _is_under(resolved, output_root(cfg))


def _image_files(directory: Path) -> list[Path]:
    files = [p for p in directory.iterdir() if p.is_file()]
    images = [
        p
        for p in files
        if p.suffix.lower() in IMAGE_EXTENSIONS and not is_mask_sidecar(p.name)
    ]
    return sorted(images, key=lambda p: p.name.lower())


def _caption_files(directory: Path) -> list[Path]:
    return sorted(
        [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".txt"],
        key=lambda p: p.name.lower(),
    )


def _probe_image(path: Path) -> tuple[int, int, bool]:
    with Image.open(path) as img:
        width, height = img.size
        has_alpha = img.mode in {"RGBA", "LA", "PA"} or (
            img.mode == "P" and "transparency" in img.info
        )
        return int(width), int(height), bool(has_alpha)


def _parse_tags(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def dataset_list(directory: Path) -> dict[str, Any]:
    images = _image_files(directory)
    captions = _caption_files(directory)
    image_stems = {p.stem for p in images}
    orphans = [p.name for p in captions if p.stem not in image_stems]
    items = [_active_item(directory, image) for image in images]
    return {"items": items, "orphans": orphans}


def _active_item(directory: Path, image: Path) -> dict[str, Any]:
    txt = directory / f"{image.stem}.txt"
    mask = mask_path_for(image)
    tags: list[str] = []
    if txt.is_file():
        tags = _parse_tags(txt.read_text(encoding="utf-8"))
    try:
        width, height, has_alpha = _probe_image(image)
    except OSError:
        width, height, has_alpha = 0, 0, False
    return {
        "stem": image.stem,
        "image": str(image),
        "txt": str(txt),
        "mask": str(mask),
        "width": width,
        "height": height,
        "tags": tags,
        "has_sidecar_mask": mask.is_file(),
        "has_alpha": has_alpha,
    }


def caption_write(directory: Path, stem: str, text: str) -> dict[str, Any]:
    path = directory / f"{stem}.txt"
    atomic_write_text(path, text)
    return {}


def dataset_drop(
    directory: Path,
    rate: float,
    seed: Optional[int] = None,
    stems: Optional[list[str]] = None,
) -> dict[str, Any]:
    if not (0.0 < rate <= 1.0):
        raise ValueError("rate must be in (0, 1]")
    rng = random.Random(seed)
    dest = trash_dir()
    dest.mkdir(parents=True, exist_ok=True)
    moved = 0
    listing = dataset_list(directory)
    if listing["orphans"]:
        raise ValueError(
            "Refusing to drop: isolated tag file(s) without a matching image: "
            + ", ".join(listing["orphans"])
        )
    wanted = set(stems) if stems is not None else None
    for item in listing["items"]:
        if wanted is not None and item["stem"] not in wanted:
            continue
        if not Path(item["txt"]).is_file():
            continue
        if rng.random() > rate:
            continue
        for key in ("image", "txt", "mask"):
            src = Path(item[key])
            if key == "mask" and not src.is_file():
                continue
            if not src.is_file():
                continue
            shutil.move(str(src), str(dest / src.name))
        moved += 1
    return {"moved": moved}


def require_trash_outside_train_data(cfg: dict[str, Any]) -> Path:
    """The review trash keeps files restorable beside the dataset, never inside it.

    The trainer walks a train-data folder recursively, so a trash root nested under one would put
    trashed images back into the next run's image list even though they are "removed" from the
    folder the user sees.
    """
    dest = trash_dir().expanduser().resolve()
    for root in train_data_roots(cfg):
        if _is_under(dest, root):
            raise ValueError(
                f"trash directory {dest} is inside train data {root}; "
                "set AXL_TRASH_DIR outside every dataset folder"
            )
    return dest


def _safe_record_file(record: Path, name: Any) -> Path:
    value = str(name or "")
    if (
        not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
    ):
        raise ValueError(f"invalid trash file name: {name!r}")
    path = record / value
    if path.parent.resolve() != record.resolve():
        raise ValueError(f"trash file escapes its record: {name!r}")
    return path


def _trash_record_dir(trash_id: Any) -> Path:
    value = str(trash_id or "").strip()
    if not TRASH_ID_PATTERN.fullmatch(value):
        raise ValueError(f"invalid trash id: {trash_id!r}")
    record = trash_dir() / value
    if not record.is_dir():
        raise ValueError(f"trash item not found: {value}")
    return record


def _write_trash_manifest(record: Path, manifest: dict[str, Any]) -> None:
    atomic_write_text(
        record / TRASH_MANIFEST_NAME,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _read_trash_manifest(record: Path) -> dict[str, Any]:
    path = record / TRASH_MANIFEST_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"unreadable trash record {record.name}: {exc}") from exc
    if not isinstance(data, dict) or data.get("version") != TRASH_MANIFEST_VERSION:
        raise ValueError(f"unsupported trash record {record.name}")
    source = str(data.get("source_directory") or "")
    if not source:
        raise ValueError(f"trash record {record.name} has no source directory")
    stem = require_stem(data.get("stem"))
    files = data.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"trash record {record.name} has no files")
    clean_files: list[dict[str, str]] = []
    image_count = 0
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError(f"trash record {record.name} has an invalid file entry")
        name = str(entry.get("name") or "")
        _safe_record_file(record, name)
        kind = str(entry.get("kind") or "file")
        if kind == "image":
            image_count += 1
        clean_files.append({"name": name, "kind": kind})
    if image_count == 0:
        raise ValueError(f"trash record {record.name} has no image")
    return {
        "version": TRASH_MANIFEST_VERSION,
        "id": str(data.get("id") or record.name),
        "source_directory": source,
        "stem": stem,
        "trashed_at": int(data.get("trashed_at") or 0),
        "files": clean_files,
    }


def _same_directory(left: Any, right: Path) -> bool:
    try:
        return Path(str(left)).expanduser().resolve() == right.expanduser().resolve()
    except OSError:
        return False


def _trash_item_record(record: Path, directory: Path) -> Optional[dict[str, Any]]:
    manifest = _read_trash_manifest(record)
    if not _same_directory(manifest["source_directory"], directory):
        return None
    files = [
        _safe_record_file(record, entry["name"])
        for entry in manifest["files"]
        if entry["kind"] == "image"
    ]
    images = sorted([path for path in files if path.is_file()], key=lambda path: path.name.lower())
    if not images:
        return None
    image = images[0]
    stem = manifest["stem"]
    txt = record / f"{stem}.txt"
    mask = record / f"{stem}{MASK_SIDECAR_SUFFIX}"
    tags: list[str] = []
    if txt.is_file():
        tags = _parse_tags(txt.read_text(encoding="utf-8"))
    try:
        width, height, has_alpha = _probe_image(image)
    except OSError:
        width, height, has_alpha = 0, 0, False
    return {
        "trash_id": record.name,
        "stem": stem,
        "image": str(image),
        "txt": str(txt),
        "mask": str(mask),
        "width": width,
        "height": height,
        "tags": tags,
        "has_sidecar_mask": mask.is_file(),
        "has_alpha": has_alpha,
        "trashed_at": manifest["trashed_at"],
        "file_count": len(manifest["files"]),
        "source_directory": str(directory),
    }


def dataset_trash_list(directory: Path) -> dict[str, Any]:
    """Review-mode items moved to Trash from [directory] and not restored yet."""
    root = trash_dir().expanduser().resolve()
    items: list[dict[str, Any]] = []
    if not root.is_dir():
        return {"items": []}
    for record in sorted(root.iterdir(), key=lambda path: path.name):
        if not record.is_dir() or not (record / TRASH_MANIFEST_NAME).is_file():
            continue
        try:
            item = _trash_item_record(record, directory)
        except (OSError, ValueError):
            continue
        if item is not None:
            items.append(item)
    items.sort(key=lambda item: (item["trashed_at"], item["stem"].lower(), item["trash_id"]))
    return {"items": items}


def _dataset_group(directory: Path, stem: str) -> tuple[list[Path], Optional[Path], Optional[Path]]:
    images = sorted(
        [path for path in _image_files(directory) if path.stem == stem],
        key=lambda path: path.name.lower(),
    )
    if not images:
        raise ValueError(f"no image with stem {stem!r} in {directory}")
    txt = directory / f"{stem}.txt"
    mask = directory / f"{stem}{MASK_SIDECAR_SUFFIX}"
    return (
        images,
        txt if txt.is_file() else None,
        mask if mask.is_file() else None,
    )


def dataset_trash(directory: Path, stem: str) -> dict[str, Any]:
    """Move one image group (image(s) + caption + mask) into a recoverable trash record."""
    stem = require_stem(stem)
    images, txt, mask = _dataset_group(directory, stem)
    record = trash_dir() / uuid.uuid4().hex
    record.mkdir(parents=True, exist_ok=False)

    planned: list[tuple[Path, str]] = []
    for path, kind in ((txt, "caption"), (mask, "mask")):
        if path is not None:
            planned.append((path, kind))
    planned.extend((path, "image") for path in images)
    manifest = {
        "version": TRASH_MANIFEST_VERSION,
        "id": record.name,
        "source_directory": str(directory),
        "stem": stem,
        "trashed_at": int(time.time() * 1000),
        "files": [{"name": path.name, "kind": kind} for path, kind in planned],
    }
    _write_trash_manifest(record, manifest)

    moved: list[tuple[Path, Path]] = []
    try:
        for source, _kind in planned:
            destination = record / source.name
            shutil.move(str(source), str(destination))
            moved.append((source, destination))
    except Exception as exc:
        rollback_ok = True
        for source, destination in reversed(moved):
            try:
                if destination.is_file() and not source.exists():
                    shutil.move(str(destination), str(source))
            except OSError:
                rollback_ok = False
        if rollback_ok:
            shutil.rmtree(record, ignore_errors=True)
            raise OSError(f"could not move {stem!r} to trash: {exc}") from exc
        raise OSError(
            f"could not move {stem!r} to trash ({exc}); the recoverable record is {record}"
        ) from exc

    item = _trash_item_record(record, directory)
    if item is None:
        raise OSError(f"trash record {record} was written but has no readable image")
    return {"stem": stem, "item": item}


def dataset_restore(directory: Path, trash_id: str) -> dict[str, Any]:
    """Move a review trash record back to the folder it came from, refusing overwrites."""
    record = _trash_record_dir(trash_id)
    manifest = _read_trash_manifest(record)
    if not _same_directory(manifest["source_directory"], directory):
        raise ValueError(
            f"trash item {trash_id} belongs to {manifest['source_directory']}, not {directory}"
        )
    stem = manifest["stem"]
    plan = [
        (_safe_record_file(record, entry["name"]), entry["kind"])
        for entry in manifest["files"]
    ]
    present = [(source, kind) for source, kind in plan if source.is_file()]
    if not any(kind == "image" for _source, kind in present):
        raise ValueError(f"trash item {trash_id} has no image to restore")
    for source, _kind in present:
        destination = directory / source.name
        if destination.exists():
            raise ValueError(f"cannot restore {source.name}: {destination} already exists")

    # Sidecars first, image last: an interrupted restore never leaves an orphan caption behind.
    order = sorted(present, key=lambda pair: 0 if pair[1] != "image" else 1)
    restored: list[tuple[Path, Path]] = []
    try:
        for source, _kind in order:
            destination = directory / source.name
            shutil.move(str(source), str(destination))
            restored.append((source, destination))
    except Exception as exc:
        for source, destination in reversed(restored):
            try:
                if destination.is_file() and not source.exists():
                    shutil.move(str(destination), str(source))
            except OSError:
                pass
        raise OSError(f"could not restore {stem!r}: {exc}") from exc

    try:
        (record / TRASH_MANIFEST_NAME).unlink()
        record.rmdir()
    except OSError:
        pass

    images = sorted(
        [path for path in _image_files(directory) if path.stem == stem],
        key=lambda path: path.name.lower(),
    )
    if not images:
        raise OSError(f"restored {stem!r} but no image is visible in {directory}")
    return {"stem": stem, "item": _active_item(directory, images[0])}


def _plan_groups(directory: Path) -> list[tuple[str, list[Path]]]:
    files = [p for p in directory.iterdir() if p.is_file()]
    images = [
        p
        for p in files
        if p.suffix.lower() in IMAGE_EXTENSIONS and not is_mask_sidecar(p.name)
    ]
    images = sorted(images, key=lambda p: p.name)
    captions = sorted(
        [p for p in files if p.suffix.lower() == ".txt"],
        key=lambda p: p.name,
    )
    masks = sorted(
        [p for p in files if is_mask_sidecar(p.name)],
        key=lambda p: p.name,
    )
    stems = sorted({p.stem for p in images})
    caption_stems = {p.stem for p in captions}
    orphans = sorted(caption_stems - set(stems))
    if orphans:
        names = ", ".join(f"{stem}.txt" for stem in orphans)
        raise ValueError(
            "Refusing to shuffle: isolated tag file(s) without a matching image: " + names
        )
    groups: list[tuple[str, list[Path]]] = []
    for stem in stems:
        members = [p for p in images if p.stem == stem]
        members += [p for p in captions if p.stem == stem]
        members += [p for p in masks if p.name[: -len(MASK_SIDECAR_SUFFIX)] == stem]
        groups.append((stem, members))
    return groups


def _suffix(stem: str, path: Path) -> str:
    return path.name[len(stem) :]


def dataset_shuffle(directory: Path, seed: Optional[int] = None) -> dict[str, Any]:
    groups = _plan_groups(directory)
    if not groups:
        return {"groups": 0, "renamed_files": 0, "first_stem": "", "last_stem": ""}
    rng = random.Random(seed)
    order = list(groups)
    rng.shuffle(order)
    applied: list[tuple[Path, Path]] = []

    def move(src: Path, dest: Path) -> None:
        try:
            os.rename(src, dest)
        except OSError as exc:
            raise OSError(f"could not rename {src.name} to {dest.name}: {exc}") from exc
        applied.append((src, dest))

    try:
        staged: list[tuple[str, tuple[str, list[Path]]]] = []
        for stem, members in order:
            temp = SHUFFLE_TEMP_PREFIX + uuid.uuid4().hex[:12]
            for member in members:
                move(member, directory / f"{temp}{_suffix(stem, member)}")
            staged.append((temp, (stem, members)))
        for index, (temp, (stem, members)) in enumerate(staged):
            base = str(index + 1).zfill(SHUFFLE_STEM_PAD)
            for member in members:
                suffix = _suffix(stem, member)
                move(directory / f"{temp}{suffix}", directory / f"{base}{suffix}")
    except Exception:
        for src, dest in reversed(applied):
            try:
                os.rename(dest, src)
            except OSError:
                pass
        raise

    renamed = sum(len(members) for _, members in order)
    last = str(len(order)).zfill(SHUFFLE_STEM_PAD)
    return {
        "groups": len(order),
        "renamed_files": renamed,
        "first_stem": SHUFFLE_FIRST,
        "last_stem": last,
    }


def mask_file(directory: Path, stem: str) -> Path:
    return directory / f"{stem}{MASK_SIDECAR_SUFFIX}"


def mask_get(directory: Path, stem: str) -> dict[str, Any]:
    import base64

    path = mask_file(directory, stem)
    if not path.is_file():
        return {"png_base64": ""}
    return {"png_base64": base64.b64encode(path.read_bytes()).decode("ascii")}


def mask_write(directory: Path, stem: str, png_base64: str) -> dict[str, Any]:
    import base64

    if not png_base64:
        raise ValueError("missing png_base64")
    try:
        data = base64.b64decode(png_base64, validate=False)
    except Exception as exc:
        raise ValueError("png_base64 is not valid base64") from exc
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("mask must be a PNG")
    atomic_write_bytes(mask_file(directory, stem), data)
    return {}


def mask_delete(directory: Path, stem: str) -> dict[str, Any]:
    path = mask_file(directory, stem)
    if path.is_file():
        path.unlink()
    return {}


def config_get() -> dict[str, Any]:
    path = config_path()
    if not path.is_file():
        raise ValueError(f"Could not locate config.toml at {path}")
    return {"path": str(path), "text": path.read_text(encoding="utf-8")}


def config_save(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("config text is empty")
    try:
        parsed = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"config.toml would not parse: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("config.toml must be a table")
    # The trainer flattens every table into one key space, so the name is looked up there.
    flat: dict[str, Any] = {}
    for section in parsed.values():
        if isinstance(section, dict):
            flat.update(section)
    if flat.get("output_name") is not None:
        # The trainer would refuse this config at startup; refuse to write it in the first place.
        name_error = validate_output_name(str(flat["output_name"]))
        if name_error:
            raise ValueError(name_error)
    path = config_path()
    if not path.is_file():
        raise ValueError(f"Config file does not exist: {path}")
    atomic_write_text(path, text)
    return {"path": str(path)}


def validate_profile_name(name: str) -> Optional[str]:
    trimmed = name.strip()
    if not trimmed:
        return "Enter a profile name"
    if len(trimmed) > PROFILE_MAX_NAME:
        return f"Use at most {PROFILE_MAX_NAME} characters"
    if trimmed.startswith("."):
        return "A profile name cannot start with a dot"
    if any(ch in PROFILE_ILLEGAL or ord(ch) < 32 for ch in trimmed):
        return 'A profile name cannot contain / \\ : * ? " < > |'
    return None


def _profile_path(name: str) -> Path:
    err = validate_profile_name(name)
    if err:
        raise ValueError(err)
    return profile_dir() / f"{name.strip()}{PROFILE_EXT}"


def profile_list() -> dict[str, Any]:
    directory = profile_dir()
    if not directory.is_dir():
        return {"profiles": []}
    profiles = []
    for path in sorted(directory.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file() or path.suffix.lower() != PROFILE_EXT:
            continue
        stat = path.stat()
        profiles.append(
            {
                "name": path.stem,
                "modified": int(stat.st_mtime * 1000),
                "size": int(stat.st_size),
            }
        )
    return {"profiles": profiles}


def profile_get(name: str) -> dict[str, Any]:
    path = _profile_path(name)
    if not path.is_file():
        raise ValueError(f"No profile named {name.strip()}")
    return {"name": path.stem, "text": path.read_text(encoding="utf-8")}


def profile_save(name: str, text: str, overwrite: bool) -> dict[str, Any]:
    err = validate_profile_name(name)
    if err:
        raise ValueError(err)
    trimmed = name.strip()
    directory = profile_dir()
    directory.mkdir(parents=True, exist_ok=True)
    existing = None
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() == PROFILE_EXT and path.stem.lower() == trimmed.lower():
            existing = path
            break
    if existing is not None and not overwrite:
        raise ValueError(f'A profile named "{trimmed}" already exists')
    target = existing if existing is not None else directory / f"{trimmed}{PROFILE_EXT}"
    atomic_write_text(target, text)
    return {"name": target.stem}


def profile_delete(name: str) -> dict[str, Any]:
    path = _profile_path(name)
    if not path.is_file():
        raise ValueError(f"The profile is already gone: {name}")
    if path.resolve().parent != profile_dir().resolve():
        raise ValueError(f"Not a profile file: {path}")
    path.unlink()
    return {}


def prompt_matrix_path() -> Path:
    return _repo_root() / PROMPT_MATRIX_NAME


def prompt_profile_dir() -> Path:
    return _repo_root() / PROMPT_PROFILE_DIR_NAME


def prompt_matrix() -> dict[str, Any]:
    """The prompt wizard's input matrix, so Chromatrix and tools/gen_prompts.py read one file."""
    path = prompt_matrix_path()
    if not path.is_file():
        raise ValueError(f"Could not locate {PROMPT_MATRIX_NAME} at {path}")
    return {"path": str(path), "text": path.read_text(encoding="utf-8")}


def _prompt_profile_path(name: str) -> Path:
    err = validate_profile_name(name)
    if err:
        raise ValueError(err)
    return prompt_profile_dir() / f"{name.strip()}{PROMPT_PROFILE_EXT}"


def _prompt_profile_version(payload: dict[str, Any]) -> Optional[int]:
    raw = payload.get("version")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return int(raw)


def prompt_profile_list() -> dict[str, Any]:
    directory = prompt_profile_dir()
    if not directory.is_dir():
        return {"profiles": []}
    profiles = []
    for path in sorted(directory.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file() or path.suffix.lower() != PROMPT_PROFILE_EXT:
            continue
        stat = path.stat()
        version: Optional[int] = None
        error: Optional[str] = None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                error = "profile must be a JSON object"
            else:
                version = _prompt_profile_version(payload)
        except OSError as exc:
            error = f"cannot read: {exc}"
        except json.JSONDecodeError as exc:
            error = f"invalid JSON ({exc})"
        profiles.append(
            {
                "name": path.stem,
                "version": version,
                "modified": int(stat.st_mtime * 1000),
                "size": int(stat.st_size),
                "error": error,
            }
        )
    return {"profiles": profiles}


def prompt_profile_get(name: str) -> dict[str, Any]:
    path = _prompt_profile_path(name)
    if not path.is_file():
        raise ValueError(f"No prompt profile named {name.strip()}")
    return {"name": path.stem, "text": path.read_text(encoding="utf-8")}


def prompt_profile_save(name: str, text: str, overwrite: bool) -> dict[str, Any]:
    err = validate_profile_name(name)
    if err:
        raise ValueError(err)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("prompt profile text is empty")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"prompt profile would not parse: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("spec"), dict):
        raise ValueError("a prompt profile needs a spec object")
    trimmed = name.strip()
    directory = prompt_profile_dir()
    directory.mkdir(parents=True, exist_ok=True)
    existing = None
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() == PROMPT_PROFILE_EXT and path.stem.lower() == trimmed.lower():
            existing = path
            break
    if existing is not None and not overwrite:
        raise ValueError(f'A prompt profile named "{trimmed}" already exists')
    target = existing if existing is not None else directory / f"{trimmed}{PROMPT_PROFILE_EXT}"
    atomic_write_text(target, text)
    return {"name": target.stem, "path": str(target)}


def prompt_profile_delete(name: str) -> dict[str, Any]:
    path = _prompt_profile_path(name)
    if not path.is_file():
        raise ValueError(f"The prompt profile is already gone: {name}")
    if path.resolve().parent != prompt_profile_dir().resolve():
        raise ValueError(f"Not a prompt profile file: {path}")
    path.unlink()
    return {}


def tag_lexicon() -> dict[str, Any]:
    path = _repo_root() / "tagger" / "selected_tags.csv"
    if not path.is_file():
        return {"text": ""}
    return {"text": path.read_text(encoding="utf-8")}


def checkpoint_export(source: Path, dest: Path) -> dict[str, Any]:
    src = source.expanduser().resolve()
    dst = dest.expanduser().resolve()
    if not src.is_file():
        raise ValueError(f"not a file: {src}")
    if dst == src:
        raise ValueError(f"destination is the same file as the source: {dst}")
    if dst.suffix.lower() != ".safetensors":
        raise ValueError("destination must end in .safetensors")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return {"bytes": dst.stat().st_size}


def logging_root(cfg: dict[str, Any]) -> Path:
    return Path(str(cfg.get("logging_dir") or "./logs")).expanduser().resolve()


def fs_listdir(path: Any) -> dict[str, Any]:
    """List one directory. No file bytes. `..` is canonicalized via resolve()."""
    if not path:
        raise ValueError("missing path")
    resolved = Path(str(path)).expanduser().resolve()
    if resolved.is_file():
        raise ValueError(f"not a directory: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"not a directory: {resolved}")
    parent = resolved.parent
    parent_s = str(parent) if parent != resolved else None
    entries: list[dict[str, Any]] = []
    try:
        children = list(resolved.iterdir())
    except OSError as exc:
        raise ValueError(f"cannot list {resolved}: {exc}") from exc
    children.sort(key=lambda p: (not p.is_dir(), p.name.lower()))
    for child in children:
        try:
            st = child.stat()
            is_dir = child.is_dir()
        except OSError:
            continue
        try:
            child_path = str(child.resolve()) if is_dir else str(child)
        except OSError:
            child_path = str(child)
        entries.append(
            {
                "name": child.name,
                "path": child_path,
                "is_dir": is_dir,
                "size": 0 if is_dir else int(st.st_size),
                "mtime_ms": int(st.st_mtime * 1000),
            }
        )
    return {"path": str(resolved), "parent": parent_s, "entries": entries}


def fs_roots(cfg: dict[str, Any]) -> dict[str, Any]:
    roots: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(name: str, path: Path) -> None:
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            return
        if not resolved.is_dir():
            return
        key = str(resolved)
        if key in seen:
            return
        seen.add(key)
        roots.append({"name": name, "path": key})

    add("Home", Path.home())
    add("Repo", _repo_root())
    for i, folder in enumerate(train_data_roots(cfg)):
        add("Train data" if i == 0 else f"Train data {i + 1}", folder)
    add("Output", output_root(cfg))
    add("Logs", logging_root(cfg))
    return {"roots": roots}
