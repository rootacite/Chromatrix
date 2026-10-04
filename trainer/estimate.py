"""How many steps a run will take: the figure the Utils → Training section shows while you edit.

Torch-free on purpose (like `runs.py`, `genjob.py` and `evaluation.py`), because api.py serves it
and the helper process must not pay for a torch import to answer a form question.

The count is the number of images the trainer's dataset would draw in one epoch, and it mirrors
`trainer/utils.list_images` for that: the same extensions, the same `*.mask.png` exclusion and the
same recursive walk over each `[[environment.train_data]]` folder — a hand-written mirror, not an
import, because `utils.py` pulls torch in. The step arithmetic lives on the Chromatrix side
(`model/StepEstimate.kt`), which is what makes epoch / batch / GA edits cost no IPC at all.

`count_train_images` also applies the validation split (`validation_split.select_validation`), the
same function `LoraImageDataset` uses for the real run, and reports the held-out images and draws as
`val_images` / `val_samples` so the estimate can subtract them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

try:
    from validation_split import select_validation
except ImportError:
    from trainer.validation_split import select_validation

# Same set `trainer/utils.list_images` walks, and the same mask exclusion.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MASK_SUFFIX = ".mask.png"


def is_mask_sidecar(path: Path) -> bool:
    return path.name.lower().endswith(MASK_SUFFIX)


def image_paths(root: Path) -> list[Path]:
    """Images under `root`, recursively, in `trainer/utils.list_images` order.

    The order matters: the validation split draws positions from this list, and the dataset's own
    copy of it has to come out in the same order for the two to hold out the same images.
    """
    found = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not is_mask_sidecar(path)
    ]
    return sorted(found)


def count_images(root: Path) -> int:
    """Images under `root`, recursively: what `LoraImageDataset` would take from that folder."""
    return len(image_paths(root))


def _entry(entry: Any) -> tuple[str, int]:
    """One requested folder as `(path, repeat)`; a `TrainDataEntry` or a plain mapping."""
    if isinstance(entry, Mapping):
        path = str(entry.get("path") or "").strip()
        raw_repeat = entry.get("repeat", 1)
    else:
        path = str(getattr(entry, "path", "") or "").strip()
        raw_repeat = getattr(entry, "repeat", 1)
    try:
        repeat = int(raw_repeat)
    except (TypeError, ValueError):
        repeat = 1
    return path, max(1, repeat)


def count_train_images(
    entries: Iterable[Any],
    val_split_percent: float = 0.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Image counts per training folder, plus the per-epoch totals a step estimate is built from.

    Each entry answers `{path, repeat, images, error}`; a folder that is missing or unreadable
    carries its reason and counts as zero, so one bad path cannot take the whole estimate down.
    `samples` is the per-epoch figure with repeats applied — what `LoraImageDataset.total_samples`
    holds at run time — and `val_images` / `val_samples` are the part the validation split holds
    out (unique images, and the draws they would have contributed), which the estimate subtracts.
    """
    rows: list[dict[str, Any]] = []
    folder_paths: list[list[Path]] = []
    folder_repeats: list[int] = []
    images_total = 0
    samples_total = 0
    for raw in entries:
        path, repeat = _entry(raw)
        images = 0
        paths: list[Path] = []
        error: Optional[str] = None
        if not path:
            error = "no path"
        else:
            root = Path(path).expanduser()
            if not root.is_dir():
                error = "not a directory"
            else:
                try:
                    paths = image_paths(root)
                    images = len(paths)
                except OSError as exc:  # an unreadable tree is this folder's answer, not the form's
                    error = str(exc)
        folder_paths.append(paths)
        folder_repeats.append(int(repeat))
        rows.append(
            {
                "path": path,
                "repeat": int(repeat),
                "images": int(images),
                "error": error,
            }
        )
        images_total += images
        samples_total += images * repeat

    held_out = select_validation(folder_paths, val_split_percent, seed)
    return {
        "entries": rows,
        "images": int(images_total),
        "samples": int(samples_total),
        "val_images": len(held_out),
        "val_samples": int(sum(folder_repeats[folder] for folder, _ in held_out)),
    }
