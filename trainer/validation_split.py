"""Which dataset images one run holds out, and which of them the fixed curve scores.

Torch-free on purpose, like `estimate.py`: `api.py` imports it in the helper process to answer the
Utils step estimate, and `trainer/dataset.py` calls the same functions for the real split, so the
estimate and the run cannot disagree about how many images leave training.

`select_diverse_subset` is the one piece that needs pixels: it picks the fixed validation sample
(`Val/Fixed_Loss`) so the scored images are as unlike each other as the held-out set allows. PIL and
numpy are already in the helper's import graph (`trainer/blobcodec.py`), so this costs the API
process nothing; only the trainer calls it.
"""

from __future__ import annotations

import math
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from PIL import Image

# Percent of the dataset's unique images held out. 0 disables the feature; the upper bound still
# leaves a training set on a small folder (`held_out_count` never takes the last image).
VAL_SPLIT_PERCENT_RANGE = (0.0, 90.0)

# One signature per image: a 32x32 RGB thumbnail, aspect-squashed so every signature has the same
# length (the buckets, not this picker, are what keep aspect ratios apart in training).
THUMBNAIL_SIZE = (32, 32)


def held_out_count(total_images: int, percent: float) -> int:
    """Whole images to hold out: `ceil` of the share, never all of them.

    The ratio counts unique images, not the per-epoch draws: an image that leaves takes its
    folder's `repeat` draws with it, so the share of drawn samples is that much larger.
    """
    total = int(total_images)
    if total <= 1:
        return 0
    share = float(percent)
    if not share > 0:
        return 0
    wanted = int(math.ceil(share / 100.0 * total))
    return max(0, min(wanted, total - 1))


def select_validation(
    folders: Sequence[Sequence[Path]],
    percent: float,
    seed: int,
) -> set[tuple[int, int]]:
    """Held-out `(folder_index, position)` pairs, drawn once from a seeded generator.

    The flat list walks the folders in entry order and each folder in its own `list_images` order,
    so the same dataset content, seed and percent always name the same images. `percent` reaches
    the seed in `%g` form, so a TOML integer (`10`) and the helper's parsed float (`10.0`) pick the
    same images.
    """
    pairs = [
        (folder, position)
        for folder, images in enumerate(folders)
        for position in range(len(images))
    ]
    count = held_out_count(len(pairs), percent)
    if count <= 0:
        return set()
    rng = random.Random(f"axl-val-split:{int(seed)}:{float(percent):g}")
    return set(rng.sample(pairs, count))


def signature_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """`tools/cmp_img.py:similarity`, on two signatures: 100 = identical, 0 = 255 apart.

    That script's definition exactly (mean absolute pixel difference over RGB), the difference being
    that the arrays here are the 32x32 thumbnails rather than full-resolution images with equal
    shapes — `cmp_img.copy_one` refuses a pair whose shapes differ, and a dataset folder is full of
    those.
    """
    diff = np.abs(left.astype(np.int16) - right.astype(np.int16))
    return float((1.0 - diff.mean() / 255.0) * 100.0)


def _signature(path: Path) -> Optional[np.ndarray]:
    """One image's thumbnail signature, or `None` when it cannot be read."""
    try:
        with Image.open(path) as img:
            # JPEG draft mode decodes straight to a smaller size; harmless for other formats.
            img.draft("RGB", (THUMBNAIL_SIZE[0] * 4, THUMBNAIL_SIZE[1] * 4))
            small = img.convert("RGB").resize(THUMBNAIL_SIZE, Image.BILINEAR)
            return np.asarray(small, dtype=np.float32).reshape(-1)
    except Exception:  # noqa: BLE001 - an unreadable image cannot be picked, and must not stop the run
        return None


def select_diverse_subset(
    paths: Sequence[Path],
    count: int,
    *,
    workers: Optional[int] = None,
) -> tuple[list[int], dict[str, Any]]:
    """`count` positions among `paths`, chosen so the images are as unlike each other as possible.

    Farthest-point traversal seeded at the most *typical* image (the one closest to the signature
    mean), then repeatedly adding the candidate whose worst-case similarity to what is already
    chosen is lowest. Every step is a strict `>` scan with index order as the tie-break, so the same
    folder contents always yield the same subset — no seed needed, and the fixed curve scores the
    same images on every pass, at every epoch, in every resume.

    Returns the sorted positions and a stats dict (`max_similarity`, `median_similarity` over the
    chosen pairs, `unreadable`) for the run's log line. `count >= len(paths)` degenerates to all of
    them; an empty or fully unreadable set returns an empty list.
    """
    total = len(paths)
    take = max(0, min(int(count), total))
    if take <= 0:
        return [], {"count": 0, "pool": total, "max_similarity": None, "median_similarity": None,
                    "unreadable": 0}

    pool = max(1, min(int(workers) if workers else 8, total))
    if pool > 1:
        with ThreadPoolExecutor(max_workers=pool) as executor:
            signatures = list(executor.map(_signature, paths))
    else:
        signatures = [_signature(path) for path in paths]

    unreadable = sum(1 for signature in signatures if signature is None)
    blank = np.zeros(THUMBNAIL_SIZE[0] * THUMBNAIL_SIZE[1] * 3, dtype=np.float32)
    vectors = [blank if signature is None else signature for signature in signatures]

    if take >= total:
        max_similarity, median_similarity = _pairwise_stats(range(total), vectors)
        return list(range(total)), {
            "count": total,
            "pool": total,
            "max_similarity": max_similarity,
            "median_similarity": median_similarity,
            "unreadable": unreadable,
        }

    # An unreadable image gets a blank signature, which no real image resembles — so it would win
    # every "least similar" comparison. Keep it out of the candidates unless the readable ones
    # cannot fill the subset themselves.
    candidates = [index for index, signature in enumerate(signatures) if signature is not None]
    if len(candidates) < take:
        candidates += [index for index, signature in enumerate(signatures)
                       if signature is None][: take - len(candidates)]

    mean = np.mean(np.stack([vectors[index] for index in candidates]), axis=0)
    # Seed on the image closest to the mean: starting from an outlier would bias the spread.
    seed_index = candidates[0]
    best_distance = None
    for index in candidates:
        distance = float(np.abs(vectors[index] - mean).mean())
        if best_distance is None or distance < best_distance:
            best_distance, seed_index = distance, index

    chosen = [seed_index]
    # `worst[index]` = similarity to the most similar image already chosen, i.e. the badness that
    # stays when this candidate joins: a duplicate of a chosen image keeps ~100 and is picked last.
    worst = {index: signature_similarity(vectors[index], vectors[seed_index]) for index in candidates}
    while len(chosen) < take:
        pick = -1
        for index in candidates:
            if index in chosen:
                continue
            if pick < 0 or worst[index] < worst[pick]:
                pick = index
        chosen.append(pick)
        for index in candidates:
            if index in chosen:
                continue
            similarity = signature_similarity(vectors[index], vectors[pick])
            if similarity > worst[index]:
                worst[index] = similarity

    max_similarity, median_similarity = _pairwise_stats(chosen, vectors)
    return sorted(chosen), {
        "count": len(chosen),
        "pool": total,
        "max_similarity": max_similarity,
        "median_similarity": median_similarity,
        "unreadable": unreadable,
    }


def _pairwise_stats(
    chosen: Sequence[int],
    vectors: Sequence[np.ndarray],
) -> tuple[Optional[float], Optional[float]]:
    """The chosen set's max and median pairwise similarity, as `signature_similarity` reports it."""
    values: list[float] = []
    for position, left in enumerate(chosen):
        for right in chosen[position + 1:]:
            values.append(signature_similarity(vectors[left], vectors[right]))
    if not values:
        return None, None
    values.sort()
    middle = len(values) // 2
    median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2.0
    return round(values[-1], 1), round(median, 1)
