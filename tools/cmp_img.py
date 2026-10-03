#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".avif",
}


def load_image(path: Path) -> Optional[np.ndarray]:
    """Load an image as RGB uint8 array."""
    try:
        with Image.open(path) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        print(f"[WARN] Failed to read {path}: {exc}")
        return None


def similarity(target: np.ndarray, candidate: np.ndarray) -> float:
    """
    Calculate similarity using mean absolute pixel difference.

    100.0 = identical
    0.0   = maximum possible RGB difference
    """
    diff = np.abs(
        target.astype(np.int16) - candidate.astype(np.int16)
    )

    mean_diff = float(diff.mean())

    return (1.0 - mean_diff / 255.0) * 100.0


def compare_one(
    path: Path,
    target: np.ndarray,
) -> Optional[tuple[float, Path]]:
    """Compare one candidate image against the target."""
    candidate = load_image(path)

    if candidate is None:
        return None

    if candidate.shape != target.shape:
        return None

    score = similarity(target, candidate)
    return score, path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find the most similar images in the same directory."
    )

    parser.add_argument(
        "image",
        type=Path,
        help="Target image path",
    )

    parser.add_argument(
        "-n",
        "--nproc",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of worker threads (default: CPU count)",
    )

    parser.add_argument(
        "-k",
        "--top",
        type=int,
        default=20,
        help="Number of results to show (default: 20)",
    )

    parser.add_argument(
        "-c",
        "--copy-to",
        type=Path,
        default=None,
        help="Copy or move matching images to this directory",
    )

    parser.add_argument(
        "-l",
        "--level",
        type=float,
        default=None,
        help="Minimum similarity percentage for copying/moving",
    )

    parser.add_argument(
        "-m",
        "--move",
        action="store_true",
        help="Move matching images instead of copying them",
    )

    args = parser.parse_args()

    if args.nproc < 1:
        parser.error("--nproc must be >= 1")

    if args.top < 1:
        parser.error("--top must be >= 1")

    if args.level is not None and not (0.0 <= args.level <= 100.0):
        parser.error("--level must be between 0 and 100")

    if args.copy_to is None and args.level is not None:
        parser.error("--level requires --copy-to")

    if args.copy_to is not None and args.level is None:
        parser.error("--copy-to requires --level")

    if args.move and args.copy_to is None:
        parser.error("--move requires --copy-to")

    if args.move and args.level is None:
        parser.error("--move requires --level")

    target_path = args.image.expanduser().resolve()

    if not target_path.is_file():
        print(f"[ERROR] File does not exist: {target_path}")
        return 1

    target = load_image(target_path)

    if target is None:
        return 1

    height, width, _ = target.shape

    print(
        f"Target : {target_path.name}\n"
        f"Size   : {width}x{height}, RGB\n"
        f"Workers: {args.nproc}\n"
    )

    # Find images in the same directory with matching dimensions.
    candidates: list[Path] = []

    for path in target_path.parent.iterdir():
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        try:
            with Image.open(path) as img:
                if img.size == (width, height):
                    candidates.append(path)
        except Exception:
            pass

    print(
        f"Found {len(candidates)} image(s) with matching dimensions.\n"
    )

    results: list[tuple[float, Path]] = []

    # The target itself is always included when copy/move mode is enabled.
    compare_candidates = [
        path for path in candidates
        if path.resolve() != target_path
    ]

    with ThreadPoolExecutor(max_workers=args.nproc) as executor:
        futures = [
            executor.submit(compare_one, path, target)
            for path in compare_candidates
        ]

        for future in futures:
            result = future.result()
            if result is not None:
                results.append(result)

    results.sort(key=lambda x: x[0], reverse=True)

    print("Similarity:")
    for score, path in results[:args.top]:
        print(f"{score:9.4f}%  {path.name}")

    # Copy/move mode.
    if args.copy_to is not None and args.level is not None:
        destination_dir = args.copy_to.expanduser().resolve()
        destination_dir.mkdir(parents=True, exist_ok=True)

        matched = [
            (score, path)
            for score, path in results
            if score >= args.level
        ]

        # Include the target image itself.
        if args.level <= 100.0:
            matched.insert(0, (100.0, target_path))

        action = "move" if args.move else "copy"
        action_gerund = "Moving" if args.move else "Copying"

        print(
            f"\n{action_gerund} {len(matched)} image(s) "
            f"with similarity >= {args.level:.4f}% "
            f"to: {destination_dir}"
        )

        for score, path in matched:
            destination = destination_dir / path.name

            try:
                # Avoid trying to copy/move a file onto itself.
                if path.resolve() == destination.resolve():
                    print(
                        f"  {score:9.4f}%  {path.name} "
                        f"(already in destination)"
                    )
                    continue

                if args.move:
                    shutil.move(str(path), str(destination))
                else:
                    shutil.copy2(path, destination)

                print(f"  {score:9.4f}%  {path.name}")

            except OSError as exc:
                print(
                    f"[WARN] Failed to {action} {path.name}: {exc}"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

