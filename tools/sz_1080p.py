#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resize all images passed on the command line to 1920x1080 using nproc threads,
while preserving the original compression quality as much as possible.

Quality preservation strategy:
  * JPEG : reuse the original quantization tables (qtables) and chroma subsampling,
           which is equivalent to "same quality"
  * PNG  : lossless format, fixed compression level, pixels are untouched
  * WebP : reuse the original quality (fallback to 80 if unavailable)
  * Also preserves EXIF (including orientation) and ICC color profiles

Usage:
    python resize.py a.jpg b.png c.webp
    python resize.py -o out/ *.jpg              # write output to out/
    python resize.py --fit contain *.png        # scale proportionally and pad
    python resize.py -j 4 *.jpg                 # manually set thread count
"""

import argparse
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from PIL import Image, ImageOps
except ImportError:
    sys.exit("Pillow is required: pip install Pillow")

W, H = 1920, 1080

# Resampling constant compatible with both old and new Pillow versions
LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def cpu_count() -> int:
    """Equivalent to shell's nproc (respects CPU affinity)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def build_save_kwargs(fmt: str, qtables, src_info: dict, exif, icc) -> dict:
    """Build save arguments from the original image info, preserving compression quality."""
    kw = {}
    fmt = fmt.upper()

    if fmt == "JPEG":
        if qtables:
            # Reuse original quantization tables => exact same compression quality
            kw["qtables"] = qtables
            kw["subsampling"] = "keep"        # keep original chroma subsampling (4:4:4 / 4:2:0 ...)
        else:
            kw["quality"] = "keep"
        kw["progressive"] = bool(src_info.get("progressive", False))
        kw["optimize"] = False

    elif fmt == "PNG":
        kw["compress_level"] = 6              # PNG is lossless; level only affects size, not quality
        kw["optimize"] = False

    elif fmt == "WEBP":
        kw["quality"] = src_info.get("quality", 80)
        kw["method"] = 4

    elif fmt == "TIFF":
        kw["compression"] = "tiff_deflate"

    if exif:
        kw["exif"] = exif
    if icc:
        kw["icc_profile"] = icc
    return kw


def resize_one(src_path: str, dst_path: str, fit: str) -> str:
    """Process a single image and return the destination path. Exceptions are raised to the caller."""
    with Image.open(src_path) as src:
        fmt = src.format
        if not fmt:
            raise ValueError("Unrecognized image format")

        # These must be read before transpose (transpose creates a new object and loses quantization)
        qtables = getattr(src, "quantization", None)
        src_info = dict(src.info)

        # Apply EXIF orientation to avoid resizing a "lying down" image
        im = ImageOps.exif_transpose(src)

        if fmt == "JPEG" and im.mode not in ("RGB", "L", "CMYK"):
            im = im.convert("RGB")

        # ---- Resize ----
        if fit == "stretch":                  # stretch directly to 1920x1080
            im = im.resize((W, H), LANCZOS)
        elif fit == "contain":                # scale proportionally, pad with black
            im = ImageOps.pad(im, (W, H), method=LANCZOS, color=(0, 0, 0))
        else:                                 # cover: scale proportionally and center-crop
            im = ImageOps.fit(im, (W, H), method=LANCZOS)

        kw = build_save_kwargs(
            fmt, qtables, src_info, im.info.get("exif"), src_info.get("icc_profile")
        )

        # ---- Atomic write: write to a temp file first, then replace, so a failure won't corrupt the original ----
        out_dir = os.path.dirname(os.path.abspath(dst_path)) or "."
        fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=os.path.splitext(dst_path)[1])
        os.close(fd)
        try:
            im.save(tmp, format=fmt, **kw)
            os.replace(tmp, dst_path)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    return dst_path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Resize images to 1920x1080 using multiple threads (preserving compression quality)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("images", nargs="+", help="image paths to process")
    ap.add_argument("-o", "--output-dir", default=None,
                    help="output directory; if not specified, overwrite original files in place")
    ap.add_argument("-j", "--workers", type=int, default=cpu_count(),
                    help="number of threads")
    ap.add_argument("--fit", choices=("stretch", "contain", "cover"), default="stretch",
                    help="stretch=stretch to target size, contain=proportional + black bars, cover=proportional + crop")
    args = ap.parse_args()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    jobs = []
    for p in args.images:
        if not os.path.isfile(p):
            print(f"[SKIP] Not a file: {p}", file=sys.stderr)
            continue
        dst = os.path.join(args.output_dir, os.path.basename(p)) if args.output_dir else p
        jobs.append((p, dst))

    if not jobs:
        print("No images to process", file=sys.stderr)
        return 1

    workers = max(1, args.workers)
    ok = fail = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(resize_one, s, d, args.fit): s for s, d in jobs}
        for fut in as_completed(futures):
            src = futures[fut]
            try:
                fut.result()
                ok += 1
                print(f"[OK]   {src}")
            except Exception as e:
                fail += 1
                print(f"[FAIL] {src}: {e}", file=sys.stderr)

    print(f"\nDone: {ok} succeeded, {fail} failed (threads: {workers})")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
