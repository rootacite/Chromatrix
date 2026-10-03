#!/bin/bash
#
# resize_images.sh
# Resize all input images to 1920x1080 while keeping compression quality high.
#
# Usage:
#   ./resize_images.sh image1.jpg image2.png ...
#   ./resize_images.sh *.jpg
#
# Output is saved to ./output

OUTPUT_DIR="./output"
TARGET_W=1920
TARGET_H=1080

# Check if ffmpeg is installed
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Error: ffmpeg not found. Please install it first." >&2
    exit 1
fi

if [ $# -eq 0 ]; then
    echo "Usage: $0 <image1> [image2] ..." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

ok_count=0
fail_count=0

for img in "$@"; do
    if [ ! -f "$img" ]; then
        echo "Skipping: $img (file not found)"
        fail_count=$((fail_count + 1))
        continue
    fi

    filename=$(basename "$img")
    name="${filename%.*}"
    ext="${filename##*.}"
    ext_lower=$(echo "$ext" | tr '[:upper:]' '[:lower:]')
    output="$OUTPUT_DIR/${name}.${ext}"

    # Choose encoding parameters that preserve quality for each format
    case "$ext_lower" in
        jpg|jpeg)
            # -q:v 2 is very high visual quality for JPEG
            # Range is 1-31, lower is better
            quality=(-q:v 2)
            ;;
        png)
            # PNG is lossless; compression_level only affects file size
            quality=(-compression_level 9)
            ;;
        webp)
            # WebP quality 100 is near-lossless
            quality=(-quality 100 -compression_level 6)
            ;;
        bmp|tiff|tif)
            # Lossless formats need no quality parameter
            quality=()
            ;;
        *)
            quality=(-q:v 2)
            ;;
    esac

    echo "Processing: $img  ->  $output"

    # Scale proportionally to fit inside 1920x1080,
    # then pad with black bars to exactly 1920x1080
    if ffmpeg -hide_banner -loglevel error -y -i "$img" \
        -vf "scale=${TARGET_W}:${TARGET_H}:force_original_aspect_ratio=decrease,pad=${TARGET_W}:${TARGET_H}:(ow-iw)/2:(oh-ih)/2:color=black" \
        "${quality[@]}" \
        "$output"; then
        ok_count=$((ok_count + 1))
    else
        echo "Failed: $img" >&2
        fail_count=$((fail_count + 1))
    fi
done

echo ""
echo "Done: $ok_count succeeded, $fail_count failed"
echo "Output directory: $OUTPUT_DIR"
