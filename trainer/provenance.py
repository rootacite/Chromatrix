"""PNG text metadata ("provenance") for the sample images this repo renders.

Every image we draw records how it was drawn: the seed actually used, the prompt pair, the sampling
values, which slot of the run it belongs to and the LoRA settings of the checkpoint it came from.
The keys live under the `axl_` prefix — the convention `tools/mask_blur.py` established for its
`axl_mask_blur` — so they cannot collide with a PNG's own text chunks, and a file copied out of the
run directory still says what it is.

The trainer writes them from `trainer/sampling.py` (its own sample points) and the generator from
`trainer/generate_sample.py` (single, sets and evaluation top-ups). `api.py`'s `regenerate_sample`
reads `axl_seed` back to redraw one image at its own seed.

No repo imports here, so both import styles work (see AGENT.md "Import dualism").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Union

from PIL import Image
from PIL.PngImagePlugin import PngInfo

PREFIX = "axl_"

# The slot an image occupies: the trainer's own samples, a generated sets pass (including range /
# pinned batch children and evaluation top-ups), or the manual single-image form.
SOURCE_TRAINING = "training"
SOURCE_SETS = "sets"
SOURCE_SINGLE = "single"
SOURCE_EVALUATE = "evaluate"

WRITER_TRAINER = "trainer"
WRITER_GENERATOR = "generate_sample"

# The LoRA/base settings copied from the checkpoint's own kohya metadata, key -> `TrainConfig` field.
_LORA_FIELDS = (
    ("network_type", "network_type"),
    ("network_dim", "network_dim"),
    ("network_alpha", "network_alpha"),
    ("conv_dim", "conv_dim"),
    ("conv_alpha", "conv_alpha"),
    ("base_model_version", "base_model_version"),
    ("pretrained_model", "pretrained_model_name_or_path"),
    ("clip_skip", "clip_skip"),
    ("max_token_length", "max_token_length"),
)


def _put(info: PngInfo, key: str, value: Any) -> None:
    """Skip empty values, the rule `sample_provenance` has always used."""
    text = "" if value is None else str(value)
    if text:
        info.add_text(f"{PREFIX}{key}", text)


def lora_fields(cfg: Any) -> dict[str, Any]:
    """The LoRA/base settings of a wrapped pipeline, keyed the way `build_pnginfo` writes them."""
    return {name: getattr(cfg, field, None) for name, field in _LORA_FIELDS}


def build_pnginfo(
    *,
    seed: int,
    prompt: str,
    negative: str = "",
    width: Any = None,
    height: Any = None,
    steps: Any = None,
    guidance: Any = None,
    guidance_rescale: Any = None,
    run_id: str = "",
    output_name: str = "",
    step: Any = None,
    set_index: Optional[int] = None,
    repeat_idx: Optional[int] = None,
    source: str = "",
    writer: str = "",
    checkpoint: str = "",
    lora: Optional[Mapping[str, Any]] = None,
) -> PngInfo:
    """The text chunks for one rendered image.

    `step` is whatever the caller's naming uses (the trainer formats it `%06d`); `set_index` /
    `repeat_idx` are omitted for a manual single image, which belongs to no prompt set.
    """
    info = PngInfo()
    # The order the trainer's own `sample_provenance` wrote, kept so existing files stay comparable.
    _put(info, "run_id", run_id)
    _put(info, "output_name", output_name)
    _put(info, "step", step)
    _put(info, "set", set_index)
    _put(info, "repeat", repeat_idx)
    _put(info, "seed", seed)
    _put(info, "prompt", prompt)
    _put(info, "negative", negative)
    _put(info, "width", width)
    _put(info, "height", height)
    _put(info, "steps", steps)
    _put(info, "guidance", guidance)
    _put(info, "guidance_rescale", guidance_rescale)
    for name, value in (lora or {}).items():
        _put(info, name, value)
    _put(info, "checkpoint", checkpoint)
    _put(info, "source", source)
    _put(info, "writer", writer)
    return info


def read_provenance(path: Union[str, Path]) -> dict[str, str]:
    """The `axl_*` text chunks of one image, empty when it carries none.

    An unreadable image answers empty rather than raising: the caller already checked the file
    exists, and a missing seed is a state the UI has a warning for.
    """
    try:
        with Image.open(path) as image:
            text = dict(getattr(image, "text", None) or {})
    except Exception:  # noqa: BLE001 - a corrupt/unreadable file is "no provenance", not a crash
        return {}
    return {key: value for key, value in text.items() if key.startswith(PREFIX)}
