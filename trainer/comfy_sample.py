"""Render a checkpoint's sample sets through ComfyUI, on the bundled `beta/Sampling.json`.

The Dashboard's Checkpoints section renders a LoRA's sample images either locally (the diffusers
path in `trainer/generate_sample.py`) or through a listening ComfyUI. This module is the second
one: the workflow is bundled and fixed, so its node ids are constants the way
`run_automation.py` hard-codes the Universal workflow's, and everything the run's prompt sets say
(prompt, negative, size, steps, CFG, RescaleCFG multiplier, seed) is written into it. The two
things the user picks — the base model and the LoRA strength — come from the request instead.

The LoRA is the checkpoint being sampled. ComfyUI can only load a model by name from its own
`models/loras`, so the file is copied there under `axl-sample/` with a name unique to the pass and
removed when the pass ends (`stage_lora` / `unstage_lora`); one process stages at most one copy.

Torch-free on purpose, like `comfy.py` and `automation.py`: `api.py` imports this to plan a job,
and `generate_sample.py` runs it in its own process.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple, Optional

# `python trainer/generate_sample.py` puts trainer/ on sys.path, `import api` does not;
# support both (see AGENT.md "Import dualism").
try:
    import automation
    import comfy
    import genjob
    import run_automation
except ImportError:
    from trainer import automation
    from trainer import comfy
    from trainer import genjob
    from trainer import run_automation

# `beta/Sampling.json`: positive prompt, negative prompt, KSampler, seed node, empty latent,
# RescaleCFG, LoRA and the checkpoint loader. The sampler's `seed` input is a link to the seed
# node, so the seed is written on the node, as `run_automation.set_seed` does for its own graph.
SAMPLING_WORKFLOW_NAME = "Sampling.json"
SAMPLING_POSITIVE_NODE = "215"
SAMPLING_NEGATIVE_NODE = "216"
SAMPLING_SAMPLER_NODE = "19"
SAMPLING_SEED_NODE = "224"
SAMPLING_LATENT_NODE = "217"
SAMPLING_RESCALE_NODE = "207:208"
SAMPLING_LORA_NODE = "207:219"
SAMPLING_CHECKPOINT_NODE = "207:266"

# node id -> the inputs an override writes, so a swapped workflow is refused before anything runs.
_SAMPLING_INPUTS: dict[str, tuple[str, ...]] = {
    SAMPLING_POSITIVE_NODE: ("text",),
    SAMPLING_NEGATIVE_NODE: ("text",),
    SAMPLING_SAMPLER_NODE: ("steps", "cfg"),
    SAMPLING_SEED_NODE: ("seed",),
    SAMPLING_LATENT_NODE: ("width", "height"),
    SAMPLING_RESCALE_NODE: ("multiplier",),
    SAMPLING_LORA_NODE: ("lora_name", "strength_model", "strength_clip"),
    SAMPLING_CHECKPOINT_NODE: ("ckpt_name",),
}

# Where a staged LoRA goes, and the bounds api.py and Chromatrix both enforce.
LORA_STAGE_DIRNAME = "axl-sample"
DEFAULT_LORA_STRENGTH = 0.95
MIN_LORA_STRENGTH = 0.0
MAX_LORA_STRENGTH = 2.0

_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")


def sampling_workflow_path() -> Path:
    return automation.repo_root() / "beta" / SAMPLING_WORKFLOW_NAME


def require_sampling_nodes(workflow: Mapping[str, Any]) -> None:
    """Every node and input an override writes. Raises before a prompt is queued."""
    for node_id, inputs in _SAMPLING_INPUTS.items():
        node = workflow.get(node_id)
        node_inputs = node.get("inputs") if isinstance(node, dict) else None
        if not isinstance(node_inputs, dict):
            raise comfy.ComfyError(
                f"sampling workflow node {node_id!r} has no inputs object"
            )
        for name in inputs:
            if name not in node_inputs:
                raise comfy.ComfyError(
                    f"sampling workflow node {node_id!r} has no {name!r} input"
                )


def apply_sampling(
    workflow: dict[str, Any],
    *,
    prompt: str,
    negative: str,
    steps: int,
    cfg: float,
    rescale: float,
    width: int,
    height: int,
    seed: int,
    lora_name: str,
    strength: float,
    checkpoint_name: str = "",
) -> None:
    """Write one sample set onto the workflow.

    `rescale` is the set's own `guidance_rescale`, which is what the local path hands to diffusers
    and means the same thing here: RescaleCFG's multiplier, `0` = no rescale. A blank
    `checkpoint_name` keeps the base model the bundled workflow already names.
    """
    require_sampling_nodes(workflow)
    inputs = workflow[SAMPLING_POSITIVE_NODE]["inputs"]
    inputs["text"] = str(prompt)
    inputs = workflow[SAMPLING_NEGATIVE_NODE]["inputs"]
    inputs["text"] = str(negative or "")
    inputs = workflow[SAMPLING_SAMPLER_NODE]["inputs"]
    inputs["steps"] = int(steps)
    inputs["cfg"] = float(cfg)
    inputs = workflow[SAMPLING_SEED_NODE]["inputs"]
    inputs["seed"] = int(seed)
    inputs = workflow[SAMPLING_LATENT_NODE]["inputs"]
    inputs["width"] = int(width)
    inputs["height"] = int(height)
    inputs = workflow[SAMPLING_RESCALE_NODE]["inputs"]
    inputs["multiplier"] = float(rescale)
    inputs = workflow[SAMPLING_LORA_NODE]["inputs"]
    inputs["lora_name"] = str(lora_name)
    inputs["strength_model"] = float(strength)
    inputs["strength_clip"] = float(strength)
    chosen = str(checkpoint_name or "").strip()
    if chosen:
        workflow[SAMPLING_CHECKPOINT_NODE]["inputs"]["ckpt_name"] = chosen


def normalize_strength(value: Any, default: float = DEFAULT_LORA_STRENGTH) -> float:
    """The LoRA strength a request asks for; absent means the bundled workflow's own value."""
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        raise ValueError("lora strength must be a number")
    try:
        strength = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("lora strength must be a number") from exc
    if not (MIN_LORA_STRENGTH <= strength <= MAX_LORA_STRENGTH):
        raise ValueError(
            f"lora strength must be between {MIN_LORA_STRENGTH:g} and {MAX_LORA_STRENGTH:g}"
        )
    return strength


def normalize_checkpoint_name(value: Any) -> str:
    """The base model a request names; blank keeps the workflow's own."""
    return str(value or "").strip()


class StagedLora(NamedTuple):
    """The copy ComfyUI loads by `name` and the file this process removes when the pass ends."""

    name: str
    path: Path


def stage_lora(server: str, checkpoint: Path, job_id: str) -> StagedLora:
    """Copy one checkpoint under `models/loras/axl-sample` and name it for `LoraLoader`.

    The name carries the job id, so two runs whose checkpoints share a file name can never load
    each other's weights, and the copy is what `unstage_lora` takes away again.
    """
    found = comfy.loras_for_server(server)
    root = str(found.get("root") or "")
    if not root:
        raise comfy.ComfyError(str(found.get("error") or "ComfyUI's LoRA folder was not found"))
    loras = Path(root) / "models" / "loras"
    staged = loras / LORA_STAGE_DIRNAME / f"{_stage_name(job_id)}.safetensors"
    staged.parent.mkdir(parents=True, exist_ok=True)
    tmp = staged.with_name(f".{staged.name}.{os.getpid()}.tmp")
    shutil.copyfile(str(checkpoint), str(tmp))
    os.replace(tmp, staged)
    return StagedLora(name=staged.relative_to(loras).as_posix(), path=staged)


def unstage_lora(staged: Optional[StagedLora]) -> None:
    """Remove a staged copy. A file that is already gone (or unremovable) is not a failure."""
    if staged is None:
        return
    try:
        Path(staged.path).unlink()
    except OSError:
        pass


def _stage_name(job_id: Any) -> str:
    cleaned = _UNSAFE_NAME.sub("_", str(job_id or "")).strip(".")
    return cleaned[:120] or "sample"


def open_sampler(
    server: str,
    *,
    lora_name: str,
    strength: float,
    checkpoint_name: str = "",
) -> "ComfySampler":
    """Load the bundled workflow and check it, once for a whole pass."""
    if not str(lora_name or "").strip():
        raise comfy.ComfyError("a LoRA name is required")
    workflow = automation.load_workflow(sampling_workflow_path())
    require_sampling_nodes(workflow)
    return ComfySampler(
        server,
        workflow=workflow,
        lora_name=str(lora_name),
        strength=float(strength),
        checkpoint_name=str(checkpoint_name or ""),
    )


class ComfySampler:
    """One ComfyUI session: the bundled graph plus this pass's fixed overrides."""

    def __init__(
        self,
        server: str,
        *,
        workflow: Mapping[str, Any],
        lora_name: str,
        strength: float,
        checkpoint_name: str = "",
        client: Optional[comfy.ComfyClient] = None,
        poll_interval: float = 0.5,
    ) -> None:
        self.client = client or comfy.ComfyClient(server)
        self.poll_interval = float(poll_interval)
        self._workflow = copy.deepcopy(dict(workflow))
        self._save_nodes = {
            str(node_id)
            for node_id, node in self._workflow.items()
            if isinstance(node, dict) and node.get("class_type") == "SaveImage"
        }
        self._lora_name = lora_name
        self._strength = float(strength)
        self._checkpoint_name = checkpoint_name

    def render(
        self,
        sample_set: Any,
        seed: int,
        *,
        should_stop: Optional[Callable[[], bool]] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> bytes:
        """Queue one image of `sample_set` at `seed` and return its PNG bytes.

        `on_progress` is called as `(current_step, total_steps)` while ComfyUI denoises, which is
        the only step-level progress it reports for a queued prompt.
        """
        workflow = copy.deepcopy(self._workflow)
        apply_sampling(
            workflow,
            prompt=sample_set.prompt,
            negative=sample_set.negative,
            steps=sample_set.steps,
            cfg=sample_set.guidance_scale,
            rescale=sample_set.guidance_rescale,
            width=sample_set.width,
            height=sample_set.height,
            seed=seed,
            lora_name=self._lora_name,
            strength=self._strength,
            checkpoint_name=self._checkpoint_name,
        )
        prompt_id = self.client.queue_prompt(workflow)
        entry = self._wait(prompt_id, should_stop=should_stop, on_progress=on_progress)
        found = run_automation.saved_images(entry, self._save_nodes)
        if not found:
            raise comfy.ComfyError("the prompt finished but SaveImage returned no image")
        image = found[0]
        return self.client.get_image(image["filename"], image["subfolder"], image["type"])

    def interrupt(self) -> None:
        """Stop ComfyUI's current execution, so a cancelled pass frees the GPU."""
        try:
            self.client.interrupt()
        except comfy.ComfyError:
            pass

    def _wait(
        self,
        prompt_id: str,
        *,
        should_stop: Optional[Callable[[], bool]],
        on_progress: Optional[Callable[[int, int], None]],
    ) -> dict[str, Any]:
        last_report = 0.0
        while True:
            if should_stop is not None and should_stop():
                raise comfy.ComfyCancelled("cancelled while waiting for ComfyUI")
            entry = self.client.history(prompt_id).get(prompt_id)
            if entry is not None:
                status = entry.get("status") or {}
                if status.get("completed"):
                    if status.get("status_str") != "success":
                        raise comfy.ComfyError(
                            "ComfyUI execution failed: "
                            + json.dumps(status.get("messages"), ensure_ascii=False)[:800]
                        )
                    return entry
            now = time.monotonic()
            if on_progress is not None and now - last_report >= self.poll_interval:
                last_report = now
                value = self._current_step(prompt_id)
                if value is not None:
                    on_progress(*value)
            time.sleep(self.poll_interval)

    def _current_step(self, prompt_id: str) -> Optional[tuple[int, int]]:
        try:
            payload = self.client.progress()
        except comfy.ComfyError:
            return None
        if str(payload.get("prompt_id") or "") != prompt_id:
            return None
        value = payload.get("value")
        maximum = payload.get("max")
        if isinstance(value, int) and isinstance(maximum, int) and maximum > 0:
            return value, maximum
        return None
