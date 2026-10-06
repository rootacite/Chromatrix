package com.acite.axlranko.pages.components

import com.acite.axlranko.data.ComfyCheckpointList
import com.acite.axlranko.model.SampleBackend
import com.acite.axlranko.model.SampleBackendChoice
import com.acite.axlranko.model.SamplePassKind
import com.acite.axlranko.model.SAMPLE_BACKEND_COMFY
import com.acite.axlranko.model.SAMPLE_LORA_STRENGTH_MAX
import com.acite.axlranko.model.SAMPLE_LORA_STRENGTH_MIN

/**
 * The backend picker's own rules, pure and testable without a window.
 *
 * Everything here mirrors a refusal `api.py` would make, so the button can be off with the reason
 * beside it before a click costs a spawned process. The strengths and the message shape are
 * `trainer/comfy_sample.py`'s.
 */

/** The title of the dialog an entry opens. */
internal fun samplePassTitle(kind: SamplePassKind): String = when (kind) {
    SamplePassKind.SAMPLE_RANGE -> "Sample range"
    SamplePassKind.PINNED -> "Generate pinned samples"
    SamplePassKind.CHECKPOINT_SAMPLES -> "Generate samples"
}

/**
 * What this pass will render, in the entry's own terms — the range row and the pinned row keep
 * their own fields, so the dialog only repeats what they already say.
 */
internal fun samplePassDetail(
    kind: SamplePassKind,
    count: Int,
    fromStep: Int = 0,
    toStep: Int = 0,
    round: Int = 1,
    rounds: Int = 1,
): String = when (kind) {
    SamplePassKind.SAMPLE_RANGE -> "Steps $fromStep–$toStep · ${checkpointCount(count)}"
    SamplePassKind.PINNED -> buildString {
        append("${checkpointCount(count)} pinned")
        if (rounds > 1) append(" · round $round/$rounds")
    }
    SamplePassKind.CHECKPOINT_SAMPLES -> when {
        count == 1 -> "The run's single sample set"
        count > 1 -> "The run's $count sample sets"
        // The prompt sets were not read (the section was never opened): say what it renders.
        else -> "The run's sample sets"
    }
}

private fun checkpointCount(count: Int): String =
    if (count == 1) "1 checkpoint" else "$count checkpoints"

/** Why the ComfyUI path cannot be confirmed yet, or null when it can. The built-in path is always ready. */
internal fun sampleBackendError(
    choice: SampleBackendChoice,
    checkpoints: ComfyCheckpointList?,
    loading: Boolean,
): String? {
    if (!choice.isComfy) return null
    if (loading) return "Looking for ComfyUI…"
    if (checkpoints == null) return "ComfyUI was not checked yet"
    if (checkpoints.root.isBlank()) {
        return checkpoints.error.ifBlank { "No ComfyUI is listening on this machine" }
    }
    loraStrengthError(choice.comfyLoraStrength)?.let { return it }
    val name = choice.comfyCheckpoint.trim()
    if (name.isNotEmpty() && name !in checkpoints.checkpoints) {
        return "ComfyUI has no checkpoint named $name"
    }
    return null
}

/** The LoRA strength the field holds, or the reason it is not a number api.py would take. */
internal fun loraStrengthError(text: String): String? {
    val trimmed = text.trim().replace(',', '.')
    if (trimmed.isEmpty()) return "Enter a LoRA strength"
    val value = trimmed.toFloatOrNull() ?: return "LoRA strength must be a number"
    if (value < SAMPLE_LORA_STRENGTH_MIN || value > SAMPLE_LORA_STRENGTH_MAX) {
        return "LoRA strength must be between ${SAMPLE_LORA_STRENGTH_MIN.toInt()} and " +
            "${SAMPLE_LORA_STRENGTH_MAX.toInt()}"
    }
    return null
}

/** What the request carries. The built-in path sends nothing at all, which is what an absent
 * `backend` means to the helper. */
internal fun sampleBackendPayload(choice: SampleBackendChoice): SampleBackend? {
    if (!choice.isComfy) return null
    return SampleBackend(
        backend = SAMPLE_BACKEND_COMFY,
        comfyCheckpoint = choice.comfyCheckpoint.trim(),
        comfyLoraStrength = choice.comfyLoraStrength.trim().replace(',', '.').toFloatOrNull(),
    )
}
