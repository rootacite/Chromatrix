package com.acite.axlranko.model

/**
 * The renderer behind a sample pass: the local diffusers pipeline, or a listening ComfyUI running
 * the bundled `beta/Sampling.json`.
 *
 * The values mirror `trainer/genjob.py`'s `BACKEND_*` and `trainer/comfy_sample.py`'s strength
 * bounds. Absent means the built-in path, so a request that sends nothing keeps working.
 */
const val SAMPLE_BACKEND_BUILTIN = "builtin"
const val SAMPLE_BACKEND_COMFY = "comfy"

/** The LoRA strength bounds api.py enforces, and the value the bundled workflow ships with. */
const val SAMPLE_LORA_STRENGTH_MIN = 0.0f
const val SAMPLE_LORA_STRENGTH_MAX = 2.0f
const val SAMPLE_LORA_STRENGTH_DEFAULT = "0.95"

/**
 * What the dialogs let the user edit: the renderer plus the two things only the ComfyUI path
 * needs. The strength is the text the field holds, so a half-typed number is not a value yet;
 * `SampleBackendRules` turns it into a request or into the reason it cannot be one.
 */
data class SampleBackendChoice(
    val backend: String = SAMPLE_BACKEND_BUILTIN,
    /** A name under ComfyUI's `models/checkpoints`; blank keeps the workflow's own base model. */
    val comfyCheckpoint: String = "",
    val comfyLoraStrength: String = SAMPLE_LORA_STRENGTH_DEFAULT,
) {
    val isComfy: Boolean get() = backend == SAMPLE_BACKEND_COMFY

    companion object {
        /** The built-in path: what every dialog opens on until the user picks otherwise. */
        val BuiltIn = SampleBackendChoice()
    }
}

/** One request as the helper takes it: no strings that still need validating. */
data class SampleBackend(
    val backend: String = SAMPLE_BACKEND_BUILTIN,
    val comfyCheckpoint: String = "",
    val comfyLoraStrength: Float? = null,
)

/**
 * Which Checkpoints entry a backend dialog was opened from. The entry's own controls stay where
 * they are (the range row's steps, the pinned row's rounds), so the dialog only has to say what it
 * is about to start.
 */
enum class SamplePassKind {
    SAMPLE_RANGE,
    PINNED,
    CHECKPOINT_SAMPLES,
}

/** The pass a backend dialog will start when it is confirmed, and the line describing it. */
data class SamplePassRequest(
    val kind: SamplePassKind,
    /** Shown under the title: what this pass will render, in the entry's own terms. */
    val detail: String,
    val checkpoint: CheckpointItem? = null,
    val fromStep: Int = 0,
    val toStep: Int = 0,
    val rounds: Int = 1,
)
