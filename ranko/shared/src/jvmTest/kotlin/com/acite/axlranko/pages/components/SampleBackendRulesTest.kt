package com.acite.axlranko.pages.components

import com.acite.axlranko.data.ComfyCheckpointList
import com.acite.axlranko.model.SAMPLE_BACKEND_COMFY
import com.acite.axlranko.model.SampleBackendChoice
import com.acite.axlranko.model.SamplePassKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * The backend picker's rules, without a window: what a click would send, and why it cannot be sent.
 * Every message here has a counterpart in `trainer/comfy_sample.py`, which is the point — a bad
 * click must not cost a spawned process.
 */
class SampleBackendRulesTest {

    private val listed = ComfyCheckpointList(
        root = "/home/acite/LLM/comfyui",
        checkpoints = listOf("waiIllustrious_v170.safetensors", "other.safetensors"),
    )

    @Test
    fun theBuiltInPathAlwaysConfirms() {
        // No ComfyUI, still loading, no list at all: the local path does not depend on any of it.
        assertNull(sampleBackendError(SampleBackendChoice.BuiltIn, null, loading = false))
        assertNull(sampleBackendError(SampleBackendChoice.BuiltIn, null, loading = true))
        assertNull(sampleBackendError(SampleBackendChoice(), ComfyCheckpointList(error = "gone"), loading = false))
    }

    @Test
    fun theComfyUiPathNeedsTheServerTheStrengthAndTheBaseModel() {
        val choice = SampleBackendChoice(backend = SAMPLE_BACKEND_COMFY)
        assertTrue(sampleBackendError(choice, null, loading = false)!!.isNotBlank())
        assertTrue(sampleBackendError(choice, listed, loading = true)!!.contains("ComfyUI"))
        assertEquals(
            "no ComfyUI found listening on this machine",
            sampleBackendError(
                choice,
                ComfyCheckpointList(error = "no ComfyUI found listening on this machine"),
                loading = false,
            ),
        )
        assertNull(sampleBackendError(choice, listed, loading = false))
        assertEquals(
            "ComfyUI has no checkpoint named gone.safetensors",
            sampleBackendError(choice.copy(comfyCheckpoint = "gone.safetensors"), listed, loading = false),
        )
        assertNull(sampleBackendError(choice.copy(comfyCheckpoint = "other.safetensors"), listed, loading = false))
    }

    @Test
    fun theStrengthFieldMirrorsTheHelpersBounds() {
        assertNull(loraStrengthError("0.95"))
        assertNull(loraStrengthError(" 1,5 "))
        assertNull(loraStrengthError("0"))
        assertNull(loraStrengthError("2"))
        assertEquals("Enter a LoRA strength", loraStrengthError("  "))
        assertEquals("LoRA strength must be a number", loraStrengthError("strong"))
        assertTrue(loraStrengthError("-0.1")!!.contains("between 0 and 2"))
        assertTrue(loraStrengthError("2.5")!!.contains("between 0 and 2"))
    }

    @Test
    fun theBuiltInPathSendsNoRendererFields() {
        assertNull(sampleBackendPayload(SampleBackendChoice.BuiltIn))
        val payload = assertNotNull(
            sampleBackendPayload(
                SampleBackendChoice(
                    backend = SAMPLE_BACKEND_COMFY,
                    comfyCheckpoint = " wai.safetensors ",
                    comfyLoraStrength = "0,8",
                )
            )
        )
        assertEquals(SAMPLE_BACKEND_COMFY, payload.backend)
        assertEquals("wai.safetensors", payload.comfyCheckpoint)
        assertEquals(0.8f, payload.comfyLoraStrength)
    }

    @Test
    fun eachEntryNamesWhatItWillRender() {
        assertEquals("Sample range", samplePassTitle(SamplePassKind.SAMPLE_RANGE))
        assertEquals("Generate pinned samples", samplePassTitle(SamplePassKind.PINNED))
        assertEquals("Generate samples", samplePassTitle(SamplePassKind.CHECKPOINT_SAMPLES))

        assertEquals(
            "Steps 1000–3000 · 1 checkpoint",
            samplePassDetail(SamplePassKind.SAMPLE_RANGE, 1, fromStep = 1000, toStep = 3000),
        )
        assertEquals(
            "Steps 1000–3000 · 5 checkpoints",
            samplePassDetail(SamplePassKind.SAMPLE_RANGE, 5, fromStep = 1000, toStep = 3000),
        )
        // The rounds line only appears when there is more than one round to run.
        assertEquals("3 checkpoints pinned", samplePassDetail(SamplePassKind.PINNED, 3))
        assertEquals(
            "3 checkpoints pinned · round 2/2",
            samplePassDetail(SamplePassKind.PINNED, 3, round = 2, rounds = 2),
        )
        assertEquals("The run's single sample set", samplePassDetail(SamplePassKind.CHECKPOINT_SAMPLES, 1))
        assertEquals("The run's 4 sample sets", samplePassDetail(SamplePassKind.CHECKPOINT_SAMPLES, 4))
        assertEquals("The run's sample sets", samplePassDetail(SamplePassKind.CHECKPOINT_SAMPLES, 0))
    }
}
