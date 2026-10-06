package com.acite.axlranko.pages

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.ImageComposeScene
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.input.pointer.PointerButton
import androidx.compose.ui.input.pointer.PointerButtons
import androidx.compose.ui.input.pointer.PointerEventType
import androidx.compose.ui.semantics.SemanticsNode
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.semantics.getOrNull
import androidx.compose.ui.unit.Density
import com.acite.axlranko.data.ComfyCheckpointList
import com.acite.axlranko.model.SAMPLE_BACKEND_COMFY
import com.acite.axlranko.model.SampleBackend
import com.acite.axlranko.model.SampleBackendChoice
import com.acite.axlranko.model.SamplePassKind
import com.acite.axlranko.model.SamplePassRequest
import com.acite.axlranko.pages.components.SampleBackendDialog
import com.acite.axlranko.ui.theme.RankoTheme
import java.awt.GraphicsEnvironment
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * The renderer picker is composed for real and clicked: the built-in path confirms with no renderer
 * fields at all, picking ComfyUI has to bring up the base-model list and the strength field, and
 * the button stays off — with the reason on screen — while ComfyUI cannot serve the request.
 *
 * `ImageComposeScene` draws a dialog's layer inside its own scene, so what is asserted is the screen
 * after the click rather than the state behind it. Skipped on a headless JVM.
 */
class SampleBackendDialogRenderTest {

    private val listed = ComfyCheckpointList(
        root = "/home/acite/LLM/comfyui",
        checkpoints = listOf("waiIllustrious_v170.safetensors"),
    )

    /** What a `render` call saw: whether Start was reached, and what it sent. */
    private data class Confirmation(val sent: Boolean, val backend: SampleBackend?)

    private fun walk(node: SemanticsNode, out: MutableList<SemanticsNode>) {
        out += node
        node.children.forEach { walk(it, out) }
    }

    private fun nodes(scene: ImageComposeScene): List<SemanticsNode> {
        scene.render()
        val found = mutableListOf<SemanticsNode>()
        scene.semanticsOwners.forEach { owner -> walk(owner.rootSemanticsNode, found) }
        return found
    }

    private fun nodeWithText(nodes: List<SemanticsNode>, text: String): SemanticsNode? =
        nodes.firstOrNull { node ->
            node.config.getOrNull(SemanticsProperties.Text).orEmpty().any { it.text == text }
        }

    /** A text field's value is not a `Text` node: Compose reports it as editable text. */
    private fun nodeWithValue(nodes: List<SemanticsNode>, text: String): SemanticsNode? =
        nodes.firstOrNull { node ->
            node.config.getOrNull(SemanticsProperties.EditableText)?.text == text
        }

    private fun click(scene: ImageComposeScene, node: SemanticsNode) {
        val at = Offset(
            node.positionInRoot.x + node.size.width / 2f,
            node.positionInRoot.y + node.size.height / 2f,
        )
        scene.sendPointerEvent(
            PointerEventType.Press,
            at,
            button = PointerButton.Primary,
            buttons = PointerButtons(isPrimaryPressed = true),
        )
        scene.sendPointerEvent(
            PointerEventType.Release,
            at,
            button = PointerButton.Primary,
            buttons = PointerButtons(),
        )
        scene.render()
    }

    private fun startNode(nodes: List<SemanticsNode>): SemanticsNode =
        assertNotNull(nodeWithText(nodes, "Start"), "the dialog offers a Start button")

    private fun disabled(node: SemanticsNode): Boolean =
        node.config.getOrNull(SemanticsProperties.Disabled) != null

    /**
     * Composes the dialog on one choice, optionally clicking Start the way a user would, and hands
     * the scene to [inspect] before it is measured.
     */
    private fun render(
        request: SamplePassRequest,
        choice: SampleBackendChoice,
        checkpoints: ComfyCheckpointList?,
        starting: Boolean = false,
        error: String? = null,
        inspect: (ImageComposeScene) -> Unit = {},
    ): Confirmation {
        if (GraphicsEnvironment.isHeadless()) return Confirmation(false, null)
        var confirmed: SampleBackend? = null
        var reached = false
        val scene = ImageComposeScene(width = 900, height = 900, density = Density(1f)) {
            RankoTheme {
                SampleBackendDialog(
                    request = request,
                    choice = choice,
                    checkpoints = checkpoints,
                    checkpointsLoading = false,
                    checkpointsError = checkpoints?.error?.ifBlank { null },
                    starting = starting,
                    error = error,
                    onChoiceChange = {},
                    onRefreshCheckpoints = {},
                    onConfirm = { confirmed = it; reached = true },
                    onDismiss = {},
                )
            }
        }
        try {
            inspect(scene)
            scene.render()
        } finally {
            scene.close()
        }
        return Confirmation(reached, confirmed)
    }

    /** Clicks Start on the scene's own button; a disabled button simply does nothing. */
    private fun confirm(scene: ImageComposeScene) {
        click(scene, startNode(nodes(scene)))
    }

    @Test
    fun theBuiltInPathIsTheDefaultAndStartsImmediately() {
        val sent = render(
            request = SamplePassRequest(
                kind = SamplePassKind.SAMPLE_RANGE,
                detail = "Steps 1000–3000 · 2 checkpoints",
                fromStep = 1000,
                toStep = 3000,
            ),
            choice = SampleBackendChoice.BuiltIn,
            checkpoints = null,
        ) { scene ->
            val open = nodes(scene)
            assertNotNull(nodeWithText(open, "Sample range"), "the entry names itself")
            assertNotNull(nodeWithText(open, "Steps 1000–3000 · 2 checkpoints"), "the pass is described")
            assertNotNull(nodeWithText(open, "Built-in"))
            assertNotNull(nodeWithText(open, "ComfyUI"))
            // The ComfyUI fields stay out of the way until that path is picked, and the button is
            // live even with no ComfyUI in sight: the local path needs neither.
            assertNull(nodeWithText(open, "Base model"), "no base model on the built-in path")
            assertNull(nodeWithText(open, "LoRA strength"), "no strength on the built-in path")
            assertFalse(disabled(startNode(open)), "the built-in path can start at once")
            confirm(scene)
        }
        assertTrue(sent.sent, "Start was reached")
        assertNull(sent.backend, "the built-in path sends no renderer fields")
    }

    @Test
    fun pickingComfyUiBringsUpTheBaseModelAndTheStrength() {
        val confirmation = render(
            request = SamplePassRequest(
                kind = SamplePassKind.CHECKPOINT_SAMPLES,
                detail = "The run's 2 sample sets",
            ),
            choice = SampleBackendChoice(
                backend = SAMPLE_BACKEND_COMFY,
                comfyCheckpoint = "waiIllustrious_v170.safetensors",
                comfyLoraStrength = "0.8",
            ),
            checkpoints = listed,
        ) { scene ->
            val open = nodes(scene)
            assertNotNull(nodeWithText(open, "Base model"), "the base-model dropdown is shown")
            assertNotNull(nodeWithText(open, "LoRA strength"), "the strength field is shown")
            assertNotNull(nodeWithValue(open, "waiIllustrious_v170.safetensors"), "the picked model is shown")
            assertNotNull(nodeWithText(open, "/home/acite/LLM/comfyui"), "the install root is named")
            assertFalse(disabled(startNode(open)), "a usable ComfyUI leaves Start live")
            confirm(scene)
        }
        assertTrue(confirmation.sent, "Start was reached")
        val payload = assertNotNull(confirmation.backend, "the ComfyUI path sends its fields")
        assertEquals(SAMPLE_BACKEND_COMFY, payload.backend)
        assertEquals("waiIllustrious_v170.safetensors", payload.comfyCheckpoint)
        assertEquals(0.8f, payload.comfyLoraStrength)
    }

    @Test
    fun clickingComfyUiSwitchesTheDialogOver() {
        if (GraphicsEnvironment.isHeadless()) return
        var choice by mutableStateOf(SampleBackendChoice.BuiltIn)
        val scene = ImageComposeScene(width = 900, height = 900, density = Density(1f)) {
            RankoTheme {
                SampleBackendDialog(
                    request = SamplePassRequest(SamplePassKind.PINNED, "2 checkpoints pinned"),
                    choice = choice,
                    checkpoints = listed,
                    checkpointsLoading = false,
                    checkpointsError = null,
                    starting = false,
                    error = null,
                    onChoiceChange = { choice = it },
                    onRefreshCheckpoints = {},
                    onConfirm = {},
                    onDismiss = {},
                )
            }
        }
        try {
            assertNull(nodeWithText(nodes(scene), "Base model"), "the built-in path hides them")
            click(scene, assertNotNull(nodeWithText(nodes(scene), "ComfyUI")))
            assertEquals(SAMPLE_BACKEND_COMFY, choice.backend, "the click reports the ComfyUI path")
            val switched = nodes(scene)
            assertNotNull(nodeWithText(switched, "Base model"), "the base-model dropdown appears")
            assertNotNull(nodeWithText(switched, "LoRA strength"), "the strength field appears")
        } finally {
            scene.close()
        }
    }

    @Test
    fun theComfyUiPathStaysOffWhileNoComfyUiAnswers() {
        val confirmation = render(
            request = SamplePassRequest(SamplePassKind.PINNED, "2 checkpoints pinned"),
            choice = SampleBackendChoice(backend = SAMPLE_BACKEND_COMFY),
            checkpoints = ComfyCheckpointList(error = "no ComfyUI found listening on this machine"),
        ) { scene ->
            val open = nodes(scene)
            assertNotNull(
                nodeWithText(open, "no ComfyUI found listening on this machine"),
                "the reason is on screen",
            )
            assertTrue(disabled(startNode(open)), "Start is off while no ComfyUI answers")
            confirm(scene)
        }
        assertTrue(!confirmation.sent, "a refused dialog confirms nothing")
    }

    @Test
    fun aStrengthOutOfRangeKeepsStartOffWithTheReason() {
        val confirmation = render(
            request = SamplePassRequest(SamplePassKind.CHECKPOINT_SAMPLES, "The run's 2 sample sets"),
            choice = SampleBackendChoice(
                backend = SAMPLE_BACKEND_COMFY,
                comfyLoraStrength = "9",
            ),
            checkpoints = listed,
        ) { scene ->
            val open = nodes(scene)
            assertNotNull(nodeWithText(open, "LoRA strength must be between 0 and 2"), "the bound is named")
            assertTrue(disabled(startNode(open)), "Start is off")
            confirm(scene)
        }
        assertTrue(!confirmation.sent, "an invalid strength confirms nothing")
    }

    @Test
    fun aFailedStartIsReportedOnTheDialog() {
        render(
            request = SamplePassRequest(SamplePassKind.SAMPLE_RANGE, "Steps 0–100 · 1 checkpoint"),
            choice = SampleBackendChoice.BuiltIn,
            checkpoints = null,
            error = "a generation is already running (rein_s0-100_batch_gen_1)",
        ) { scene ->
            assertNotNull(
                nodeWithText(nodes(scene), "a generation is already running (rein_s0-100_batch_gen_1)"),
                "the helper's refusal is shown where it was asked for",
            )
        }
    }

    @Test
    fun aStartInFlightCannotBeClickedTwice() {
        val confirmation = render(
            request = SamplePassRequest(SamplePassKind.CHECKPOINT_SAMPLES, "The run's 2 sample sets"),
            choice = SampleBackendChoice.BuiltIn,
            checkpoints = null,
            starting = true,
        ) { scene ->
            val open = nodes(scene)
            val busy = assertNotNull(nodeWithText(open, "Starting…"), "the button says what is happening")
            assertTrue(disabled(busy), "the button is inert while the request is out")
            assertNull(nodeWithText(open, "Start"), "there is no second Start to click")
        }
        assertTrue(!confirmation.sent, "nothing was confirmed while starting")
    }
}
