package com.acite.axlranko.pages

import androidx.compose.ui.ImageComposeScene
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.input.pointer.PointerButton
import androidx.compose.ui.input.pointer.PointerButtons
import androidx.compose.ui.input.pointer.PointerEventType
import androidx.compose.ui.semantics.SemanticsNode
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.semantics.getOrNull
import androidx.compose.ui.unit.Density
import com.acite.axlranko.data.DatasetRefreshHub
import com.acite.axlranko.data.DatasetSelection
import com.acite.axlranko.data.TrainerIpcClient
import com.acite.axlranko.model.AutomationSection
import com.acite.axlranko.model.AutomationSettingsDraft
import com.acite.axlranko.model.AutomationUiState
import com.acite.axlranko.model.UNIVERSAL_LORA_STRENGTH_MAX
import com.acite.axlranko.pages.components.automation.UniversalPane
import com.acite.axlranko.ui.theme.RankoTheme
import com.acite.axlranko.util.PathPicker
import java.awt.GraphicsEnvironment
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * The Universal (Beta) LoRA strength slider, dragged for real: a click on the track has to reach the
 * page's own model (`AutomationSettingsDraft.universalLoraStrength`), and the label above it has to
 * read the value the draft holds. Composing alone would not catch a slider whose callback goes
 * nowhere or whose range is not the 0–2 `api.py` enforces.
 */
class UniversalStrengthSliderTest {

    private class NoopPathPicker : PathPicker {
        override suspend fun pickDirectory(title: String, current: String): String? = null
        override suspend fun pickFile(title: String, current: String, extensions: List<String>?): String? = null
        override suspend fun saveFile(suggestedName: String, current: String): String? = null
        override fun deleteEmptyPlaceholder(path: String) = Unit
    }

    private fun viewModel() = AutomationScreenViewModel(
        ipc = TrainerIpcClient(),
        refreshHub = DatasetRefreshHub(),
        datasetSelection = DatasetSelection(),
        pathPicker = NoopPathPicker(),
    )

    private fun state(strength: Float) = AutomationUiState(
        section = AutomationSection.Universal,
        settings = AutomationSettingsDraft(
            server = "http://127.0.0.1:8188",
            universalLora = "Yui_s002850.safetensors",
            universalTrigger = "(yui_character:1.1)",
            universalLoraStrength = strength,
        ),
    )

    private fun scene(
        viewModel: AutomationScreenViewModel,
        state: AutomationUiState,
        width: Int = 1200,
    ) = ImageComposeScene(
        width = width,
        height = 2600,
        density = Density(1f),
    ) {
        RankoTheme { UniversalPane(state = state, viewModel = viewModel) }
    }

    private fun nodes(scene: ImageComposeScene): List<SemanticsNode> {
        scene.render()
        val found = mutableListOf<SemanticsNode>()
        fun walk(node: SemanticsNode) {
            found += node
            node.children.forEach { walk(it) }
        }
        scene.semanticsOwners.forEach { walk(it.rootSemanticsNode) }
        return found
    }

    private fun slider(nodes: List<SemanticsNode>): SemanticsNode? =
        nodes.firstOrNull { it.config.getOrNull(SemanticsProperties.ProgressBarRangeInfo) != null }

    private fun texts(nodes: List<SemanticsNode>): List<String> =
        nodes.flatMap { node -> node.config.getOrNull(SemanticsProperties.Text).orEmpty().map { it.text } }

    /** A press-release a quarter of the way along the track: 0.25 of a 0–2 range is 0.5. */
    private fun clickQuarter(scene: ImageComposeScene, node: SemanticsNode) {
        val x = node.positionInRoot.x + node.size.width * 0.25f
        val y = node.positionInRoot.y + node.size.height / 2f
        val down = PointerButtons(isPrimaryPressed = true)
        scene.sendPointerEvent(PointerEventType.Press, Offset(x, y), button = PointerButton.Primary, buttons = down)
        scene.sendPointerEvent(PointerEventType.Release, Offset(x, y), button = PointerButton.Primary, buttons = PointerButtons())
        scene.render()
    }

    @Test
    fun theSliderWritesTheDraftAndTheLabelReadsIt() {
        if (GraphicsEnvironment.isHeadless()) return
        val viewModel = viewModel()
        val first = scene(viewModel, state(1f))
        try {
            val label = "LoRA strength  1.00"
            assertTrue(texts(nodes(first)).contains(label), texts(nodes(first)).toString())
            val track = slider(nodes(first))
            assertNotNull(track, "the LoRA card carries no slider")
            val range = track.config.getOrNull(SemanticsProperties.ProgressBarRangeInfo)!!.range
            assertEquals(0f, range.start)
            assertEquals(UNIVERSAL_LORA_STRENGTH_MAX, range.endInclusive)
            clickQuarter(first, track)
        } finally {
            first.close()
        }
        val dragged = viewModel.uiState.value.settings.universalLoraStrength
        assertTrue(abs(dragged - 0.5f) < 0.15f, "a quarter along the track gave $dragged")

        // The pane labels whatever the draft holds, which is what a repaint after the drag shows.
        val second = scene(viewModel, state(dragged))
        try {
            val shown = texts(nodes(second)).filter { it.startsWith("LoRA strength") }
            assertTrue(shown.isNotEmpty(), texts(nodes(second)).toString())
            assertEquals("LoRA strength  0.50", shown.first())
        } finally {
            second.close()
        }
    }

    @Test
    fun theSliderStaysInsideANarrowPane() {
        if (GraphicsEnvironment.isHeadless()) return
        // The portrait width the Automation pages are laid out at: the slider has to be measured
        // against the card's own width rather than an unbounded one.
        val scene = scene(viewModel(), state(1f), width = 360)
        try {
            val track = slider(nodes(scene))
            assertNotNull(track, "the narrow LoRA card carries no slider")
            assertTrue(track.size.width in 100..360, "slider width ${track.size.width}")
        } finally {
            scene.close()
        }
    }

    @Test
    fun theSliderEndpointIsTheTopOfTheRange() {
        if (GraphicsEnvironment.isHeadless()) return
        val viewModel = viewModel()
        val scene = scene(viewModel, state(1f))
        try {
            val track = slider(nodes(scene))
            assertNotNull(track, "the LoRA card carries no slider")
            val x = track.positionInRoot.x + track.size.width * 0.99f
            val y = track.positionInRoot.y + track.size.height / 2f
            val down = PointerButtons(isPrimaryPressed = true)
            scene.sendPointerEvent(PointerEventType.Press, Offset(x, y), button = PointerButton.Primary, buttons = down)
            scene.sendPointerEvent(PointerEventType.Release, Offset(x, y), button = PointerButton.Primary, buttons = PointerButtons())
            scene.render()
        } finally {
            scene.close()
        }
        assertEquals(
            UNIVERSAL_LORA_STRENGTH_MAX,
            viewModel.uiState.value.settings.universalLoraStrength,
        )
    }
}
