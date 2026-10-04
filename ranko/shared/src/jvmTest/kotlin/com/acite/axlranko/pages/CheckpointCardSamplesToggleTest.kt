package com.acite.axlranko.pages

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.padding
import androidx.compose.ui.ImageComposeScene
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.input.pointer.PointerButton
import androidx.compose.ui.input.pointer.PointerButtons
import androidx.compose.ui.input.pointer.PointerEventType
import androidx.compose.ui.semantics.SemanticsNode
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.semantics.getOrNull
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.dp
import com.acite.axlranko.model.CheckpointItem
import com.acite.axlranko.model.SampleItem
import com.acite.axlranko.pages.components.CheckpointRow
import com.acite.axlranko.ui.theme.RankoTheme
import java.awt.GraphicsEnvironment
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * The sample area of a checkpoint card folds away and comes back: the card is composed with 24
 * images and the Hide/Show button is clicked the way a user would, so what is asserted is the
 * screen after the click rather than the state variable behind it. Skipped on a headless JVM.
 */
class CheckpointCardSamplesToggleTest {

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

    private fun click(scene: ImageComposeScene, x: Float, y: Float) {
        val down = PointerButtons(isPrimaryPressed = true)
        scene.sendPointerEvent(
            PointerEventType.Press,
            Offset(x, y),
            button = PointerButton.Primary,
            buttons = down,
        )
        scene.sendPointerEvent(
            PointerEventType.Release,
            Offset(x, y),
            button = PointerButton.Primary,
            buttons = PointerButtons(),
        )
        scene.render()
    }

    @Test
    fun theSampleGridFoldsAndUnfolds() {
        if (GraphicsEnvironment.isHeadless()) return
        val checkpoint = CheckpointItem(
            path = "/out/rein_20260911_120000/rein_s3050/rein.safetensors",
            runId = "rein_20260911_120000",
            dir = "rein_s3050",
            filename = "rein.safetensors",
            step = 3050,
            sizeBytes = 24_000_000,
            outputName = "rein",
        )
        val samples = (0 until 24).map { index ->
            SampleItem(
                filename = "rein_003050_p${index % 6}_${index / 6}.png",
                setIndex = index % 6,
                repeatIdx = index / 6,
                path = "/out/rein_20260911_120000/rein_samples/rein_003050_p${index % 6}_${index / 6}.png",
            )
        }
        val scene = ImageComposeScene(width = 900, height = 1500, density = Density(1f)) {
            RankoTheme {
                Column(Modifier.padding(12.dp)) {
                    CheckpointRowCard(
                        row = CheckpointRow(
                            checkpoint = checkpoint,
                            step = 3050,
                            samples = samples,
                            generated = emptyList(),
                            running = null,
                        ),
                        thumbSize = 60f,
                        showSetBadges = false,
                        newJobIds = emptySet(),
                        gpuFree = true,
                        starting = false,
                        busyElsewhere = false,
                        startingEvaluation = false,
                        pinning = false,
                        pinEnabled = true,
                        exportInFlightPath = null,
                        exportResult = null,
                        clearingSamples = false,
                        clearSamplesResult = null,
                        onOpen = {},
                        onGenerate = {},
                        onEvaluate = { _, _, _ -> },
                        onCancelEvaluation = {},
                        onOpenEvaluation = {},
                        onTogglePin = {},
                        onSaveAs = {},
                        onSendToAutomation = {},
                        onClearSamples = {},
                    )
                }
            }
        }
        try {
            val open = nodes(scene)
            assertTrue(nodeWithText(open, "24 images") != null, "the image count is shown")
            val hide = nodeWithText(open, "Hide")
            assertTrue(hide != null, "the card offers a Hide button")

            click(
                scene,
                hide.positionInRoot.x + hide.size.width / 2f,
                hide.positionInRoot.y + hide.size.height / 2f,
            )

            val folded = nodes(scene)
            assertTrue(nodeWithText(folded, "Show") != null, "the button flips to Show")
            assertTrue(nodeWithText(folded, "Hide") == null, "Hide is gone")
            assertTrue(nodeWithText(folded, "24 images") != null, "the count stays while folded")

            val show = nodeWithText(folded, "Show")!!
            click(
                scene,
                show.positionInRoot.x + show.size.width / 2f,
                show.positionInRoot.y + show.size.height / 2f,
            )
            assertTrue(nodeWithText(nodes(scene), "Hide") != null, "the grid comes back")
        } finally {
            scene.close()
        }
    }
}
