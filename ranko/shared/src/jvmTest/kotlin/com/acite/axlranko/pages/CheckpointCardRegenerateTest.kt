package com.acite.axlranko.pages

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.padding
import androidx.compose.ui.ImageComposeScene
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.toComposeImageBitmap
import androidx.compose.ui.input.pointer.PointerButton
import androidx.compose.ui.input.pointer.PointerButtons
import androidx.compose.ui.input.pointer.PointerEventType
import androidx.compose.ui.semantics.SemanticsNode
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.semantics.getOrNull
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.dp
import com.acite.axlranko.model.CheckpointItem
import com.acite.axlranko.model.GeneratedSampleJob
import com.acite.axlranko.model.SampleItem
import com.acite.axlranko.pages.components.CheckpointRow
import com.acite.axlranko.pages.components.JOB_DONE
import com.acite.axlranko.ui.theme.RankoTheme
import java.awt.GraphicsEnvironment
import kotlin.math.abs
import kotlin.math.roundToInt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * The ↻ button on a sample thumbnail: it exists only where there is a prompt set to redraw from,
 * it calls back with the image it sits on, and a busy GPU leaves it visible but inert. Composed
 * and clicked for real, so what is asserted is the screen, not the state behind it.
 */
class CheckpointCardRegenerateTest {

    private val checkpoint = CheckpointItem(
        path = "/out/rein_20260911_120000/rein_s3050/rein.safetensors",
        runId = "rein_20260911_120000",
        dir = "rein_s3050",
        filename = "rein.safetensors",
        step = 3050,
        sizeBytes = 24_000_000,
        outputName = "rein",
    )

    private fun sample(set: Int, repeat: Int) = SampleItem(
        filename = "rein_003050_p${set}_$repeat.png",
        setIndex = set,
        repeatIdx = repeat,
        path = "/out/rein_20260911_120000/rein_samples/rein_003050_p${set}_$repeat.png",
    )

    /** A manual single generation: `set_index` -1, so it belongs to no prompt set. */
    private fun manualJob() = GeneratedSampleJob(
        id = "rein_s003050_manual_gen_1",
        state = JOB_DONE,
        mode = "single",
        step = 3050,
        checkpoint = checkpoint.path,
        imagePath = "/out/rein_20260911_120000/rein_samples/generated/rein_s003050_manual_gen_1.png",
    )

    private fun walk(node: SemanticsNode, out: MutableList<SemanticsNode>) {
        out += node
        node.children.forEach { walk(it, out) }
    }

    /**
     * The merged tree: the ↻ IconButton is a clickable of its own, so it survives its card's
     * clickable ancestor as one node and carries that node's disabled state.
     */
    private fun nodes(scene: ImageComposeScene): List<SemanticsNode> {
        scene.render()
        val found = mutableListOf<SemanticsNode>()
        scene.semanticsOwners.forEach { owner -> walk(owner.rootSemanticsNode, found) }
        return found
    }

    private fun buttons(nodes: List<SemanticsNode>): List<SemanticsNode> =
        nodes.filter { node ->
            node.config.getOrNull(SemanticsProperties.ContentDescription)
                .orEmpty()
                .any { it == "Regenerate sample" }
        }

    /** Every pixel of the node's box (plus [margin] around it) that is the brand amber (#F8A818). */
    private fun amberPoints(
        scene: ImageComposeScene,
        node: SemanticsNode,
        margin: Int,
    ): List<Pair<Int, Int>> {
        val bitmap = scene.render().toComposeImageBitmap()
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.readPixels(pixels)
        val left = node.positionInRoot.x.roundToInt()
        val top = node.positionInRoot.y.roundToInt()
        val found = mutableListOf<Pair<Int, Int>>()
        for (y in (top - margin).coerceAtLeast(0)..(top + node.size.height - 1 + margin).coerceAtMost(bitmap.height - 1)) {
            for (x in (left - margin).coerceAtLeast(0)..(left + node.size.width - 1 + margin).coerceAtMost(bitmap.width - 1)) {
                val argb = pixels[y * bitmap.width + x]
                val r = (argb shr 16) and 0xFF
                val g = (argb shr 8) and 0xFF
                val b = argb and 0xFF
                if (abs(r - 0xF8) <= 8 && abs(g - 0xA8) <= 8 && abs(b - 0x18) <= 8) found += x to y
            }
        }
        return found
    }

    private fun click(scene: ImageComposeScene, node: SemanticsNode) {
        val x = node.positionInRoot.x + node.size.width / 2f
        val y = node.positionInRoot.y + node.size.height / 2f
        val down = PointerButtons(isPrimaryPressed = true)
        scene.sendPointerEvent(PointerEventType.Press, Offset(x, y), button = PointerButton.Primary, buttons = down)
        scene.sendPointerEvent(PointerEventType.Release, Offset(x, y), button = PointerButton.Primary, buttons = PointerButtons())
        scene.render()
    }

    private fun scene(
        samples: List<SampleItem>,
        generated: List<GeneratedSampleJob>,
        gpuFree: Boolean = true,
        onRegenerate: (SampleItem) -> Unit = {},
    ) = ImageComposeScene(width = 900, height = 1200, density = Density(1f)) {
        RankoTheme {
            Column(Modifier.padding(12.dp)) {
                CheckpointRowCard(
                    row = CheckpointRow(
                        checkpoint = checkpoint,
                        step = 3050,
                        samples = samples,
                        generated = generated,
                        running = null,
                    ),
                    thumbSize = 60f,
                    showSetBadges = false,
                    newJobIds = emptySet(),
                    gpuFree = gpuFree,
                    starting = false,
                    busyElsewhere = false,
                    startingEvaluation = false,
                    pinning = false,
                    pinEnabled = true,
                    exportInFlightPath = null,
                    exportResult = null,
                    clearingSamples = false,
                    clearSamplesResult = null,
                    onRegenerate = { item, _ -> onRegenerate(item) },
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

    @Test
    fun theButtonIsASmallAmberDotInThePictureCorner() {
        if (GraphicsEnvironment.isHeadless()) return
        val scene = scene(samples = listOf(sample(0, 0)), generated = emptyList())
        try {
            val button = buttons(nodes(scene)).singleOrNull()
            assertNotNull(button, "the ↻ button is missing")
            // Three times the 7 dp dot this started as. The scene is density 1, so its Int pixel
            // size is the dp size.
            assertTrue(
                button.size.width == 21 && button.size.height == 21,
                "the disc is ${button.size.width}×${button.size.height} px, not 21×21",
            )
            // The disc is what it looks like: the amber it paints stays inside its own box (a
            // Material3 `IconButton` grew the painted circle far past the clickable node), and it
            // is the brand amber (#F8A818) read off the composed surface.
            val painted = amberPoints(scene, button, margin = 20)
            assertTrue(painted.isNotEmpty(), "the disc does not paint the brand amber")
            val left = button.positionInRoot.x.roundToInt()
            val top = button.positionInRoot.y.roundToInt()
            val outside = painted.filter { (x, y) ->
                x < left - 1 || x > left + button.size.width || y < top - 1 || y > top + button.size.height
            }
            assertTrue(outside.isEmpty(), "the disc paints outside its own box at ${outside.take(8)}")
        } finally {
            scene.close()
        }
    }

    @Test
    fun aSetImageCarriesTheButtonAndAManualOneDoesNot() {
        if (GraphicsEnvironment.isHeadless()) return
        var clicked: SampleItem? = null
        val training = sample(0, 0)
        val scene = scene(
            samples = listOf(training),
            generated = listOf(manualJob()),
            onRegenerate = { clicked = it },
        )
        try {
            val found = buttons(nodes(scene))
            assertEquals(1, found.size, "one ↻ for the set image, none for the manual one")
            assertNull(clicked, "nothing is clicked yet")

            click(scene, found.single())
            assertEquals(training.path, clicked?.path, "the button reports the image it sits on")
        } finally {
            scene.close()
        }
    }

    @Test
    fun aBusyGpuLeavesTheButtonVisibleAndInert() {
        if (GraphicsEnvironment.isHeadless()) return
        var clicked: SampleItem? = null
        val scene = scene(
            samples = listOf(sample(0, 0)),
            generated = emptyList(),
            gpuFree = false,
            onRegenerate = { clicked = it },
        )
        try {
            val found = buttons(nodes(scene)).singleOrNull()
            assertNotNull(found, "the button stays on screen while the GPU is busy")
            assertTrue(
                found.config.getOrNull(SemanticsProperties.Disabled) != null,
                "a busy GPU disables the button",
            )
            click(scene, found)
            assertNull(clicked, "a disabled button starts nothing")
        } finally {
            scene.close()
        }
    }

    @Test
    fun aSamplesOnlyRowHasNoButton() {
        if (GraphicsEnvironment.isHeadless()) return
        // The weights are gone (a Reset), so there is no checkpoint to redraw from.
        val scene = ImageComposeScene(width = 900, height = 900, density = Density(1f)) {
            RankoTheme {
                Column(Modifier.padding(12.dp)) {
                    CheckpointRowCard(
                        row = CheckpointRow(
                            checkpoint = null,
                            step = 3050,
                            samples = listOf(sample(0, 0)),
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
                        onRegenerate = { _, _ -> },
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
            assertTrue(buttons(nodes(scene)).isEmpty(), "no checkpoint, no redraw")
        } finally {
            scene.close()
        }
    }
}
