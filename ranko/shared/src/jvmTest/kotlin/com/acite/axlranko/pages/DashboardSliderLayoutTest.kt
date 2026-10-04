package com.acite.axlranko.pages

import androidx.compose.ui.ImageComposeScene
import androidx.compose.ui.semantics.SemanticsNode
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.semantics.getOrNull
import androidx.compose.ui.unit.Density
import com.acite.axlranko.data.TrainerIpcClient
import com.acite.axlranko.ui.theme.RankoTheme
import com.acite.axlranko.util.PathPicker
import java.awt.GraphicsEnvironment
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * The Dashboard's three moved sliders are measured on the page the user sees: Curve Smoothing and
 * Chart Line inside the Training Charts block (below its section header), Sample Size inside the
 * Checkpoints block, and none of the three in the page header any more. A very tall scene keeps
 * every LazyColumn item composed at once. Skipped on a headless JVM.
 */
class DashboardSliderLayoutTest {

    private class NoopPathPicker : PathPicker {
        override suspend fun pickDirectory(title: String, current: String): String? = null
        override suspend fun pickFile(title: String, current: String, extensions: List<String>?): String? = null
        override suspend fun saveFile(suggestedName: String, current: String): String? = null
        override fun deleteEmptyPlaceholder(path: String) = Unit
    }

    private fun walk(node: SemanticsNode, out: MutableList<SemanticsNode>) {
        out += node
        node.children.forEach { walk(it, out) }
    }

    private fun texts(scene: ImageComposeScene): List<Pair<String, Float>> {
        scene.render()
        val found = mutableListOf<SemanticsNode>()
        scene.semanticsOwners.forEach { owner -> walk(owner.rootSemanticsNode, found) }
        return found.flatMap { node ->
            val y = node.positionInRoot.y
            node.config.getOrNull(SemanticsProperties.Text).orEmpty().map { it.text to y }
        }
    }

    @Test
    fun slidersSitInTheirOwnSections() {
        if (GraphicsEnvironment.isHeadless()) return
        val viewModel = DashboardScreenViewModel(TrainerIpcClient(), NoopPathPicker())
        val scene = ImageComposeScene(width = 1600, height = 6000, density = Density(1f)) {
            RankoTheme { DashboardScreen(viewModel = viewModel) }
        }
        try {
            val lines = texts(scene)
            fun y(label: String): Float? = lines.firstOrNull { it.first.startsWith(label) }?.second
            val trainingControl = y("Training Control")
            val trainingCharts = y("Training Charts")
            val samplingPrompts = y("Sampling Prompts")
            val checkpoints = y("Checkpoints")
            val curve = y("Curve Smoothing:")
            val chartLine = y("Chart Line:")
            val sampleSize = y("Sample Size:")

            assertTrue(trainingControl != null, "the page did not compose: $lines")
            assertTrue(trainingCharts != null && checkpoints != null, "section headers missing")
            assertTrue(curve != null && chartLine != null && sampleSize != null, "moved sliders missing")

            // None of the three is left in the header, which is the first block on the page.
            val headerEnd = trainingControl
            assertTrue(curve > headerEnd + 8f, "Curve Smoothing still in the header (y=$curve)")
            assertTrue(chartLine > headerEnd + 8f, "Chart Line still in the header (y=$chartLine)")
            assertTrue(sampleSize > headerEnd + 8f, "Sample Size still in the header (y=$sampleSize)")

            // Curve Smoothing and Chart Line belong to Training Charts, before Sampling Prompts.
            assertTrue(curve > trainingCharts, "Curve Smoothing above its section (y=$curve)")
            assertTrue(chartLine > trainingCharts, "Chart Line above its section (y=$chartLine)")
            assertTrue(
                samplingPrompts == null || curve < samplingPrompts,
                "Curve Smoothing past the next section (y=$curve)",
            )

            // Sample Size belongs to Checkpoints.
            assertTrue(sampleSize > checkpoints, "Sample Size above Checkpoints (y=$sampleSize)")
        } finally {
            scene.close()
        }
    }
}
