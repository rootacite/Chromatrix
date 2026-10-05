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
import androidx.compose.ui.state.ToggleableState
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.dp
import com.acite.axlranko.pages.components.automation.PromptEditorActions
import com.acite.axlranko.pages.components.automation.PromptPageEditor
import com.acite.axlranko.prompt.PromptLang
import com.acite.axlranko.prompt.PromptMode
import com.acite.axlranko.prompt.defaultSpec
import com.acite.axlranko.prompt.parseMatrix
import com.acite.axlranko.prompt.t
import com.acite.axlranko.ui.theme.RankoTheme
import java.awt.GraphicsEnvironment
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * The `invisible penis` switch on the wizard's stages page, composed and clicked for real: it sits
 * under the stage sliders, starts off, and hands the value it was flipped to to the editor's action.
 * The page's own text is asserted too — a mistyped string key falls back to the key itself, which is
 * what the title check catches.
 */
class PromptStagesToggleTest {

    private val matrix by lazy {
        parseMatrix(File(repoRoot(), "input_matrix.txt").readText(Charsets.UTF_8))
    }

    private fun repoRoot(): File {
        var dir: File? = File(System.getProperty("user.dir").orEmpty()).absoluteFile
        while (dir != null) {
            if (File(dir, "input_matrix.txt").isFile && File(dir, "trainer").isDirectory) return dir
            dir = dir.parentFile
        }
        throw AssertionError("could not find the repo root")
    }

    private fun actions(onInvisiblePenis: (Boolean) -> Unit) = PromptEditorActions(
        setCharacter = {}, setQualitySuffix = {}, setMode = {}, setExposure = { _, _ -> },
        setClothingAny = {}, toggleClothing = {}, setChest = {}, setBelly = {}, setFigure = {},
        setPussyShape = {}, setPussyHair = {}, setFaceGroup = { _, _ -> }, toggleFaceTag = { _, _ -> },
        setSceneAny = {}, toggleScene = {}, setSceneGroup = { _, _ -> }, setFamilyAny = {},
        toggleFamily = {}, setVaginalRatio = {}, setStageWeight = { _, _ -> },
        setInvisiblePenis = onInvisiblePenis, setPoseAny = {}, togglePose = {}, setCount = {},
        setSeedText = {},
    )

    private fun scene(lang: PromptLang, actions: PromptEditorActions): ImageComposeScene {
        val spec = defaultSpec().also { it.mode = PromptMode.Sex }
        return ImageComposeScene(width = 520, height = 1000, density = Density(1f)) {
            RankoTheme {
                Column(Modifier.padding(14.dp)) {
                    PromptPageEditor(
                        pageKey = "stages",
                        spec = spec,
                        matrix = matrix,
                        lang = lang,
                        seedText = "",
                        actions = actions,
                    )
                }
            }
        }
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

    private fun switch(nodes: List<SemanticsNode>): SemanticsNode? =
        nodes.firstOrNull { it.config.getOrNull(SemanticsProperties.ToggleableState) != null }

    private fun texts(nodes: List<SemanticsNode>): List<String> =
        nodes.flatMap { node ->
            node.config.getOrNull(SemanticsProperties.Text).orEmpty().map { it.text }
        }

    private fun click(scene: ImageComposeScene, node: SemanticsNode) {
        val x = node.positionInRoot.x + node.size.width / 2f
        val y = node.positionInRoot.y + node.size.height / 2f
        val down = PointerButtons(isPrimaryPressed = true)
        scene.sendPointerEvent(PointerEventType.Press, Offset(x, y), button = PointerButton.Primary, buttons = down)
        scene.sendPointerEvent(PointerEventType.Release, Offset(x, y), button = PointerButton.Primary, buttons = PointerButtons())
        scene.render()
    }

    private fun assertSwitchAndTitle(lang: PromptLang) {
        val flipped = mutableListOf<Boolean>()
        val scene = scene(lang, actions { flipped += it })
        try {
            val toggle = switch(nodes(scene))
            assertNotNull(toggle, "the stages page carries no switch")
            assertEquals(ToggleableState.Off, toggle.config.getOrNull(SemanticsProperties.ToggleableState))
            val title = t(lang, "invisible_penis_title")
            assertTrue(texts(nodes(scene)).any { it == title }, texts(nodes(scene)).toString())
            click(scene, toggle)
            assertEquals(listOf(true), flipped)
        } finally {
            scene.close()
        }
    }

    @Test
    fun theStagesPageCarriesTheSwitchInBothLanguages() {
        if (GraphicsEnvironment.isHeadless()) return
        assertSwitchAndTitle(PromptLang.English)
        assertSwitchAndTitle(PromptLang.Chinese)
    }
}
