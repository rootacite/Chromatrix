package com.acite.axlranko.pages

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.width
import androidx.compose.ui.ImageComposeScene
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.SemanticsActions
import androidx.compose.ui.semantics.SemanticsNode
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.semantics.getOrNull
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.dp
import com.acite.axlranko.ui.theme.RankoTheme
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * The Quick page's character trigger control, composed for real offscreen. What the page hands it —
 * the tag the validation sets agree on, or null when they do not — decides whether it can be typed
 * into, and the supporting line says which of the two it is. No window, so no display either.
 */
class CharacterTriggerFieldRenderTest {

    @Test
    fun theFieldShowsTheSharedTagAndReportsEdits() {
        var edited: String? = null
        val scene = triggerScene(sharedTag = "alice") { edited = it }
        try {
            val field = scene.triggerField()
            assertEquals("alice", field.config.getOrNull(SemanticsProperties.EditableText)?.text)
            assertTrue(field.config.getOrNull(SemanticsProperties.Disabled) == null, "the field is off")

            val setText = assertNotNull(
                field.config.getOrNull(SemanticsActions.SetText)?.action,
                "the field cannot be typed into",
            )
            setText(AnnotatedString("bob"))
            assertEquals("bob", edited)

            assertTrue(
                scene.texts().any { it.contains("first tag of every Validation sample set") },
                "the supporting line does not say what the field writes: ${scene.texts()}",
            )
        } finally {
            scene.close()
        }
    }

    @Test
    fun theFieldIsOffWhileTheSetsDisagree() {
        val scene = triggerScene(sharedTag = null) { }
        try {
            val field = scene.triggerField()
            assertNotNull(field.config.getOrNull(SemanticsProperties.Disabled), "the field is editable")
            assertTrue(
                scene.texts().any { it.contains("different tags") },
                "the reason the field is off is not on screen: ${scene.texts()}",
            )
        } finally {
            scene.close()
        }
    }

    private fun triggerScene(sharedTag: String?, onChange: (String) -> Unit): ImageComposeScene =
        ImageComposeScene(width = 420, height = 160, density = Density(1f)) {
            RankoTheme {
                Column(modifier = Modifier.width(380.dp)) {
                    CharacterTriggerField(sharedTag = sharedTag, onChange = onChange)
                }
            }
        }.also { it.render() }

    private fun ImageComposeScene.nodes(): List<SemanticsNode> {
        render()
        val found = mutableListOf<SemanticsNode>()
        fun walk(node: SemanticsNode) {
            found += node
            node.children.forEach { walk(it) }
        }
        semanticsOwners.forEach { walk(it.unmergedRootSemanticsNode) }
        return found
    }

    private fun ImageComposeScene.triggerField(): SemanticsNode = nodes().first { node ->
        node.config.getOrNull(SemanticsProperties.EditableText) != null ||
            node.config.getOrNull(SemanticsProperties.Disabled) != null
    }

    private fun ImageComposeScene.texts(): List<String> = nodes().flatMap { node ->
        node.config.getOrNull(SemanticsProperties.Text).orEmpty().map { it.text }
    }
}
