package com.acite.axlranko.pages.components

import androidx.compose.foundation.ScrollState
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.verticalScroll
import androidx.compose.runtime.Composable
import androidx.compose.ui.ExperimentalComposeUiApi
import androidx.compose.ui.ImageComposeScene
import androidx.compose.ui.InternalComposeUiApi
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.PointerEventType
import androidx.compose.ui.input.pointer.PointerId
import androidx.compose.ui.input.pointer.PointerType
import androidx.compose.ui.scene.ComposeScenePointer
import androidx.compose.ui.semantics.SemanticsNode
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.semantics.getOrNull
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.dp
import com.acite.axlranko.model.MetricPoint
import com.acite.axlranko.ui.theme.RankoTheme
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Finger gestures driven through the real pointer pipeline of a chart card, which is the only way
 * to see the touch rules hold: a swipe has to leave the chart alone and scroll the page, a long
 * press has to hand the chart the gesture, a pinch has to move one axis only, and the cursor
 * control has to park a cursor that a second tap turns into a pick.
 *
 * The scene is offscreen, so no window and no display are involved.
 */
@OptIn(ExperimentalComposeUiApi::class, InternalComposeUiApi::class)
class ChartTouchGestureTest {

    private val points = (0..400 step 10).map { MetricPoint(step = it, value = 1f - it / 800f) }

    /** The span the card opens on, so a pan or a zoom has somewhere to go. */
    private val stepSpan = 100f

    @Test
    fun aFingerSwipeScrollsThePageInsteadOfPanningTheChart() {
        val store = StepChartInteractionStore()
        val scroll = ScrollState(0)
        val scene = chartPage(store, scroll)
        try {
            // Starting on the chart itself and swiping up, the gesture a phone user makes to page
            // down the Dashboard.
            scene.send(PointerEventType.Press, finger(1, Offset(260f, 220f), pressed = true))
            repeat(4) { step ->
                scene.send(
                    PointerEventType.Move,
                    finger(1, Offset(260f, 220f - 30f * (step + 1)), pressed = true),
                )
            }
            scene.send(PointerEventType.Release, finger(1, Offset(260f, 100f), pressed = false))

            assertTrue(scroll.value > 0, "the page did not scroll under a finger swipe")
            assertFalse(store.state.detached, "a finger swipe panned the chart")
            assertEquals(null, store.state.xWindow)
        } finally {
            scene.close()
        }
    }

    @Test
    fun aLongPressThenADragPansTheChart() {
        val store = StepChartInteractionStore()
        val scroll = ScrollState(0)
        val scene = chartPage(store, scroll)
        try {
            scene.send(PointerEventType.Press, finger(1, Offset(200f, 120f), pressed = true))
            // The long-press window runs on real time; nothing may be consumed before it expires.
            Thread.sleep(700)
            repeat(4) { step ->
                scene.send(
                    PointerEventType.Move,
                    finger(1, Offset(200f + 25f * (step + 1), 120f), pressed = true),
                )
            }
            scene.send(PointerEventType.Release, finger(1, Offset(300f, 120f), pressed = false))

            assertTrue(store.state.detached, "a long press then a drag did not pan the chart")
            val window = assertNotNull(store.state.xWindow, "the pan left no x window")
            // The chart opened on its newest 100 steps (300..400) and was dragged to the right,
            // which walks the window back towards earlier steps.
            assertTrue(window.min < 300f, "the window did not move: $window")
            assertEquals(0, scroll.value, "the page scrolled while the chart was panning")
        } finally {
            scene.close()
        }
    }

    @Test
    fun aHorizontalPinchZoomsTheStepAxisAndLeavesYAlone() {
        val store = StepChartInteractionStore()
        val scene = chartPage(store, ScrollState(0))
        try {
            pinch(scene, start = Offset(230f, 140f), gap = 60f, spreadPerStep = 20f, horizontal = true)

            val window = assertNotNull(store.state.xWindow, "the pinch did not zoom x")
            assertTrue(window.range < stepSpan, "the step window did not narrow: $window")
            // The series carries its own 0..1 domain, so the chart draws in the normalized 0..100
            // space and the y axis the pinch never touched is exactly that.
            assertEquals(ChartWindow(0f, 100f), store.state.yWindows[CHART_ID])
        } finally {
            scene.close()
        }
    }

    @Test
    fun aVerticalPinchZoomsYAloneAndLeavesTheStepWindowOpen() {
        val store = StepChartInteractionStore()
        val scene = chartPage(store, ScrollState(0))
        try {
            pinch(scene, start = Offset(230f, 140f), gap = 60f, spreadPerStep = 20f, horizontal = false)

            val y = assertNotNull(store.state.yWindows[CHART_ID], "the pinch did not zoom y")
            assertTrue(y.range < 100f, "the y window did not narrow: $y")
            // x was never part of the gesture, so it is still the window the card opened on.
            val x = assertNotNull(store.state.xWindow)
            assertEquals(400f - stepSpan, x.min, 1e-3f)
            assertEquals(400f, x.max, 1e-3f)
        } finally {
            scene.close()
        }
    }

    @Test
    fun draggingTheCursorControlPlacesTheReadoutAndASecondTapPicks() {
        val store = StepChartInteractionStore()
        val picks = mutableListOf<Pair<Float, Offset>>()
        val scene = chartPage(store, ScrollState(0), picks)
        try {
            val home = scene.cursorControlCentre()
            // Drag the control out of the title row and down over the plot.
            val overPlot = home + Offset(-140f, 120f)
            scene.send(PointerEventType.Press, finger(1, home, pressed = true))
            scene.send(PointerEventType.Move, finger(1, overPlot, pressed = true))
            scene.send(PointerEventType.Release, finger(1, overPlot, pressed = false))

            val hovered = assertNotNull(store.state.hoverStep, "the dragged control placed no cursor")
            assertEquals(CHART_ID, store.state.hoverOwner)
            assertTrue(hovered in 0f..400f, "the cursor landed outside the data: $hovered")
            assertTrue(picks.isEmpty(), "dragging the control picked a checkpoint")

            // The control stayed where it was dropped, so the second tap is a tap on the cursor.
            val parked = scene.cursorControlCentre()
            scene.send(PointerEventType.Press, finger(1, parked, pressed = true))
            scene.send(PointerEventType.Release, finger(1, parked, pressed = false))

            assertEquals(1, picks.size, "the tap on a parked control did not pick: $picks")
            assertEquals(hovered, picks.first().first)
            assertTrue(
                picks.first().second.x > 0f && picks.first().second.y > 0f,
                "the anchor was not measured in root space",
            )
        } finally {
            scene.close()
        }
    }

    @Test
    fun draggingTheCursorControlBackToTheTitleRowClearsTheCursor() {
        val store = StepChartInteractionStore()
        val scene = chartPage(store, ScrollState(0))
        try {
            val home = scene.cursorControlCentre()
            /** One drag of the control, from where it is now to [target]. */
            fun dragTo(target: Offset) {
                val from = scene.cursorControlCentre()
                scene.send(PointerEventType.Press, finger(1, from, pressed = true))
                scene.send(PointerEventType.Move, finger(1, target, pressed = true))
                scene.send(PointerEventType.Release, finger(1, target, pressed = false))
            }

            dragTo(home + Offset(-140f, 120f))
            assertNotNull(store.state.hoverStep, "the control placed no cursor to begin with")

            dragTo(home)
            assertEquals(null, store.state.hoverStep, "the cursor outlived the control going home")
            assertEquals(null, store.state.hoverOwner)
            val back = scene.cursorControlCentre()
            assertTrue(
                (back - home).getDistance() < 2f,
                "the control did not return to its slot: $back vs $home",
            )
        } finally {
            scene.close()
        }
    }

    /** Spreads or closes two fingers around [start], the way a pinch on the plot looks. */
    private fun pinch(
        scene: ImageComposeScene,
        start: Offset,
        gap: Float,
        spreadPerStep: Float,
        horizontal: Boolean,
    ) {
        fun at(half: Float, pressed: Boolean) = if (horizontal) {
            listOf(
                finger(1, Offset(start.x - half, start.y), pressed),
                finger(2, Offset(start.x + half, start.y), pressed),
            )
        } else {
            listOf(
                finger(1, Offset(start.x, start.y - half), pressed),
                finger(2, Offset(start.x, start.y + half), pressed),
            )
        }
        scene.send(PointerEventType.Press, *at(gap / 2f, pressed = true).toTypedArray())
        var half = gap / 2f
        repeat(5) {
            half += spreadPerStep / 2f
            scene.send(PointerEventType.Move, *at(half, pressed = true).toTypedArray())
        }
        scene.send(PointerEventType.Release, *at(half, pressed = false).toTypedArray())
    }

    private fun finger(id: Long, position: Offset, pressed: Boolean) = ComposeScenePointer(
        id = PointerId(id),
        position = position,
        pressed = pressed,
        type = PointerType.Touch,
    )

    private fun ImageComposeScene.send(type: PointerEventType, vararg pointers: ComposeScenePointer) {
        sendPointerEvent(eventType = type, pointers = pointers.toList())
    }

    /** The Dashboard's shape: a chart inside a scrolling page, with room under it to scroll into. */
    private fun chartPage(
        store: StepChartInteractionStore,
        scroll: ScrollState,
        picks: MutableList<Pair<Float, Offset>> = mutableListOf(),
    ): ImageComposeScene = ImageComposeScene(width = 520, height = 320, density = Density(1f)) {
        RankoTheme {
            Column(modifier = Modifier.fillMaxSize().verticalScroll(scroll)) {
                chartCard(store, picks)
                Box(modifier = Modifier.fillMaxWidth().height(400.dp))
            }
        }
    }.also { it.render() }

    @Composable
    private fun chartCard(store: StepChartInteractionStore, picks: MutableList<Pair<Float, Offset>>) {
        MultiSeriesChartCard(
            title = "Train / Avg Loss",
            series = listOf(
                ChartSeries("Avg Loss", points, Color(0xFFE85D4C), domainMin = 0f, domainMax = 1f),
            ),
            smoothing = 0f,
            modifier = Modifier.fillMaxWidth(),
            chartHeight = 260.dp,
            defaultStepSpan = stepSpan,
            onPickStep = { step, anchor -> picks += step to anchor },
            showHoverStep = true,
            chartId = CHART_ID,
            interaction = store,
        )
    }

    /** Where the control is now, read from the semantics tree rather than guessed from the layout. */
    private fun ImageComposeScene.cursorControlCentre(): Offset {
        render()
        var found: SemanticsNode? = null
        fun walk(node: SemanticsNode) {
            if (
                node.config.getOrNull(SemanticsProperties.ContentDescription)
                    ?.any { it == CURSOR_CONTROL_DESCRIPTION } == true
            ) {
                found = node
            }
            node.children.forEach { walk(it) }
        }
        semanticsOwners.forEach { walk(it.unmergedRootSemanticsNode) }
        val node = assertNotNull(found, "the cursor control was not composed")
        return node.positionInRoot + Offset(node.size.width / 2f, node.size.height / 2f)
    }

    private companion object {
        const val CHART_ID = "test-avg-loss"
    }
}
