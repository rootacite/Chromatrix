package com.acite.axlranko.pages.components

import androidx.compose.ui.geometry.Offset
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * The decisions the touch gesture loops take, without a window: which axis a pinch commits to, when
 * a finger is allowed to pan at all, and where an anchored zoom leaves the viewport.
 */
class ChartTouchTest {

    @Test
    fun aPinchTakesTheAxisTheFingersSpreadAlong() {
        assertEquals(PinchAxis.X, pinchAxis(Offset(20f, 4f)))
        assertEquals(PinchAxis.Y, pinchAxis(Offset(4f, 20f)))
        assertEquals(PinchAxis.Y, pinchAxis(Offset(-3f, -30f)))
        // A tie — and the degenerate case of two fingers landing on the same spot — goes to x, the
        // axis a step chart is read along.
        assertEquals(PinchAxis.X, pinchAxis(Offset(5f, -5f)))
        assertEquals(PinchAxis.X, pinchAxis(Offset.Zero))
    }

    @Test
    fun aFingerThatMovesBeforeTheLongPressAborts() {
        // Nothing down any more.
        assertEquals(TouchDecision.Abort, touchDecision(pressedCount = 0, anyConsumed = false, travel = 0f, touchSlop = 18f))
        // The page took the movement, or the finger left the touch slop: both mean the chart has to
        // keep its hands off.
        assertEquals(TouchDecision.Abort, touchDecision(pressedCount = 1, anyConsumed = true, travel = 2f, touchSlop = 18f))
        assertEquals(TouchDecision.Abort, touchDecision(pressedCount = 1, anyConsumed = false, travel = 30f, touchSlop = 18f))
        // A finger that is down and still is the only thing the window waits for.
        assertEquals(TouchDecision.Wait, touchDecision(pressedCount = 1, anyConsumed = false, travel = 3f, touchSlop = 18f))
        // A second finger is never part of a page scroll, so it wins even after movement.
        assertEquals(TouchDecision.Pinch, touchDecision(pressedCount = 2, anyConsumed = false, travel = 40f, touchSlop = 18f))
    }

    @Test
    fun anAnchoredZoomKeepsThePointUnderTheAnchorStill() {
        val before = ChartViewport(xMin = 0f, xMax = 100f, yMin = 0f, yMax = 10f)

        val zoomed = zoomedViewport(before, newXRange = 50f, newYRange = 5f, xFrac = 0.25f, yFrac = 0.5f)

        val anchorX = before.xMin + 0.25f * before.xRange
        val anchorY = before.yMax - 0.5f * before.yRange
        assertEquals(anchorX, zoomed.xMin + 0.25f * zoomed.xRange, 1e-3f, "x moved under the anchor: $zoomed")
        assertEquals(anchorY, zoomed.yMax - 0.5f * zoomed.yRange, 1e-3f, "y moved under the anchor: $zoomed")
        assertEquals(50f, zoomed.xRange, 1e-3f)
        assertEquals(5f, zoomed.yRange, 1e-3f)
    }

    @Test
    fun aZoomFactorStaysInsideTheChartAndItsOwnLimits() {
        // The whole range is the ceiling: a pinch in cannot show less than the data has, and the
        // minimum keeps the last steps apart.
        assertEquals(100f, zoomRange(current = 100f, factor = 4f, minRange = 20f, fullRange = 100f))
        assertEquals(20f, zoomRange(current = 100f, factor = 0.001f, minRange = 20f, fullRange = 100f))
        assertEquals(50f, zoomRange(current = 100f, factor = 0.5f, minRange = 20f, fullRange = 100f))
        // A range with no span to zoom into stays where it is instead of dividing by zero.
        assertEquals(0f, zoomRange(current = 10f, factor = 0.5f, minRange = 1f, fullRange = 0f))
    }

    @Test
    fun theCursorProbeAnswersOnlyOnceTheCanvasHasPublishedItself() {
        val probe = ChartCursorProbe()
        assertNull(probe.stepAt(Offset(10f, 10f)), "an unbound probe must not invent a step")

        probe.stepAtRoot = { root -> if (root.x > 50f) 120f else null }
        assertEquals(120f, probe.stepAt(Offset(60f, 10f)))
        assertNull(probe.stepAt(Offset(40f, 10f)))
    }
}
