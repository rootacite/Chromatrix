package com.acite.axlranko.pages.components

import androidx.compose.foundation.gestures.calculateCentroid
import androidx.compose.foundation.gestures.calculateZoom
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.input.pointer.AwaitPointerEventScope
import androidx.compose.ui.input.pointer.PointerId
import androidx.compose.ui.input.pointer.PointerInputChange
import androidx.compose.ui.input.pointer.changedToUpIgnoreConsumed
import androidx.compose.ui.input.pointer.positionChange
import androidx.compose.ui.input.pointer.positionChangeIgnoreConsumed
import androidx.compose.ui.unit.dp
import kotlin.math.abs
import kotlinx.coroutines.withTimeoutOrNull

/**
 * The touch half of the chart gestures, kept out of `ChartCard.kt` so the decisions its loops take
 * can be tested without a window.
 *
 * The rules the loops implement:
 *
 * - A mouse keeps the desktop's behaviour: the chart pans as soon as the pointer moves.
 * - A finger never pans on the first move. Until it has been held still for the long-press
 *   timeout, every event is watched and then left unconsumed ([TouchDecision.Abort]), which is what
 *   lets the `LazyColumn` holding the Dashboard scroll under a swipe.
 * - Two fingers pinch, locked to one axis for the whole gesture. A step chart's x axis counts steps
 *   and its y axis counts values, so letting one pinch edit both leaves a viewport nobody asked
 *   for.
 */

/** Which axis a pinch may change; the other one is untouched for the whole gesture. */
internal enum class PinchAxis { X, Y }

/**
 * The axis a pinch is locked to, taken from the direction the two fingers spread in. A tie goes to
 * X, the axis a step chart is read along.
 */
internal fun pinchAxis(spread: Offset): PinchAxis =
    if (abs(spread.x) >= abs(spread.y)) PinchAxis.X else PinchAxis.Y

/** What one event inside the touch decision window means. */
internal enum class TouchDecision { Wait, Abort, Pan, Pinch }

/**
 * The decision the long-press window takes for one touch event, before anything is consumed.
 *
 * [Abort] is what the page scroll depends on: nothing was consumed, so the `LazyColumn` above the
 * chart receives the same movement and scrolls. A second finger down is never part of a page
 * scroll, so it outranks the movement checks.
 */
internal fun touchDecision(
    pressedCount: Int,
    anyConsumed: Boolean,
    travel: Float,
    touchSlop: Float,
): TouchDecision = when {
    pressedCount == 0 -> TouchDecision.Abort
    anyConsumed -> TouchDecision.Abort
    pressedCount >= 2 -> TouchDecision.Pinch
    travel > touchSlop -> TouchDecision.Abort
    else -> TouchDecision.Wait
}

/** How far the fingers must spread before the pinch commits to an axis. */
private const val PINCH_AXIS_LOCK_DP = 4f

/**
 * Waits out the long-press window for the finger that went down, watching every event for a second
 * finger ([TouchDecision.Pinch]) or for movement that belongs to the page ([TouchDecision.Abort]).
 *
 * The window is bounded once rather than restarted per event, so a jittery finger cannot hold it
 * open. Running out of time — the `null` of [withTimeoutOrNull] — is the long press itself: the
 * finger is still down and still, which is exactly when the chart may take the gesture.
 */
internal suspend fun AwaitPointerEventScope.awaitTouchDecision(
    down: PointerInputChange,
): TouchDecision {
    val slop = viewConfiguration.touchSlop
    return withTimeoutOrNull(viewConfiguration.longPressTimeoutMillis) {
        var travel = 0f
        while (true) {
            val event = awaitPointerEvent()
            event.changes.firstOrNull { it.id == down.id }?.let {
                travel += it.positionChangeIgnoreConsumed().getDistance()
            }
            when (
                touchDecision(
                    pressedCount = event.changes.count { it.pressed },
                    anyConsumed = event.changes.any { it.isConsumed },
                    travel = travel,
                    touchSlop = slop,
                )
            ) {
                // `Pan` cannot come back from the decision function — inside this window the
                // long press has not fired yet, so there is nothing to pan with.
                TouchDecision.Wait, TouchDecision.Pan -> Unit
                TouchDecision.Pinch -> return@withTimeoutOrNull TouchDecision.Pinch
                TouchDecision.Abort -> return@withTimeoutOrNull TouchDecision.Abort
            }
        }
        @Suppress("UNREACHABLE_CODE")
        TouchDecision.Abort
    } ?: TouchDecision.Pan
}

/**
 * Waits until [pointerId] has moved past [slop], consuming the change that crosses it. False when
 * the pointer was raised — or when another gesture consumed the movement, which is how a parent
 * scrollable wins.
 *
 * This is the slop wait `detectDragGestures` does for a mouse, which is why the mouse's own slop
 * (a fraction of a dp) is passed in rather than the touch one: a desktop drag has to start on the
 * first pixel, the way it always has.
 */
internal suspend fun AwaitPointerEventScope.awaitDragSlop(pointerId: PointerId, slop: Float): Boolean {
    var pointer = pointerId
    var travel = 0f
    while (true) {
        val event = awaitPointerEvent()
        val change = event.changes.firstOrNull { it.id == pointer } ?: return false
        if (change.isConsumed) return false
        if (change.changedToUpIgnoreConsumed()) {
            pointer = event.changes.firstOrNull { it.pressed }?.id ?: return false
            continue
        }
        travel += change.positionChangeIgnoreConsumed().getDistance()
        if (travel > slop) {
            change.consume()
            return true
        }
    }
}

/** Pans on every change of [pointerId] until that finger is lifted. */
internal suspend fun AwaitPointerEventScope.dragPan(
    pointerId: PointerId,
    onPan: (Offset) -> Unit,
) {
    while (true) {
        val event = awaitPointerEvent()
        val change = event.changes.firstOrNull { it.id == pointerId } ?: return
        if (!change.pressed) return
        val delta = change.positionChange()
        if (delta == Offset.Zero) continue
        change.consume()
        onPan(delta)
    }
}

/**
 * Zooms through [onZoom] while two fingers are down, locked to the axis the fingers first spread
 * along. [onZoom] is handed a multiplier on the visible range, the same convention the mouse wheel
 * uses: below 1 zooms in, so spreading the fingers narrows the window.
 *
 * The pinch owns the gesture from the second finger on — the page must not scroll under it — and
 * ends when a finger is lifted.
 */
internal suspend fun AwaitPointerEventScope.pinchZoom(
    onZoom: (axis: PinchAxis, factor: Float, centroid: Offset) -> Unit,
) {
    val lockSlop = PINCH_AXIS_LOCK_DP.dp.toPx()
    var axis: PinchAxis? = null
    var lastSeparation: Offset? = null
    while (true) {
        val event = awaitPointerEvent()
        val pressed = event.changes.filter { it.pressed }.sortedBy { it.id.value }
        if (pressed.size < 2) return
        val separation = pressed[0].position - pressed[1].position
        val previous = lastSeparation
        lastSeparation = separation
        event.changes.forEach { it.consume() }
        if (previous == null) continue
        val locked = axis ?: run {
            val spread = separation - previous
            if (spread.getDistance() < lockSlop) continue
            pinchAxis(spread).also { axis = it }
        }
        val magnification = event.calculateZoom()
        if (!magnification.isFinite() || magnification <= 0f || magnification == 1f) continue
        onZoom(locked, 1f / magnification, event.calculateCentroid())
    }
}

/**
 * The viewport after zooming to [newXRange]/[newYRange] with the point under ([xFrac], [yFrac]) —
 * fractions of the plot rect — held where it is. The mouse wheel and the pinch both go through
 * this, so an anchored zoom means the same thing whichever one produced it.
 */
internal fun zoomedViewport(
    vp: ChartViewport,
    newXRange: Float,
    newYRange: Float,
    xFrac: Float,
    yFrac: Float,
): ChartViewport {
    val dataX = vp.xMin + xFrac * vp.xRange
    val dataY = vp.yMax - yFrac * vp.yRange
    val newXMin = dataX - xFrac * newXRange
    val newYMin = dataY - (1f - yFrac) * newYRange
    return ChartViewport(newXMin, newXMin + newXRange, newYMin, newYMin + newYRange)
}

/** One zoom step's range, clamped to the chart's own limits and to the data behind it. */
internal fun zoomRange(current: Float, factor: Float, minRange: Float, fullRange: Float): Float {
    if (fullRange <= 1e-9f) return 0f
    return (current * factor).coerceIn(minOf(minRange, fullRange), fullRange)
}

/**
 * The chart canvas's own hit testing, published for the card's cursor control. The control lives in
 * the card's title row, and only the canvas knows where the plot is and which logged step a
 * position over it stands for.
 */
internal class ChartCursorProbe {
    /** A window-root position over the plot → the nearest logged step; null outside the plot. */
    var stepAtRoot: ((Offset) -> Float?)? = null

    fun stepAt(root: Offset): Float? = stepAtRoot?.invoke(root)
}
