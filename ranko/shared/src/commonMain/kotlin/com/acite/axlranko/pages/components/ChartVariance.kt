package com.acite.axlranko.pages.components

import com.acite.axlranko.model.MetricPoint

/**
 * Rolling *population* variance of the raw `Val/Fixed_Loss` series, for the Learning Rate chart's
 * stability curve.
 *
 * For each point `x` the window is the closed interval `[x - span, x + span]` with
 * `span = steps_per_epoch / 4`, and the value is `Σ(v - mean)² / n` over the raw values inside it —
 * the population form, so a window of two does not report an inflated variance. A step with fewer
 * than two points in its window produces no point at all, and no `steps_per_epoch` means the whole
 * curve is absent: the caller then draws the two learning-rate curves alone.
 *
 * The variance is read off the raw series; the chart's own smoothing slider is applied to this
 * result like it is to any other series.
 */
internal fun rollingPopulationVariance(
    points: List<MetricPoint>,
    stepsPerEpoch: Int?,
): List<MetricPoint> {
    val perEpoch = stepsPerEpoch ?: return emptyList()
    if (perEpoch <= 0) return emptyList()
    val finite = points.filter { it.value.isFinite() }.sortedBy { it.step }
    if (finite.isEmpty()) return emptyList()
    val halfSpan = perEpoch / 4.0
    val out = ArrayList<MetricPoint>(finite.size)
    for (center in finite) {
        val low = center.step - halfSpan
        val high = center.step + halfSpan
        // Both ends are in: the window is closed on the step itself, and every step counts once.
        var count = 0
        var sum = 0.0
        var sumSquares = 0.0
        for (point in finite) {
            if (point.step < low) continue
            if (point.step > high) break
            val value = point.value.toDouble()
            count += 1
            sum += value
            sumSquares += value * value
        }
        if (count < 2) continue
        val mean = sum / count
        val variance = (sumSquares / count) - mean * mean
        out += MetricPoint(step = center.step, value = variance.coerceAtLeast(0.0).toFloat())
    }
    return out
}
