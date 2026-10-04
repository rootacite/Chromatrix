package com.acite.axlranko.pages.components

import com.acite.axlranko.model.MetricPoint
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The Learning Rate chart's emerald stability curve: a rolling *population* variance of the raw
 * `Val/Fixed_Loss` series over a closed `± steps_per_epoch / 4` window.
 */
class ChartVarianceTest {

    @Test
    fun theWindowIsAClosedQuarterEpochEitherSideOfEachPoint() {
        // Window length 40 → the window is ±10 steps, and the boundary step counts.
        val points = listOf(
            MetricPoint(step = 0, value = 5f),
            MetricPoint(step = 10, value = 1f),
            MetricPoint(step = 20, value = 2f),
        )
        val variance = rollingPopulationVariance(points, stepsPerEpoch = 40).associateBy { it.step }
        // Center 10: [0, 20] closed, so all three values are in — 5, 1, 2. Mean 8/3, population
        // variance 26/9. An open low end would have given (1, 2) → 0.25, which is the point.
        assertEquals(26f / 9f, variance.getValue(10).value, 1e-6f)
        assertTrue(variance.getValue(10).value != 0.25f)
        // Center 0: [-10, 10] holds 5 and 1 → mean 3, population variance 4.
        assertEquals(4f, variance.getValue(0).value, 1e-6f)
        // Center 20: [10, 30] holds 1 and 2 → mean 1.5, population variance 0.25.
        assertEquals(0.25f, variance.getValue(20).value, 1e-6f)
    }

    @Test
    fun theVarianceIsThePopulationFormNotTheSampleOne() {
        // Two values, 1 and 3: mean 2, and the population variance is 1 (the sample form would
        // report 2). A two-point window is common at the start of a run.
        val points = listOf(MetricPoint(step = 0, value = 1f), MetricPoint(step = 2, value = 3f))
        val variance = rollingPopulationVariance(points, stepsPerEpoch = 8)
        assertEquals(2, variance.size)
        assertTrue(variance.all { it.value == 1f }, "got ${variance.map { it.value }}")
    }

    @Test
    fun aWindowWithFewerThanTwoPointsIsOmitted() {
        // ±2 steps: the two points are far outside each other's window, so neither yields a value.
        val points = listOf(
            MetricPoint(step = 0, value = 1f),
            MetricPoint(step = 100, value = 2f),
        )
        assertEquals(emptyList(), rollingPopulationVariance(points, stepsPerEpoch = 8))
    }

    @Test
    fun withoutAnEpochLengthThereIsNoCurve() {
        val points = listOf(MetricPoint(step = 0, value = 1f), MetricPoint(step = 1, value = 3f))
        assertEquals(emptyList(), rollingPopulationVariance(points, stepsPerEpoch = null))
        assertEquals(emptyList(), rollingPopulationVariance(points, stepsPerEpoch = 0))
        assertEquals(emptyList(), rollingPopulationVariance(emptyList(), stepsPerEpoch = 40))
    }

    @Test
    fun eachPointKeepsItsOwnStepAndOrder() {
        val points = listOf(
            MetricPoint(step = 30, value = 3f),
            MetricPoint(step = 10, value = 1f),
            MetricPoint(step = 20, value = 2f),
            MetricPoint(step = 25, value = Float.NaN),
        )
        val variance = rollingPopulationVariance(points, stepsPerEpoch = 40)
        assertEquals(listOf(10, 20, 30), variance.map { it.step })
        // 10: [0, 20] → 1, 2 → 0.25; 20: [10, 30] → 1,2,3 → 2/3; 30: [20, 40] → 2, 3 → 0.25.
        assertEquals(listOf(0.25f, 2f / 3f, 0.25f), variance.map { it.value })
    }

    @Test
    fun nonFiniteValuesAreDroppedBeforeTheWindow() {
        val points = listOf(
            MetricPoint(step = 0, value = Float.NaN),
            MetricPoint(step = 1, value = 2f),
            MetricPoint(step = 2, value = 4f),
        )
        val variance = rollingPopulationVariance(points, stepsPerEpoch = 8)
        // ±2 window: the NaN cannot count, so every window holds both real values → mean 3,
        // population variance 1.
        assertEquals(2, variance.size)
        assertTrue(variance.all { it.value == 1f }, "got ${variance.map { it.value }}")
    }
}
