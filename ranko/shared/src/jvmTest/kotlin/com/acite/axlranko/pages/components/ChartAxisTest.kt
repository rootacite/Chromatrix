package com.acite.axlranko.pages.components

import androidx.compose.ui.graphics.Color
import com.acite.axlranko.model.MetricPoint
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * The multi-axis chart's own arithmetic: which axis each curve is drawn against, that each axis is
 * fitted and clipped on its own curve, and where the right-hand axis labels land.
 */
class ChartAxisTest {

    private fun series(label: String, values: List<Pair<Int, Float>>, color: Color = Color.Red) =
        ChartSeries(label, values.map { MetricPoint(it.first, it.second) }, color)

    /**
     * One long steady curve per magnitude — a UNet rate, a text-encoder rate and a loss variance —
     * each with a single old spike at step 0. With 40 points the spike is deep enough in the tail
     * that a 15 % clip drops it.
     */
    private fun threeScales() = listOf(
        series("UNet LR", listOf(0 to 1.2e-4f) + steady(1.0e-4f)),
        series("TE LR", listOf(0 to 1.2e-5f) + steady(1.0e-5f)),
        series("Val Fixed Var", listOf(0 to 9f) + steady(0.5f)),
    )

    private fun steady(value: Float) = (10..390 step 10).map { it to value }

    private fun assigned(axisCount: Int, outlierClip: Float = 0.15f) = assignAxisDomains(
        series = threeScales(),
        axisCount = axisCount,
        windowX = 0f to 390f,
        outlierClip = outlierClip,
        smoothing = 0f,
    )

    @Test
    fun eachCurveGetsAnAxisOfItsOwnScale() {
        val domains = assigned(axisCount = 3).map { it.domainMin!! to it.domainMax!! }
        val (unet, unetHi) = domains[0]
        val (te, teHi) = domains[1]
        val (variance, varianceHi) = domains[2]
        // Each axis hugs its own curve's steady value in its own units: the learning rates are two
        // orders of magnitude apart and the variance is a plain 0.5.
        assertTrue(unet > 0.9e-4f && unetHi < 1.1e-4f, "UNet axis was not in its own units: $unet..$unetHi")
        assertTrue(te > 0.9e-5f && teHi < 1.1e-5f, "TE axis was not in its own units: $te..$teHi")
        assertTrue(variance > 0.4f && varianceHi < 0.6f, "variance axis was not its own scale: $variance..$varianceHi")
        // The three curves really are far apart, and each axis is a thousandth of that spread: a
        // single shared scale (what a one-axis chart uses) would have drawn the learning rates as
        // two flat lines on the floor of the variance curve's plot.
        val allValues = threeScales().flatMap { item -> item.points.map { it.value } }
        val spread = allValues.max() - allValues.min()
        assertTrue(spread > 1f, "the three curves were not far apart: $spread")
        assertTrue(
            (unetHi - unet) < spread / 1000f,
            "the UNet axis did not separate from a shared scale: $unet..$unetHi against a spread of $spread",
        )
        assertTrue(
            (varianceHi - variance) < spread / 4f,
            "the variance axis swallowed the shared scale: $variance..$varianceHi against $spread",
        )
    }

    @Test
    fun eachAxisIsClippedOnItsOwnCurve() {
        val domains = assigned(axisCount = 3, outlierClip = 0.15f).map { it.domainMin!! to it.domainMax!! }
        // The old spike stays outside every axis.
        assertTrue(domains[0].second < 1.1e-4f, "the UNet spike reached its axis: ${domains[0]}")
        assertTrue(domains[1].second < 1.1e-5f, "the TE spike reached its axis: ${domains[1]}")
        assertTrue(domains[2].second < 1f, "the variance spike reached its axis: ${domains[2]}")
        // With the clip off, each axis reaches its own curve's raw maximum.
        val unclipped = assigned(axisCount = 3, outlierClip = 0f).map { it.domainMin!! to it.domainMax!! }
        assertTrue(unclipped[0].second >= 1.2e-4f, "the UNet spike was not the unclipped high end: ${unclipped[0]}")
        assertTrue(unclipped[1].second >= 1.2e-5f, "the TE spike was not the unclipped high end: ${unclipped[1]}")
        assertTrue(unclipped[2].second >= 9f, "the variance spike was not the unclipped high end: ${unclipped[2]}")
    }

    @Test
    fun aCurvePastTheLastAxisKeepsThatAxisScale() {
        // More series than axes must not be drawn against a scale the chart does not label: the
        // extra curves share the last axis.
        val domains = assigned(axisCount = 2)
        assertEquals(domains[1].domainMin, domains[2].domainMin)
        assertEquals(domains[1].domainMax, domains[2].domainMax)
    }

    @Test
    fun aCurveWithNothingInTheWindowLeavesItsAxisUnlabelled() {
        val result = assignAxisDomains(
            series = listOf(
                series("UNet LR", steady(1.0e-4f)),
                series("TE LR", emptyList()),
            ),
            axisCount = 2,
            windowX = 0f to 390f,
            outlierClip = 0.15f,
            smoothing = 0f,
        )
        assertNotNull(result[0].domainMin)
        // No values means no axis: the chart draws the one curve it has, with one left-hand scale.
        assertNull(result[1].domainMin)
        assertNull(result[1].domainMax)
    }

    @Test
    fun aHiddenCurveLeavesItsOwnAxisUncomputedWithoutMovingTheOthers() {
        val result = assignAxisDomains(
            series = threeScales(),
            axisCount = 3,
            windowX = 0f to 390f,
            outlierClip = 0.15f,
            smoothing = 0f,
            hiddenLabels = setOf("TE LR"),
        )

        assertNotNull(result[0].domainMin)
        assertNull(result[1].domainMin)
        assertNull(result[1].domainMax)
        assertNotNull(result[2].domainMin)
    }

    @Test
    fun anExplicitDomainIsKept() {
        // A series that already declares its range (`domainMin`/`domainMax`) keeps it, and that
        // range is its axis.
        val explicit = series("TE LR", steady(5.0e-5f)).copy(domainMin = 1f, domainMax = 2f)
        val result = assignAxisDomains(
            series = listOf(series("UNet LR", steady(1.0e-4f)), explicit),
            axisCount = 2,
            windowX = 0f to 390f,
            outlierClip = 0.15f,
            smoothing = 0f,
        )
        assertEquals(1f, result[1].domainMin)
        assertEquals(2f, result[1].domainMax)
    }

    @Test
    fun aSingleAxisChartLeavesEveryCurveAlone() {
        val input = threeScales()
        val result = assignAxisDomains(
            series = input,
            axisCount = 1,
            windowX = 0f to 390f,
            outlierClip = 0.15f,
            smoothing = 0f,
        )
        assertEquals(input, result)
        assertNull(result[0].domainMin)
    }

    @Test
    fun anExplicitDomainStillPutsAOneAxisChartInItsOwnSpace() {
        // The hardware charts' 0–100 percentage / capacity axes: a series that declares its own
        // range draws in the normalized space, which `normalized` reads off the series themselves —
        // not off the axis count — so those charts keep the range they always had.
        val util = ChartSeries(
            "Util",
            listOf(MetricPoint(0, 20f), MetricPoint(1, 30f)),
            Color.Red,
            domainMin = 0f,
            domainMax = 100f,
        )
        val plain = ChartSeries("Loss", listOf(MetricPoint(0, 1f), MetricPoint(1, 2f)), Color.Red)
        assertTrue(chartUsesOwnDomains(listOf(util)))
        assertFalse(chartUsesOwnDomains(listOf(plain)))
        assertFalse(chartUsesOwnDomains(emptyList()))

        // A one-axis chart hands its series through unchanged, so the explicit range survives.
        val assigned = assignAxisDomains(
            series = listOf(util),
            axisCount = 1,
            windowX = 0f to 10f,
            outlierClip = 0f,
            smoothing = 0f,
        )
        assertEquals(util, assigned.single())
        assertTrue(chartUsesOwnDomains(assigned))
        // Two series, only one of them explicit: still their own space.
        assertTrue(chartUsesOwnDomains(listOf(plain, util)))
    }

    @Test
    fun rightAxisLabelsStackFromThePlotOutwards() {
        // Two right axes: the inner one starts just past the plot, the outer one is right-aligned to
        // the card edge, which is where a two-axis chart has always put its single right label.
        assertEquals(306f, rightAxisLabelX(index = 0, count = 2, plotRight = 300f, totalWidth = 400f, labelWidth = 40f))
        assertEquals(360f, rightAxisLabelX(index = 1, count = 2, plotRight = 300f, totalWidth = 400f, labelWidth = 40f))
        // One right axis is the outer (and only) one.
        assertEquals(360f, rightAxisLabelX(index = 0, count = 1, plotRight = 300f, totalWidth = 400f, labelWidth = 40f))
        // Each further column is one padding wide.
        assertEquals(
            PLOT_RIGHT_PADDING,
            rightAxisLabelX(index = 1, count = 3, plotRight = 300f, totalWidth = 500f, labelWidth = 40f) -
                rightAxisLabelX(index = 0, count = 3, plotRight = 300f, totalWidth = 500f, labelWidth = 40f),
        )
    }
}
