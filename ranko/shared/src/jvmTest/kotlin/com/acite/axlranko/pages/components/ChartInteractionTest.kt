package com.acite.axlranko.pages.components

import androidx.compose.ui.graphics.Color
import com.acite.axlranko.model.MetricPoint
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class ChartInteractionTest {

    private val bounds = ChartWindow(0f, 1_000f)
    private val autoA = ChartViewport(0f, 100f, 0f, 10f)
    private val autoB = ChartViewport(0f, 100f, 100f, 110f)
    private val autoC = ChartViewport(0f, 100f, 200f, 210f)

    private fun store(): StepChartInteractionStore = StepChartInteractionStore().apply {
        reportAuto(STEP_CHART_AVG_LOSS, autoA, emptyMap(), bounds)
        reportAuto(STEP_CHART_TRAIN_LOSS, autoB, emptyMap(), bounds)
        reportAuto(STEP_CHART_LEARNING_RATE, autoC, emptyMap(), bounds)
    }

    @Test
    fun xEditsDetachAndSynchronizeEveryStepChart() {
        val store = store()
        store.updateViewport(
            chartId = STEP_CHART_AVG_LOSS,
            previous = autoA,
            next = autoA.copy(xMin = 250f, xMax = 350f),
            xChanged = true,
            yChanged = false,
        )

        assertTrue(store.state.detached)
        assertEquals(ChartWindow(250f, 350f), store.state.xWindow)
        assertEquals(ChartWindow(250f, 350f), store.effectiveViewport(STEP_CHART_TRAIN_LOSS, autoB).xWindow)
        assertEquals(ChartWindow(250f, 350f), store.effectiveViewport(STEP_CHART_LEARNING_RATE, autoC).xWindow)
        assertEquals(autoB.yWindow, store.effectiveViewport(STEP_CHART_TRAIN_LOSS, autoB).yWindow)
        assertEquals(autoC.yWindow, store.effectiveViewport(STEP_CHART_LEARNING_RATE, autoC).yWindow)
    }

    @Test
    fun yEditsStayIndependentAfterTheSharedDetach() {
        val store = store()
        store.updateViewport(
            chartId = STEP_CHART_AVG_LOSS,
            previous = autoA,
            next = autoA.copy(xMin = 250f, xMax = 350f),
            xChanged = true,
            yChanged = false,
        )
        store.updateViewport(
            chartId = STEP_CHART_TRAIN_LOSS,
            previous = store.effectiveViewport(STEP_CHART_TRAIN_LOSS, autoB),
            next = ChartViewport(250f, 350f, 105f, 115f),
            xChanged = false,
            yChanged = true,
        )

        assertEquals(ChartWindow(105f, 115f), store.effectiveViewport(STEP_CHART_TRAIN_LOSS, autoB).yWindow)
        assertEquals(autoA.yWindow, store.effectiveViewport(STEP_CHART_AVG_LOSS, autoA).yWindow)
        assertEquals(autoC.yWindow, store.effectiveViewport(STEP_CHART_LEARNING_RATE, autoC).yWindow)
    }

    @Test
    fun newAutoDataCannotReplaceADetachedViewport() {
        val store = store()
        store.updateViewport(
            chartId = STEP_CHART_AVG_LOSS,
            previous = autoA,
            next = autoA.copy(xMin = 250f, xMax = 350f),
            xChanged = true,
            yChanged = false,
        )
        store.reportAuto(STEP_CHART_AVG_LOSS, ChartViewport(900f, 1_000f, 50f, 60f), emptyMap(), bounds)
        store.reportAuto(STEP_CHART_TRAIN_LOSS, ChartViewport(900f, 1_000f, 70f, 80f), emptyMap(), bounds)

        assertEquals(ChartWindow(250f, 350f), store.effectiveViewport(STEP_CHART_AVG_LOSS, autoA).xWindow)
        assertEquals(autoB.yWindow, store.effectiveViewport(STEP_CHART_TRAIN_LOSS, autoB).yWindow)
    }

    @Test
    fun resetRestoresAutoAndKeepsSeriesVisibility() {
        val store = store()
        store.toggleSeries(STEP_CHART_AVG_LOSS, "Avg Loss")
        store.updateViewport(
            chartId = STEP_CHART_AVG_LOSS,
            previous = autoA,
            next = autoA.copy(xMin = 250f, xMax = 350f),
            xChanged = true,
            yChanged = false,
        )

        store.resetView()

        assertFalse(store.state.detached)
        assertNull(store.state.xWindow)
        assertEquals(setOf("Avg Loss"), store.hiddenSeries(STEP_CHART_AVG_LOSS))
        assertEquals(autoA.xWindow, store.effectiveViewport(STEP_CHART_AVG_LOSS, autoA).xWindow)
    }

    @Test
    fun hoverBelongsToTheChartThatOwnsIt() {
        val store = store()
        store.setHover(STEP_CHART_AVG_LOSS, 42f)
        store.clearHover(STEP_CHART_TRAIN_LOSS)
        assertEquals(42f, store.state.hoverStep)

        store.clearHover(STEP_CHART_AVG_LOSS)
        assertNull(store.state.hoverStep)
        assertNull(store.state.hoverOwner)
    }

    @Test
    fun detachedAxisDomainsStayAttachedToTheirLabels() {
        val store = StepChartInteractionStore()
        val initial = listOf(
            ChartSeries("UNet", listOf(MetricPoint(1, 1f)), Color.Red, domainMin = 0f, domainMax = 1f),
            ChartSeries("TE", listOf(MetricPoint(1, 2f)), Color.Blue, domainMin = 10f, domainMax = 20f),
        )
        store.reportAuto(
            STEP_CHART_LEARNING_RATE,
            autoC,
            initial.associate { it.label to ChartAxisDomain(it.domainMin, it.domainMax) },
            bounds,
        )
        store.updateViewport(
            chartId = STEP_CHART_LEARNING_RATE,
            previous = autoC,
            next = autoC.copy(xMin = 10f, xMax = 20f),
            xChanged = true,
            yChanged = false,
        )
        val changed = listOf(
            ChartSeries("TE", listOf(MetricPoint(1, 2f)), Color.Blue, domainMin = 30f, domainMax = 40f),
            ChartSeries("UNet", listOf(MetricPoint(1, 1f)), Color.Red, domainMin = 50f, domainMax = 60f),
        )

        val effective = store.effectiveSeries(STEP_CHART_LEARNING_RATE, changed)

        assertEquals(0f, effective.first { it.label == "UNet" }.domainMin)
        assertEquals(1f, effective.first { it.label == "UNet" }.domainMax)
        assertEquals(10f, effective.first { it.label == "TE" }.domainMin)
        assertEquals(20f, effective.first { it.label == "TE" }.domainMax)
    }

    @Test
    fun xBoundsUnionCoversEveryStepChart() {
        val store = StepChartInteractionStore()
        store.reportAuto(STEP_CHART_AVG_LOSS, autoA, emptyMap(), ChartWindow(0f, 100f))
        store.reportAuto(STEP_CHART_TRAIN_LOSS, autoB, emptyMap(), ChartWindow(50f, 500f))

        assertEquals(ChartWindow(0f, 500f), store.globalXBounds())
    }
}
