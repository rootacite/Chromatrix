package com.acite.axlranko.pages.components

import com.acite.axlranko.model.CheckpointPin
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class ChartMarkerTest {

    private fun pin(step: Int?) = CheckpointPin(path = "p$step", dir = "run", step = step)

    @Test
    fun pinnedStepsSkipMissingDedupeAndSort() {
        assertEquals(
            listOf(100, 200, 500),
            pinnedMarkerSteps(listOf(pin(200), pin(null), pin(500), pin(100), pin(200))),
        )
        assertEquals(emptyList(), pinnedMarkerSteps(emptyList()))
        assertEquals(emptyList(), pinnedMarkerSteps(listOf(pin(null), pin(null))))
    }

    @Test
    fun hoverPreviewSnapsToTheNearestStepPreferringTheLowerOnTies() {
        assertEquals(100, nearestMarkedStep(listOf(100, 200), 118f))
        assertEquals(150, nearestMarkedStep(listOf(100, 150, 200), 149f))
        assertEquals(100, nearestMarkedStep(listOf(100, 200), 150f))
        assertEquals(200, nearestMarkedStep(listOf(100, 200), 9_999f))
        assertNull(nearestMarkedStep(emptyList(), 50f))
    }

    @Test
    fun chartHeightClampKeepsDraggedHeightsUsable() {
        assertEquals(CHART_HEIGHT_MIN_DP, clampChartHeight(40f))
        assertEquals(CHART_HEIGHT_MAX_DP, clampChartHeight(5_000f))
        assertEquals(320f, clampChartHeight(320f))
        assertEquals(CHART_HEIGHT_MIN_DP, clampChartHeight(CHART_HEIGHT_MIN_DP))
        assertEquals(CHART_HEIGHT_MAX_DP, clampChartHeight(CHART_HEIGHT_MAX_DP))
    }
}
