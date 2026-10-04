package com.acite.axlranko.pages.components

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue

internal const val STEP_CHART_AVG_LOSS = "avg-loss"
internal const val STEP_CHART_TRAIN_LOSS = "train-loss"
internal const val STEP_CHART_LEARNING_RATE = "learning-rate"

/** A data-space interval used by the shared step-chart interaction state. */
internal data class ChartWindow(
    val min: Float,
    val max: Float,
) {
    val range: Float
        get() = (max - min).coerceAtLeast(1e-6f)

    fun clampTo(bounds: ChartWindow): ChartWindow {
        val width = range.coerceAtMost(bounds.range)
        val start = if (width >= bounds.range) {
            bounds.min
        } else {
            min.coerceIn(bounds.min, maxOf(bounds.min, bounds.max - width))
        }
        return ChartWindow(start, start + width)
    }
}

/** The viewport a chart draws in. The step charts share x while keeping their own y ranges. */
internal data class ChartViewport(
    val xMin: Float,
    val xMax: Float,
    val yMin: Float,
    val yMax: Float,
) {
    val xRange: Float
        get() = (xMax - xMin).coerceAtLeast(1e-6f)

    val yRange: Float
        get() = (yMax - yMin).coerceAtLeast(1e-6f)

    val xWindow: ChartWindow
        get() = ChartWindow(xMin, xMax)

    val yWindow: ChartWindow
        get() = ChartWindow(yMin, yMax)

    fun clampToBounds(bounds: ChartViewport): ChartViewport {
        val x = xWindow.clampTo(bounds.xWindow)
        val y = yWindow.clampTo(bounds.yWindow)
        return ChartViewport(x.min, x.max, y.min, y.max)
    }
}

/** The optional y range of one curve, retained while a detached chart keeps its axis scale. */
internal data class ChartAxisDomain(
    val min: Float?,
    val max: Float?,
)

private data class ChartAutoSnapshot(
    val viewport: ChartViewport,
    val domains: Map<String, ChartAxisDomain>,
    val xBounds: ChartWindow,
)

/**
 * In-memory state shared by the three step-axis charts. It deliberately lives outside the chart
 * composables so scrolling a chart out of a LazyColumn cannot drop a user's viewport.
 */
internal data class StepChartInteractionState(
    val detached: Boolean = false,
    val xWindow: ChartWindow? = null,
    val yWindows: Map<String, ChartWindow> = emptyMap(),
    val axisDomains: Map<String, Map<String, ChartAxisDomain>> = emptyMap(),
    val hiddenSeries: Map<String, Set<String>> = emptyMap(),
    val hoverOwner: String? = null,
    val hoverStep: Float? = null,
)

internal class StepChartInteractionStore {
    var state by mutableStateOf(StepChartInteractionState())
        private set

    private val auto = mutableMapOf<String, ChartAutoSnapshot>()

    fun hiddenSeries(chartId: String): Set<String> =
        state.hiddenSeries[chartId].orEmpty()

    fun toggleSeries(chartId: String, label: String) {
        val hidden = state.hiddenSeries[chartId].orEmpty()
        val next = if (label in hidden) hidden - label else hidden + label
        state = state.copy(hiddenSeries = state.hiddenSeries + (chartId to next))
    }

    fun reportAuto(
        chartId: String,
        viewport: ChartViewport,
        domains: Map<String, ChartAxisDomain>,
        xBounds: ChartWindow,
    ) {
        auto[chartId] = ChartAutoSnapshot(viewport, domains, xBounds)
    }

    fun effectiveViewport(chartId: String, autoViewport: ChartViewport): ChartViewport {
        val current = state
        if (!current.detached) return autoViewport
        val x = current.xWindow?.clampTo(globalXBounds() ?: autoViewport.xWindow)
            ?: autoViewport.xWindow
        val y = current.yWindows[chartId] ?: autoViewport.yWindow
        return ChartViewport(x.min, x.max, y.min, y.max)
    }

    fun effectiveDomains(
        chartId: String,
        autoDomains: Map<String, ChartAxisDomain>,
    ): Map<String, ChartAxisDomain> {
        val frozen = state.axisDomains[chartId] ?: return autoDomains
        return autoDomains.mapValues { (label, domain) -> frozen[label] ?: domain }
    }

    fun effectiveSeries(chartId: String, autoSeries: List<ChartSeries>): List<ChartSeries> {
        val domains = effectiveDomains(
            chartId,
            autoSeries.associate { it.label to ChartAxisDomain(it.domainMin, it.domainMax) },
        )
        return autoSeries.map { item ->
            val domain = domains[item.label] ?: ChartAxisDomain(item.domainMin, item.domainMax)
            item.copy(domainMin = domain.min, domainMax = domain.max)
        }
    }

    /** Applies one accepted drag or wheel edit, entering Detach on the first one. */
    fun updateViewport(
        chartId: String,
        previous: ChartViewport,
        next: ChartViewport,
        xChanged: Boolean,
        yChanged: Boolean,
    ) {
        val current = state
        if (!current.detached) {
            val xBounds = globalXBounds() ?: previous.xWindow
            val sharedX = if (xChanged) next.xWindow else previous.xWindow
            val capturedY = auto.mapValues { it.value.viewport.yWindow }.toMutableMap()
            capturedY[chartId] = previous.yWindow
            if (yChanged) capturedY[chartId] = next.yWindow
            state = current.copy(
                detached = true,
                xWindow = sharedX.clampTo(xBounds),
                yWindows = capturedY,
                axisDomains = auto.mapValues { it.value.domains },
            )
            return
        }

        state = current.copy(
            xWindow = if (xChanged) {
                (current.xWindow ?: previous.xWindow).let { old ->
                    if (xChanged) next.xWindow.clampTo(globalXBounds() ?: old) else old
                }
            } else {
                current.xWindow
            },
            yWindows = if (yChanged) {
                current.yWindows + (chartId to next.yWindow)
            } else {
                current.yWindows
            },
        )
    }

    fun setHover(chartId: String, step: Float) {
        state = state.copy(hoverOwner = chartId, hoverStep = step)
    }

    fun clearHover(chartId: String) {
        if (state.hoverOwner != chartId) return
        state = state.copy(hoverOwner = null, hoverStep = null)
    }

    /** Reset returns the charts to automatic viewport fitting but keeps curve visibility. */
    fun resetView() {
        auto.clear()
        state = state.copy(
            detached = false,
            xWindow = null,
            yWindows = emptyMap(),
            axisDomains = emptyMap(),
            hoverOwner = null,
            hoverStep = null,
        )
    }

    /** Run changes get a clean chart interaction state, including visibility. */
    fun resetAll() {
        auto.clear()
        state = StepChartInteractionState()
    }

    fun globalXBounds(): ChartWindow? {
        val bounds = auto.values.map { it.xBounds }
        if (bounds.isEmpty()) return null
        return ChartWindow(bounds.minOf { it.min }, bounds.maxOf { it.max })
    }
}
