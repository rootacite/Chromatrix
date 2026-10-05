package com.acite.axlranko.pages

import androidx.compose.ui.unit.dp
import com.acite.axlranko.IoDispatcher
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.unit.DpSize
import com.acite.axlranko.data.TrainerIpcClient
import com.acite.axlranko.model.ChartPickState
import com.acite.axlranko.model.CheckpointExport
import com.acite.axlranko.model.CheckpointItem
import com.acite.axlranko.model.CheckpointSend
import com.acite.axlranko.model.CheckpointSendResult
import com.acite.axlranko.model.DashboardUiState
import com.acite.axlranko.model.EvaluationTarget
import com.acite.axlranko.model.GeneratedSampleJob
import com.acite.axlranko.model.HardwareHistory
import com.acite.axlranko.model.HardwareStatus
import com.acite.axlranko.model.MetricPoint
import com.acite.axlranko.model.RegenerateConfirm
import com.acite.axlranko.model.RegenerateError
import com.acite.axlranko.model.RunSummary
import com.acite.axlranko.model.SampleClearResult
import com.acite.axlranko.model.UnpinnedClearResult
import com.acite.axlranko.model.SampleItem
import com.acite.axlranko.model.SampleSetForm
import com.acite.axlranko.model.TrainStatus
import com.acite.axlranko.model.guessCharacterTrigger
import com.acite.axlranko.model.sampleSetInfos
import com.acite.axlranko.pages.components.JOB_ERROR
import com.acite.axlranko.pages.components.JOB_RUNNING
import com.acite.axlranko.pages.components.MAX_PINNED_ROUNDS
import com.acite.axlranko.pages.components.checkpointRows
import com.acite.axlranko.pages.components.checkpointsForRun
import com.acite.axlranko.pages.components.clampChartHeight
import com.acite.axlranko.pages.components.displayedRun
import com.acite.axlranko.pages.components.evaluationPrefillSelection
import com.acite.axlranko.pages.components.generateFormDefaults
import com.acite.axlranko.pages.components.generateFormError
import com.acite.axlranko.pages.components.generatedFailureLine
import com.acite.axlranko.pages.components.generatedSampleItems
import com.acite.axlranko.pages.components.nearestCheckpoint
import com.acite.axlranko.pages.components.newlyFailedJob
import com.acite.axlranko.pages.components.sampleRevisions
import com.acite.axlranko.pages.components.StepChartInteractionStore
import com.acite.axlranko.pages.components.sectionImages
import com.acite.axlranko.util.PathPicker
import com.acite.axlranko.util.TagLexicon
import com.acite.axlranko.util.checkpointSaveName
import com.acite.axlranko.util.ensureSafetensorsExtension
import com.acite.axlranko.util.formatBytes
import dev.zacsweers.metro.AppScope
import dev.zacsweers.metro.ContributesIntoMap
import dev.zacsweers.metro.Inject
import dev.zacsweers.metrox.viewmodel.ViewModelKey
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.JsonObject
import kotlin.math.roundToInt
import kotlin.time.Duration.Companion.milliseconds

@Inject
@ViewModelKey
@ContributesIntoMap(AppScope::class)
class DashboardScreenViewModel(
    private val ipc: TrainerIpcClient,
    private val pathPicker: PathPicker,
) : ViewModel() {

    private val _uiState = MutableStateFlow(DashboardUiState())
    val uiState: StateFlow<DashboardUiState> = _uiState.asStateFlow()

    /** Survives LazyColumn chart disposal and dashboard polling; never serialized to disk. */
    internal val chartInteractions = StepChartInteractionStore()

    private var pollingJob: Job? = null
    private var hardwareJob: Job? = null
    private var hardwareStep = 0
    private var entered = false

    /** Guards a second checkpoint scan while the first is still reading safetensors headers. */
    private var checkpointScanInFlight = false

    /** Jobs started in this session, which the panel highlights as new. */
    private val sessionJobIds = mutableSetOf<String>()
    private var generatedPollJob: Job? = null

    /** True while a pin is being written, so a poll that started before it cannot undo it. */
    private var pinsWriteInFlight = false
    private var chartViewWriteInFlight = false

    fun onEnter() {
        if (!entered) {
            entered = true
            startPolling()
            startHardwarePolling()
        } else {
            refreshNow()
        }
    }

    /**
     * The page is off screen: stop polling. `entered` goes back to false so coming back restarts
     * the loops, which is what they are for.
     *
     * The pollers used to keep running for the rest of the session — a poll chain of seven calls
     * plus an `nvtop` snapshot every second, behind whatever page the user had moved on to. That
     * is what made the Automation page lag while its own job was running.
     */
    fun onLeave() {
        pollingJob?.cancel()
        pollingJob = null
        hardwareJob?.cancel()
        hardwareJob = null
        entered = false
    }

    /**
     * Pins the dashboard to one run of the history list; `null` follows the current run.
     *
     * A pinned run brings its own samples, charts and checkpoints, so the previous run's
     * pick panel and preview are dropped rather than left pointing at another run's images.
     */
    fun selectRun(runId: String?) {
        chartInteractions.resetAll()
        val run = runId?.let { id -> _uiState.value.runs.firstOrNull { it.runId == id } }
        _uiState.update {
            it.copy(
                selectedRun = run,
                // Unpinning clears the id too: the run to follow is the next fetch's answer, and
                // a leftover id would keep the page on the run just unpinned until it lands.
                runId = run?.runId,
                // A Home click arrives before this page has ever listed the runs, so an id with no
                // entry behind it is kept for the next fetch to look up instead of dropped.
                pendingRunId = if (run == null) runId else null,
                chartPick = null,
                previewIndex = null,
                evaluationTarget = null,
                evaluationDetailsOpen = false,
                evaluationTagSelection = emptySet(),
                evaluationPrompts = null,
                evaluationPromptsError = null,
                evaluationError = null,
                // The prompts, and the outcome of the last clear, belong to the run they were
                // read for: the section reloads them for the run now being shown.
                samplePrompts = null,
                samplePromptsLoading = false,
                samplePromptsError = null,
                samplePromptsEditorOpen = false,
                clearingSamplesPath = null,
                clearSamplesResult = null,
                stepsPerEpoch = null,
                runSaveEveryNSteps = null,
                outlierClip = DashboardUiState().outlierClip,
                smoothExtraDp = DashboardUiState().smoothExtraDp,
                stepSpan = DashboardUiState().stepSpan,
                sampleThumbSize = DashboardUiState().sampleThumbSize,
                chartViewError = null,
                clearingUnpinned = false,
                unpinnedClearResult = null,
                samples = emptyMap(),
                checkpointPins = emptyList(),
                checkpointPinsFile = null,
                pinsError = null,
                latestStats = JsonObject(emptyMap()),
                metrics = emptyMap(),
            )
        }
        refreshNow()
    }

    fun toggleAutoRefresh(enabled: Boolean) {
        _uiState.update { it.copy(autoRefresh = enabled) }
        if (enabled) {
            startPolling()
            startHardwarePolling()
        } else {
            pollingJob?.cancel()
            hardwareJob?.cancel()
        }
    }

    fun setSmoothing(value: Float) {
        _uiState.update { it.copy(smoothing = value.coerceIn(0f, 0.99f)) }
    }

    fun setChartStroke(value: Float) {
        _uiState.update { it.copy(chartStroke = value.coerceIn(1f, 8f)) }
    }

    fun setSampleThumbSize(value: Float) {
        _uiState.update {
            it.copy(sampleThumbSize = value.roundToInt().coerceIn(80, 360).toFloat(), chartViewError = null)
        }
    }

    /** Newest steps a step-axis chart opens on. Stored as a whole number of steps. */
    fun setStepSpan(value: Float) {
        _uiState.update {
            it.copy(stepSpan = value.roundToInt().coerceIn(100, 8000).toFloat(), chartViewError = null)
        }
    }

    /** Tail fraction dropped when fitting Avg Loss and Train/Loss. 0.40 is 40%. */
    fun setOutlierClip(value: Float) {
        _uiState.update { it.copy(outlierClip = value.coerceIn(0f, 0.40f), chartViewError = null) }
    }

    /** Extra thickness of the smoothed stroke, in dp, stored to one decimal. */
    fun setSmoothExtraDp(value: Float) {
        val tenths = (value * 10f).roundToInt().coerceIn(0, 60)
        _uiState.update { it.copy(smoothExtraDp = tenths / 10f, chartViewError = null) }
    }

    /** The Avg Loss card's grip: apply one drag's vertical delta (dp) to its session height. */
    fun resizeChartHeightTop(deltaDp: Float) {
        if (!deltaDp.isFinite()) return
        _uiState.update {
            it.copy(chartHeightTop = clampChartHeight(it.chartHeightTop.value + deltaDp).dp)
        }
    }

    /** The Train/Loss and Learning Rate cards share one height, so either grip moves both. */
    fun resizeChartHeightSide(deltaDp: Float) {
        if (!deltaDp.isFinite()) return
        _uiState.update {
            it.copy(chartHeightSide = clampChartHeight(it.chartHeightSide.value + deltaDp).dp)
        }
    }

    /** Detach's Reset button: return every step chart to its automatic viewport. */
    fun resetChartView() {
        chartInteractions.resetView()
    }

    /**
     * The displayed run's chart sliders, from its log directory. A poll does not call this: only a
     * run change does, and a save that is still in flight keeps the value the user just set.
     */
    fun loadChartView() {
        val state = _uiState.value
        val shown = displayedRun(state.runs, state.selectedRun, state.runId)
        val runId = shown?.runId ?: state.runId
        if (runId.isNullOrBlank() || chartViewWriteInFlight) return
        viewModelScope.launch {
            try {
                val response = withContext(IoDispatcher) {
                    ipc.chartView(name = shown?.outputName, runId = runId)
                }
                _uiState.update { current ->
                    if (chartViewWriteInFlight) return@update current
                    val now = displayedRun(current.runs, current.selectedRun, current.runId)
                    if ((now?.runId ?: current.runId) != runId) return@update current
                    current.copy(
                        smoothExtraDp = response.smoothExtraDp,
                        outlierClip = response.outlierClip,
                        stepSpan = response.stepSpan.coerceIn(100, 8000).toFloat(),
                        sampleThumbSize = response.sampleThumbDp.coerceIn(80, 360).toFloat(),
                        chartViewError = null,
                    )
                }
            } catch (e: Exception) {
                _uiState.update { it.copy(chartViewError = e.message ?: e.toString()) }
            }
        }
    }

    /** Write the sliders into the displayed run's log directory. No run means nothing is stored. */
    fun saveChartView() {
        val state = _uiState.value
        val shown = displayedRun(state.runs, state.selectedRun, state.runId)
        val runId = shown?.runId ?: state.runId
        if (runId.isNullOrBlank()) return
        chartViewWriteInFlight = true
        viewModelScope.launch {
            try {
                val response = withContext(IoDispatcher) {
                    ipc.setChartView(
                        smoothExtraDp = state.smoothExtraDp,
                        outlierClip = state.outlierClip,
                        stepSpan = state.stepSpan.roundToInt(),
                        sampleThumbDp = state.sampleThumbSize.roundToInt(),
                        name = shown?.outputName,
                        runId = runId,
                    )
                }
                _uiState.update { current ->
                    val now = displayedRun(current.runs, current.selectedRun, current.runId)
                    if ((now?.runId ?: current.runId) != runId) return@update current
                    current.copy(
                        smoothExtraDp = response.smoothExtraDp,
                        outlierClip = response.outlierClip,
                        stepSpan = response.stepSpan.coerceIn(100, 8000).toFloat(),
                        sampleThumbSize = response.sampleThumbDp.coerceIn(80, 360).toFloat(),
                        chartViewError = null,
                    )
                }
            } catch (e: Exception) {
                _uiState.update { it.copy(chartViewError = e.message ?: e.toString()) }
            } finally {
                chartViewWriteInFlight = false
            }
        }
    }

    fun openPreview(sample: SampleItem) {
        val index = currentPreviewList().indexOfFirst { it.path == sample.path }
        if (index >= 0) {
            _uiState.update { it.copy(previewIndex = index) }
        }
    }

    fun closePreview() {
        _uiState.update { it.copy(previewIndex = null) }
    }

    /**
     * Ctrl+click on the Avg Loss chart: resolve the checkpoint nearest to [step] from the list the
     * page already polls, then rescan in the background and re-resolve in place.
     */
    fun pickCheckpointAt(step: Float, anchor: Offset) {
        val previous = _uiState.value.chartPick
        val defaults = generateFormDefaults(_uiState.value.config)
        _uiState.update {
            it.copy(
                chartPick = ChartPickState(
                    step = step,
                    anchor = anchor,
                    checkpoint = nearestCheckpoint(it.checkpoints, step),
                    isLoading = it.checkpoints.isEmpty(),
                    // A prompt typed for an earlier pick survives; an untouched form is re-seeded
                    // from config.toml's sample settings.
                    prompt = previous?.prompt?.takeIf { text -> text.isNotBlank() } ?: defaults.prompt,
                    negativePrompt = previous?.negativePrompt?.takeIf { text -> text.isNotBlank() }
                        ?: defaults.negativePrompt,
                    cfg = previous?.cfg?.takeIf { text -> text.isNotBlank() } ?: defaults.cfg,
                    steps = previous?.steps?.takeIf { text -> text.isNotBlank() } ?: defaults.steps,
                    seed = previous?.seed?.takeIf { text -> text.isNotBlank() } ?: defaults.seed,
                    isFormOpen = previous?.isFormOpen ?: false,
                ),
            )
        }
        rescanCheckpoints()
        loadGeneratedSamples()
    }

    private fun rescanCheckpoints() {
        if (checkpointScanInFlight) return
        checkpointScanInFlight = true
        viewModelScope.launch {
            try {
                val shown = displayedRun(
                    _uiState.value.runs,
                    _uiState.value.selectedRun,
                    _uiState.value.runId,
                )
                val name = shown?.outputName ?: _uiState.value.selectedRun?.outputName
                val response = withContext(IoDispatcher) { ipc.listCheckpoints(name = name) }
                _uiState.update { state ->
                    val found = if (shown == null) {
                        emptyList()
                    } else {
                        checkpointsForRun(response.checkpoints, shown.runId)
                    }
                    val pick = state.chartPick ?: return@update state.copy(checkpoints = found)
                    state.copy(
                        checkpoints = found,
                        chartPick = pick.copy(
                            checkpoint = nearestCheckpoint(found, pick.step),
                            isLoading = false,
                            error = null,
                        ),
                    )
                }
            } catch (e: Exception) {
                _uiState.update { state ->
                    val pick = state.chartPick ?: return@update state
                    state.copy(
                        chartPick = pick.copy(isLoading = false, error = e.message ?: e.toString()),
                    )
                }
            } finally {
                checkpointScanInFlight = false
            }
        }
    }

    fun dismissChartPick() {
        generatedPollJob?.cancel()
        _uiState.update { it.copy(chartPick = null) }
    }

    /** Remembers a dragged panel size for the rest of the session; the UI clamps it to the window. */
    fun setChartPanelSize(size: DpSize) {
        _uiState.update { it.copy(chartPanelSize = size) }
    }

    /** Edits the panel's generate form; a change clears the message of the previous attempt. */
    fun updateChartPickForm(transform: ChartPickState.() -> ChartPickState) {
        _uiState.update { state ->
            val pick = state.chartPick ?: return@update state
            state.copy(chartPick = pick.transform().copy(formError = null))
        }
    }

    fun toggleGenerateForm() {
        updateChartPickForm { copy(isFormOpen = !isFormOpen) }
    }

    /**
     * Loads the run's generated samples from disk; a job still running keeps the poll loop alive.
     */
    fun loadGeneratedSamples() {
        val selected = _uiState.value.selectedRun
        val runId = selected?.runId ?: _uiState.value.runId
        if (runId.isNullOrBlank()) return
        viewModelScope.launch {
            val jobs = fetchGeneratedJobs(runId, selected?.outputName)
            _uiState.update { state ->
                state.copy(generatedJobs = jobs, sampleRevisions = sampleRevisions(jobs))
            }
            if (jobs.any { it.state == JOB_RUNNING }) startGeneratedPolling()
        }
    }

    /**
     * "Generate a sample with this checkpoint": validate the form, hand it to api.py (which spawns
     * the generator detached) and follow the job until it finishes.
     *
     * [rowStep] is the sample row the panel is showing — the checkpoint's own step, or the nearest
     * sampled step when that one has no images — so the new image lands in the row the user sees.
     */
    fun generateSample(rowStep: Int? = null) {
        val pick = _uiState.value.chartPick ?: return
        val checkpoint = pick.checkpoint ?: return
        if (pick.isGenerating) return

        val error = generateFormError(pick.prompt, pick.cfg, pick.steps, pick.seed)
        if (error != null) {
            updateChartPickForm { copy(formError = error) }
            return
        }

        updateChartPickForm { copy(isGenerating = true) }
        viewModelScope.launch {
            try {
                val selected = _uiState.value.selectedRun
                val response = withContext(IoDispatcher) {
                    ipc.generateSample(
                        checkpoint = checkpoint.path,
                        prompt = pick.prompt,
                        negativePrompt = pick.negativePrompt,
                        cfg = pick.cfg.trim().toFloat(),
                        steps = pick.steps.trim().toInt(),
                        seed = pick.seed.trim().toLong(),
                        step = rowStep ?: checkpoint.step,
                        name = selected?.outputName,
                        runId = selected?.runId ?: _uiState.value.runId,
                    )
                }
                sessionJobIds += response.job.id
                _uiState.update { state ->
                    state.copy(
                        sessionJobIds = sessionJobIds.toSet(),
                        generatedJobs = (listOf(response.job) + state.generatedJobs).distinctBy { it.id },
                        generatedError = null,
                        chartPick = state.chartPick?.copy(isGenerating = false),
                    )
                }
                startGeneratedPolling()
            } catch (e: Exception) {
                _uiState.update { state ->
                    state.copy(
                        generatedError = e.message ?: e.toString(),
                        chartPick = state.chartPick?.copy(isGenerating = false),
                    )
                }
            }
        }
    }

    /**
     * The ↻ button on a sample thumbnail: ask the helper what a redraw of [sample] would do, then
     * open the confirmation dialog with the plan's warning — an image that records no seed cannot
     * be redrawn on the same one. Nothing is rendered until [confirmRegenerate].
     */
    fun planRegenerateSample(sample: SampleItem, checkpoint: CheckpointItem) {
        val path = sample.path
        if (_uiState.value.regenerateLoadingPath != null) return
        _uiState.update { it.copy(regenerateLoadingPath = path, regenerateError = null) }
        viewModelScope.launch {
            try {
                val selected = _uiState.value.selectedRun
                val plan = withContext(IoDispatcher) {
                    ipc.regenerateSample(
                        path = path,
                        checkpoint = checkpoint.path,
                        planOnly = true,
                        name = selected?.outputName,
                        runId = selected?.runId ?: _uiState.value.runId,
                    )
                }
                _uiState.update { state ->
                    state.copy(
                        regenerateLoadingPath = null,
                        regenerateConfirm = RegenerateConfirm(sample, checkpoint, plan.warn),
                    )
                }
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(
                        regenerateLoadingPath = null,
                        regenerateError = RegenerateError(
                            path = path,
                            checkpointPath = checkpoint.path,
                            message = e.message ?: e.toString(),
                        ),
                    )
                }
            }
        }
    }

    /**
     * Confirms the dialog: redraw the image over its own file. `allowNewSeed` accepts a fresh seed
     * for an image that records none — the dialog has already said so.
     */
    fun confirmRegenerate() {
        val confirm = _uiState.value.regenerateConfirm ?: return
        val path = confirm.sample.path
        _uiState.update {
            it.copy(regenerateConfirm = null, regenerateLoadingPath = path, regenerateError = null)
        }
        viewModelScope.launch {
            try {
                val selected = _uiState.value.selectedRun
                val response = withContext(IoDispatcher) {
                    ipc.regenerateSample(
                        path = path,
                        checkpoint = confirm.checkpoint.path,
                        allowNewSeed = true,
                        name = selected?.outputName,
                        runId = selected?.runId ?: _uiState.value.runId,
                    )
                }
                val job = response.job
                if (job == null) {
                    _uiState.update { it.copy(regenerateLoadingPath = null) }
                    return@launch
                }
                sessionJobIds += job.id
                _uiState.update { state ->
                    state.copy(
                        sessionJobIds = sessionJobIds.toSet(),
                        regenerateLoadingPath = null,
                        generatedJobs = (listOf(job) + state.generatedJobs).distinctBy { it.id },
                    )
                }
                startGeneratedPolling()
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(
                        regenerateLoadingPath = null,
                        regenerateError = RegenerateError(
                            path = path,
                            checkpointPath = confirm.checkpoint.path,
                            message = e.message ?: e.toString(),
                        ),
                    )
                }
            }
        }
    }

    fun cancelRegenerate() {
        _uiState.update { it.copy(regenerateConfirm = null) }
    }

    /**
     * "Generate samples" for one checkpoint: the config's whole `[[validation.samples]]` list,
     * rendered detached into the run's `_samples/generated/`. Only offered while the GPU is free
     * (the run is paused, stopped or over) — api.py refuses it otherwise.
     */
    fun generateCheckpointSamples(checkpoint: CheckpointItem) {
        if (_uiState.value.isGeneratingCheckpoint != null) return
        _uiState.update { it.copy(isGeneratingCheckpoint = checkpoint.path, generatedError = null) }
        viewModelScope.launch {
            try {
                val selected = _uiState.value.selectedRun
                val response = withContext(IoDispatcher) {
                    ipc.generateCheckpointSamples(
                        checkpoint = checkpoint.path,
                        name = selected?.outputName,
                        runId = selected?.runId ?: _uiState.value.runId,
                    )
                }
                sessionJobIds += response.job.id
                _uiState.update { state ->
                    state.copy(
                        sessionJobIds = sessionJobIds.toSet(),
                        isGeneratingCheckpoint = null,
                        generatedJobs = (listOf(response.job) + state.generatedJobs).distinctBy { it.id },
                    )
                }
                startGeneratedPolling()
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(
                        isGeneratingCheckpoint = null,
                        generatedError = e.message ?: e.toString(),
                    )
                }
            }
        }
    }

    /**
     * "Sample range": one detached job that renders the config's sample sets for every checkpoint
     * whose step is inside `fromStep..toStep`, oldest first.
     */
    fun startSampleBatch(fromStep: Int, toStep: Int) {
        if (_uiState.value.isStartingBatch) return
        _uiState.update { it.copy(isStartingBatch = true, batchError = null) }
        viewModelScope.launch {
            try {
                val selected = _uiState.value.selectedRun
                val response = withContext(IoDispatcher) {
                    ipc.generateCheckpointSamplesBatch(
                        fromStep = fromStep,
                        toStep = toStep,
                        name = selected?.outputName,
                        runId = selected?.runId ?: _uiState.value.runId,
                    )
                }
                sessionJobIds += response.job.id
                _uiState.update { state ->
                    state.copy(
                        sessionJobIds = sessionJobIds.toSet(),
                        isStartingBatch = false,
                        generatedJobs = (listOf(response.job) + state.generatedJobs).distinctBy { it.id },
                    )
                }
                startGeneratedPolling()
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(isStartingBatch = false, batchError = e.message ?: e.toString())
                }
            }
        }
    }

    /**
     * "Generate pinned samples": the detached pass over every valid pinned checkpoint of the shown
     * run, in the pin file's own order — repeated [rounds] times. The helper renders one pass per
     * call and refuses while another generation is running, so the rounds are sequenced here: the
     * next one starts only once the previous job has stopped. A round that failed, or that the user
     * stopped, ends the sequence. A stale pin cannot fail the batch: the helper drops it.
     */
    fun startPinnedSampleBatch(rounds: Int = 1) {
        if (_uiState.value.isStartingPinnedBatch) return
        val total = rounds.coerceIn(1, MAX_PINNED_ROUNDS)
        _uiState.update {
            it.copy(
                isStartingPinnedBatch = true,
                pinnedRoundIndex = 1,
                pinnedRoundsTotal = total,
                batchError = null,
            )
        }
        viewModelScope.launch {
            var failure: String? = null
            for (round in 1..total) {
                _uiState.update { it.copy(pinnedRoundIndex = round) }
                val selected = _uiState.value.selectedRun
                val runId = selected?.runId ?: _uiState.value.runId
                val started = try {
                    withContext(IoDispatcher) {
                        ipc.generatePinnedCheckpointSamples(name = selected?.outputName, runId = runId)
                    }
                } catch (e: Exception) {
                    failure = e.message ?: e.toString()
                    break
                }
                sessionJobIds += started.job.id
                _uiState.update { state ->
                    state.copy(
                        sessionJobIds = sessionJobIds.toSet(),
                        generatedJobs = (listOf(started.job) + state.generatedJobs).distinctBy { it.id },
                    )
                }
                startGeneratedPolling()
                if (round == total) break
                val finished = awaitGeneratedJob(started.job.id, runId)
                if (finished == null) break
                if (finished.state == JOB_ERROR) {
                    failure = finished.error ?: "the pinned-sample pass failed"
                    break
                }
                if (finished.cancelRequested) break
            }
            _uiState.update {
                it.copy(
                    isStartingPinnedBatch = false,
                    pinnedRoundIndex = 0,
                    pinnedRoundsTotal = 0,
                    batchError = failure ?: it.batchError,
                )
            }
        }
    }

    /**
     * Waits for one generation job to stop, reading `list_generated_samples` at the poller's own
     * cadence. Null when the job can no longer be followed — the run on screen changed, or the
     * helper kept reporting nothing for it — which ends the round sequence quietly.
     */
    private suspend fun awaitGeneratedJob(jobId: String, runId: String?): GeneratedSampleJob? {
        var misses = 0
        while (true) {
            delay(GENERATED_POLL_MILLIS.milliseconds)
            val state = _uiState.value
            if ((state.selectedRun?.runId ?: state.runId) != runId) return null
            val jobs = withContext(IoDispatcher) {
                runCatching {
                    ipc.listGeneratedSamples(name = state.selectedRun?.outputName, runId = runId).jobs
                }.getOrNull()
            }
            val job = jobs?.firstOrNull { it.id == jobId }
            if (job == null) {
                misses += 1
                if (misses > GENERATED_JOB_MISS_LIMIT) return null
                continue
            }
            if (job.state != JOB_RUNNING) return job
            misses = 0
        }
    }

    /**
     * Opens the evaluation panel for one checkpoint. [existingImages] is what the card already
     * shows, and it prefills the depth field — the depth is a floor, so that is the "score what is
     * there" default. [jobId] is the evaluation the panel should report on (a running one, or the
     * newest finished one), which is what lets the same panel show a result that was produced in an
     * earlier session. The tagger's own categories and the run's prompt tags are fetched once per
     * checkpoint; a helper that cannot answer leaves the pass on its own defaults.
     */
    fun openEvaluation(checkpoint: CheckpointItem, existingImages: Int, jobId: String? = null) {
        _uiState.update {
            it.copy(
                evaluationTarget = EvaluationTarget(checkpoint, existingImages, jobId),
                evaluationError = null,
                evaluationDetailsOpen = false,
                evaluationTagSelection = emptySet(),
                evaluationPrompts = null,
                evaluationPromptsError = null,
            )
        }
        if (_uiState.value.taggerInfo == null) {
            viewModelScope.launch {
                val info = runCatching { withContext(IoDispatcher) { ipc.taggerInfo() } }.getOrNull()
                _uiState.update { state -> state.copy(taggerInfo = info) }
            }
        }
        loadEvaluationPrompts(checkpoint)
    }

    /** Reopens the panel on one of a card's evaluation jobs (the card's `Evaluation` entry point). */
    fun showEvaluation(jobId: String) {
        val job = _uiState.value.generatedJobs.firstOrNull { it.id == jobId } ?: return
        val checkpoint = checkpointForEvaluation(job) ?: return
        openEvaluation(checkpoint, generatedSampleItems(job).size, jobId)
    }

    /**
     * The checkpoint a job names, as a card: the real one when it is still listed, else a stand-in
     * built from the job — a Reset can remove the weights while the evaluation's images and scores
     * stay on their `samples only` row, and its panel must still open.
     */
    private fun checkpointForEvaluation(job: GeneratedSampleJob): CheckpointItem? {
        val state = _uiState.value
        state.checkpoints.firstOrNull { it.path == job.checkpoint }?.let { return it }
        if (job.checkpoint.isBlank()) return null
        val parts = job.checkpoint.trimEnd('/').split('/')
        val shown = displayedRun(state.runs, state.selectedRun, state.runId) ?: state.selectedRun
        return CheckpointItem(
            path = job.checkpoint,
            dir = if (parts.size >= 2) parts[parts.size - 2] else "",
            filename = parts.lastOrNull().orEmpty(),
            step = job.step,
            outputName = shown?.outputName.orEmpty(),
        )
    }

    private fun loadEvaluationPrompts(checkpoint: CheckpointItem) {
        val state = _uiState.value
        if (state.evaluationPrompts?.checkpoint == checkpoint.path) return
        viewModelScope.launch {
            _uiState.update { it.copy(evaluationPromptsLoading = true) }
            try {
                val selected = _uiState.value.selectedRun
                val prompts = withContext(IoDispatcher) {
                    ipc.evaluationPrompts(
                        checkpoint = checkpoint.path,
                        name = selected?.outputName,
                        runId = selected?.runId ?: _uiState.value.runId,
                    )
                }
                _uiState.update { current ->
                    // A panel closed under the call keeps its own state; the reply's recorded
                    // selection is the one this run's last evaluation scored with.
                    if (current.evaluationTarget?.checkpoint?.path != prompts.checkpoint) {
                        return@update current.copy(evaluationPromptsLoading = false)
                    }
                    current.copy(
                        evaluationPrompts = prompts,
                        evaluationPromptsLoading = false,
                        evaluationTagSelection = evaluationPrefillSelection(
                            prompts.selectedTags,
                            prompts.tags.map { it.tag },
                        ),
                    )
                }
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(
                        evaluationPromptsLoading = false,
                        evaluationPromptsError = e.message ?: e.toString(),
                    )
                }
            }
        }
    }

    /**
     * Loads the displayed run's sampling prompts, once per run: its own saved config, or the sets
     * the editor wrote for it (`sample_prompts`). Called when the section appears and whenever the
     * page follows a different run; a poll does not re-read it, because only an edit changes it.
     */
    fun loadSamplePrompts(force: Boolean = false) {
        val state = _uiState.value
        val shown = displayedRun(state.runs, state.selectedRun, state.runId)
        val runId = shown?.runId ?: state.runId
        if (runId.isNullOrBlank()) return
        if (!force && (state.samplePrompts?.runId == runId || state.samplePromptsLoading)) return
        viewModelScope.launch {
            _uiState.update { it.copy(samplePromptsLoading = true) }
            try {
                val response = withContext(IoDispatcher) {
                    ipc.samplePrompts(name = shown?.outputName, runId = runId)
                }
                _uiState.update { current ->
                    // A switch to another run under the call must not install the old run's prompts.
                    val now = displayedRun(current.runs, current.selectedRun, current.runId)
                    if ((now?.runId ?: current.runId) != runId) return@update current
                    current.copy(samplePrompts = response, samplePromptsLoading = false, samplePromptsError = null)
                }
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(samplePromptsLoading = false, samplePromptsError = e.message ?: e.toString())
                }
            }
        }
    }

    fun openSamplePromptsEditor() {
        _uiState.update { it.copy(samplePromptsEditorOpen = true, samplePromptsError = null) }
    }

    fun closeSamplePromptsEditor() {
        _uiState.update { it.copy(samplePromptsEditorOpen = false) }
    }

    /**
     * Saves the edited prompt sets for the displayed run. The helper stores them whole in that
     * run's log directory, where the trainer's own sample points read them before every pass — so
     * the next checkpoint of a live run uses them too — and where evaluations resolve them.
     */
    fun saveSamplePrompts(forms: List<SampleSetForm>) {
        if (_uiState.value.samplePromptsSaving) return
        val sets = sampleSetInfos(forms)
        if (sets == null) {
            _uiState.update { it.copy(samplePromptsError = "Fix the highlighted fields first") }
            return
        }
        val state = _uiState.value
        val shown = displayedRun(state.runs, state.selectedRun, state.runId)
        _uiState.update { it.copy(samplePromptsSaving = true, samplePromptsError = null) }
        viewModelScope.launch {
            try {
                val response = withContext(IoDispatcher) {
                    ipc.setSamplePrompts(sets, name = shown?.outputName, runId = shown?.runId ?: state.runId)
                }
                _uiState.update {
                    it.copy(
                        samplePromptsSaving = false,
                        samplePrompts = response,
                        samplePromptsEditorOpen = false,
                    )
                }
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(samplePromptsSaving = false, samplePromptsError = e.message ?: e.toString())
                }
            }
        }
    }

    /** Drops this run's edited prompts, so it samples with the config it started from again. */
    fun resetSamplePrompts() {
        if (_uiState.value.samplePromptsSaving) return
        val state = _uiState.value
        val shown = displayedRun(state.runs, state.selectedRun, state.runId)
        _uiState.update { it.copy(samplePromptsSaving = true, samplePromptsError = null) }
        viewModelScope.launch {
            try {
                val response = withContext(IoDispatcher) {
                    ipc.setSamplePrompts(null, name = shown?.outputName, runId = shown?.runId ?: state.runId)
                }
                _uiState.update {
                    it.copy(samplePromptsSaving = false, samplePrompts = response)
                }
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(samplePromptsSaving = false, samplePromptsError = e.message ?: e.toString())
                }
            }
        }
    }

    /**
     * Clears one checkpoint's sample images: the run's own at its step and every pass the card
     * shows, with the job records that produced them. One at a time; api.py refuses it while the
     * GPU is busy or a generation is running.
     */
    fun clearCheckpointSamples(checkpoint: CheckpointItem) {
        if (_uiState.value.clearingSamplesPath != null) return
        _uiState.update {
            it.copy(clearingSamplesPath = checkpoint.path, clearSamplesResult = null)
        }
        viewModelScope.launch {
            try {
                val state = _uiState.value
                val shown = displayedRun(state.runs, state.selectedRun, state.runId)
                val cleared = withContext(IoDispatcher) {
                    ipc.clearCheckpointSamples(
                        checkpoint = checkpoint.path,
                        name = shown?.outputName,
                        runId = shown?.runId ?: state.runId,
                    )
                }
                _uiState.update {
                    it.copy(
                        clearingSamplesPath = null,
                        clearSamplesResult = cleared.copy(path = checkpoint.path),
                    )
                }
                fetchOnce()
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(
                        clearingSamplesPath = null,
                        clearSamplesResult = SampleClearResult(
                            path = checkpoint.path,
                            error = e.message ?: e.toString(),
                        ),
                    )
                }
            }
        }
    }

    /**
     * Deletes every unpinned checkpoint weight directory of the run on screen. Samples, pins and
     * other runs stay. One at a time; api.py refuses it while the GPU is busy or a generation runs.
     */
    fun clearUnpinnedWeights() {
        if (_uiState.value.clearingUnpinned) return
        _uiState.update { it.copy(clearingUnpinned = true, unpinnedClearResult = null) }
        viewModelScope.launch {
            try {
                val state = _uiState.value
                val shown = displayedRun(state.runs, state.selectedRun, state.runId)
                val cleared = withContext(IoDispatcher) {
                    ipc.clearUnpinnedCheckpoints(
                        name = shown?.outputName,
                        runId = shown?.runId ?: state.runId,
                    )
                }
                _uiState.update { it.copy(clearingUnpinned = false, unpinnedClearResult = cleared) }
                fetchOnce()
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(
                        clearingUnpinned = false,
                        unpinnedClearResult = UnpinnedClearResult(error = e.message ?: e.toString()),
                    )
                }
            }
        }
    }

    /** Adds or removes one prompt tag from the scorable selection. */
    fun toggleEvaluationTag(tag: String) {
        _uiState.update { state ->
            val next = if (tag in state.evaluationTagSelection) {
                state.evaluationTagSelection - tag
            } else {
                state.evaluationTagSelection + tag
            }
            state.copy(evaluationTagSelection = next)
        }
    }

    /** Back to "score every tag the prompts ask for". */
    fun clearEvaluationTags() {
        _uiState.update { it.copy(evaluationTagSelection = emptySet()) }
    }

    fun dismissEvaluation() {
        _uiState.update { it.copy(evaluationTarget = null, evaluationError = null) }
    }

    /**
     * Starts the evaluation: api.py tops the checkpoint's sample images up to the depth, tags every
     * one of them and scores the tags against each prompt, in one detached job this page follows.
     * [tags] narrows the scoring to those prompt tags; empty scores every tag the prompt asks for.
     */
    fun startEvaluation(
        depth: Int,
        threshold: Float,
        categories: List<String>,
        tags: List<String> = emptyList(),
    ) {
        val target = _uiState.value.evaluationTarget ?: return
        if (_uiState.value.isStartingEvaluation != null) return
        _uiState.update { it.copy(isStartingEvaluation = target.checkpoint.path, evaluationError = null) }
        viewModelScope.launch {
            try {
                val selected = _uiState.value.selectedRun
                val response = withContext(IoDispatcher) {
                    ipc.evaluateCheckpoint(
                        checkpoint = target.checkpoint.path,
                        depth = depth,
                        threshold = threshold,
                        categories = categories,
                        tags = tags,
                        name = selected?.outputName,
                        runId = selected?.runId ?: _uiState.value.runId,
                    )
                }
                sessionJobIds += response.job.id
                _uiState.update { state ->
                    state.copy(
                        sessionJobIds = sessionJobIds.toSet(),
                        isStartingEvaluation = null,
                        // The panel stays open on the pass it just started, so its progress and
                        // result are visible where they were asked for.
                        evaluationTarget = state.evaluationTarget?.copy(jobId = response.job.id),
                        evaluationDetailsOpen = false,
                        generatedJobs = (listOf(response.job) + state.generatedJobs).distinctBy { it.id },
                    )
                }
                startGeneratedPolling()
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(
                        isStartingEvaluation = null,
                        evaluationError = e.message ?: e.toString(),
                    )
                }
            }
        }
    }

    /** Expands or collapses the panel's per-prompt details block. */
    fun toggleEvaluationDetails() {
        _uiState.update { state -> state.copy(evaluationDetailsOpen = !state.evaluationDetailsOpen) }
    }

    /** Ask the running generation (the batch one included) to stop; its images so far stay. */
    fun cancelGeneration(id: String? = null) {
        viewModelScope.launch {
            try {
                val response = withContext(IoDispatcher) { ipc.cancelGeneration(id) }
                _uiState.update { state ->
                    state.copy(
                        batchError = null,
                        generatedJobs = (
                            listOf(response.job) + state.generatedJobs.filterNot { it.id == response.job.id }
                            ).distinctBy { it.id },
                    )
                }
                startGeneratedPolling()
            } catch (e: Exception) {
                _uiState.update { it.copy(batchError = e.message ?: e.toString()) }
            }
        }
    }

    /**
     * Pin or unpin one checkpoint of the run the page shows. Pins are the run's own state: the
     * helper writes them into `checkpoint_pins.json` inside that run's log directory, and Chromatrix
     * never touches the file itself.
     */
    fun toggleCheckpointPin(checkpoint: CheckpointItem) {
        if (_uiState.value.pinningPath != null) return
        val pinned = _uiState.value.checkpointPins.any { it.path == checkpoint.path }
        pinsWriteInFlight = true
        _uiState.update { it.copy(pinningPath = checkpoint.path, pinsError = null) }
        viewModelScope.launch {
            try {
                val shown = displayedRun(
                    _uiState.value.runs,
                    _uiState.value.selectedRun,
                    _uiState.value.runId,
                )
                val response = withContext(IoDispatcher) {
                    ipc.setCheckpointPin(
                        path = checkpoint.path,
                        pinned = !pinned,
                        dir = checkpoint.dir,
                        step = checkpoint.step,
                        name = shown?.outputName ?: _uiState.value.selectedRun?.outputName,
                        runId = shown?.runId ?: _uiState.value.selectedRun?.runId ?: _uiState.value.runId,
                    )
                }
                _uiState.update {
                    it.copy(
                        pinningPath = null,
                        checkpointPins = response.pins,
                        checkpointPinsFile = response.file,
                    )
                }
            } catch (e: Exception) {
                _uiState.update { it.copy(pinningPath = null, pinsError = e.message ?: e.toString()) }
            } finally {
                pinsWriteInFlight = false
            }
        }
    }

    private suspend fun fetchGeneratedJobs(runId: String, name: String? = null): List<GeneratedSampleJob> =
        try {
            withContext(IoDispatcher) { ipc.listGeneratedSamples(name = name, runId = runId) }.jobs
        } catch (_: Exception) {
            emptyList()
        }

    /** 1.5 s poll while a generator works: it runs in its own process and reports through its job file. */
    private fun startGeneratedPolling() {
        if (generatedPollJob?.isActive == true) return
        // Everything already finished is history: a run keeps its failed job records, so announcing
        // the newest one would repeat a failure from an earlier session on every later generation.
        val history = _uiState.value.generatedJobs
            .filter { it.state != JOB_RUNNING }
            .map { it.id }
            .toSet()
        generatedPollJob = viewModelScope.launch {
            // The failure this poll announced, so the line can come down once that job stops
            // failing — a helper that closed it before its pid was readable leaves an error the
            // generator's own record then overwrites with `done`.
            var announced: GeneratedSampleJob? = null
            while (isActive) {
                delay(GENERATED_POLL_MILLIS.milliseconds)
                val state = _uiState.value
                val selected = state.selectedRun
                val runId = selected?.runId ?: state.runId ?: return@launch
                val jobs = fetchGeneratedJobs(runId, selected?.outputName)
                if (jobs.isEmpty()) return@launch
                newlyFailedJob(jobs, history)?.let { announced = it }
                _uiState.update { current ->
                    // A switch to another run under the poll must not inject the old run's jobs.
                    if ((current.selectedRun?.runId ?: current.runId) != runId) return@update current
                    current.copy(
                        generatedJobs = jobs,
                        sampleRevisions = sampleRevisions(jobs),
                        generatedError = generatedFailureLine(announced, jobs, current.generatedError),
                    )
                }
                if (jobs.none { it.state == JOB_RUNNING }) return@launch
            }
        }
    }


    /**
     * "Save As" for one checkpoint: pick a destination in the OS save dialog, then copy the LoRA
     * off the run directory. The Ctrl+click panel and the Checkpoints section's cards share it —
     * there is one save dialog and one copy in flight, so a second request is ignored until the
     * first lands. Chromatrix only asks the helper to copy; the file bytes never pass through here.
     */
    fun saveCheckpointAs(checkpoint: CheckpointItem) {
        if (_uiState.value.exportInFlightPath != null) return
        val source = checkpoint.path
        _uiState.update {
            it.copy(exportInFlightPath = source, exportResult = null)
        }
        viewModelScope.launch {
            val parent = source.substringBeforeLast('/', missingDelimiterValue = "")
            val chosen = pathPicker.saveFile(checkpointSaveName(checkpoint), parent)
            if (chosen == null) {
                _uiState.update { it.copy(exportInFlightPath = null) }
                return@launch
            }
            val destName = ensureSafetensorsExtension(chosen.substringAfterLast('/'))
            val destParent = chosen.substringBeforeLast('/', missingDelimiterValue = "")
            val target = if (destParent.isEmpty()) destName else "$destParent/$destName"
            if (target != chosen) pathPicker.deleteEmptyPlaceholder(chosen)

            try {
                val exported = withContext(IoDispatcher) {
                    ipc.checkpointExport(source, target)
                }
                _uiState.update {
                    it.copy(
                        exportInFlightPath = null,
                        exportResult = CheckpointExport(
                            path = source,
                            savedPath = "$target (${formatBytes(exported.bytes)})",
                        ),
                    )
                }
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(
                        exportInFlightPath = null,
                        exportResult = CheckpointExport(
                            path = source,
                            error = e.message ?: e.toString(),
                        ),
                    )
                }
            }
        }
    }

    /**
     * "Send to Automation" for one checkpoint: make sure the LoRA file sits under the listening
     * ComfyUI's `models/loras` (copy it there when the name is missing), guess the character
     * trigger from the shown run's sampling prompts, then hand both to the Automation page through
     * [onOpenUniversal] — which runs only once the file is in place.
     *
     * The copy is a server-local `checkpoint_export`, which claims the destination file and needs
     * no GPU, so a send may run while training. A guess that fails sends an empty trigger; the
     * Universal section's own start check is what asks for one.
     */
    fun sendCheckpointToAutomation(
        checkpoint: CheckpointItem,
        onOpenUniversal: (CheckpointSend) -> Unit,
    ) {
        if (_uiState.value.sendInFlightPath != null) return
        val source = checkpoint.path
        _uiState.update { it.copy(sendInFlightPath = source, sendResult = null) }
        viewModelScope.launch {
            val result = try {
                withContext(IoDispatcher) { prepareCheckpointSend(checkpoint) }
            } catch (e: Exception) {
                CheckpointSendResult(path = source, error = e.message ?: e.toString())
            }
            _uiState.update { it.copy(sendInFlightPath = null, sendResult = result) }
            if (result.error == null) {
                onOpenUniversal(CheckpointSend(result.loraName, result.trigger))
            }
        }
    }

    /** The copy and the guess behind [sendCheckpointToAutomation]. */
    private suspend fun prepareCheckpointSend(checkpoint: CheckpointItem): CheckpointSendResult {
        val server = runCatching { ipc.automationConfigGet().settings.server }.getOrDefault("")
        val listed = ipc.automationLoras(server)
        if (listed.error.isNotBlank() || listed.root.isBlank()) {
            return CheckpointSendResult(
                path = checkpoint.path,
                error = listed.error.ifBlank { "ComfyUI's LoRA folder was not found" },
            )
        }
        val existing = listed.loras.firstOrNull { it.substringAfterLast('/') == checkpoint.filename }
        val name = existing ?: checkpoint.filename
        val copied = existing == null
        if (copied) {
            // `root` is the ComfyUI install root; LoraLoader resolves the name under `models/loras`.
            val loras = "${listed.root.trimEnd('/')}/models/loras"
            ipc.checkpointExport(checkpoint.path, "$loras/${checkpoint.filename}")
        }
        return CheckpointSendResult(
            path = checkpoint.path,
            loraName = name,
            trigger = guessTrigger(),
            copied = copied,
        )
    }

    /** The trigger guess, best-effort: the shown run's prompts against `selected_tags.csv`. */
    private suspend fun guessTrigger(): String {
        val state = _uiState.value
        val shown = displayedRun(state.runs, state.selectedRun, state.runId)
        val runId = shown?.runId ?: state.runId
        val sets = state.samplePrompts?.takeIf { it.runId == runId }?.sets
            ?: runId?.let {
                runCatching { ipc.samplePrompts(name = shown?.outputName, runId = it).sets }.getOrNull()
            }
            ?: return ""
        val lexicon = runCatching { TagLexicon.load(ipc.tagLexicon().text) }.getOrNull() ?: return ""
        return guessCharacterTrigger(sets, lexicon::isKnownTag) ?: ""
    }

    fun previewNext() = movePreview(1)

    fun previewPrev() = movePreview(-1)

    fun refreshNow() {
        viewModelScope.launch { fetchOnce() }
        viewModelScope.launch { fetchHardwareOnce() }
    }

    /**
     * Retunes the run in progress: the checkpoint cadence, the sampling switch, or both. The
     * trainer adopts the request at its next optimizer step — a sample pass already running
     * finishes first — and the reply carries it back as `requested`, which the card shows as the
     * value in place plus a line saying when it lands.
     */
    fun applyTrainSettings(saveEveryNSteps: Int? = null, samplingEnabled: Boolean? = null) {
        if (_uiState.value.settingsInFlight) return
        _uiState.update { it.copy(settingsInFlight = true, settingsError = null) }
        viewModelScope.launch {
            try {
                val status = withContext(IoDispatcher) {
                    ipc.trainSettings(saveEveryNSteps = saveEveryNSteps, samplingEnabled = samplingEnabled)
                }
                _uiState.update { it.copy(settingsInFlight = false, trainStatus = status) }
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(settingsInFlight = false, settingsError = e.message ?: e.toString())
                }
            }
        }
    }

    fun startTraining() = runTrainCommand { ipc.trainStart() }
    fun pauseTraining() = runTrainCommand(pending = "pause") { ipc.trainPause() }

    fun resumeTraining() = runTrainCommand(pending = "resume") { ipc.trainResume() }

    fun stopTraining() = runTrainCommand(pending = "stop") { ipc.trainStop() }

    /** Clears the run's state so Start can launch a new one. Nothing on disk is deleted. */
    fun resetTraining() = runTrainCommand(refreshAll = true) {
        ipc.trainReset()
    }

    private fun runTrainCommand(
        pending: String? = null,
        refreshAll: Boolean = false,
        block: suspend () -> TrainStatus,
    ) {
        viewModelScope.launch {
            _uiState.update { it.copy(commandInFlight = true, pendingCommand = pending ?: it.pendingCommand) }
            try {
                val status = withContext(IoDispatcher) { block() }
                _uiState.update {
                    it.copy(
                        commandInFlight = false,
                        trainStatus = status,
                        errorMessage = null,
                        pendingCommand = resolvedPending(pending ?: it.pendingCommand, status.status),
                    )
                }
                if (refreshAll) fetchOnce()
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(
                        commandInFlight = false,
                        pendingCommand = null,
                        errorMessage = e.message ?: e.toString(),
                    )
                }
            }
        }
    }

    fun retry() {
        viewModelScope.launch {
            _uiState.update { it.copy(isLoading = true, errorMessage = null) }
            try {
                ipc.restart()
                hardwareStep = 0
                _uiState.update { it.copy(hardware = HardwareStatus(), hardwareHistory = HardwareHistory()) }
                fetchOnce()
                fetchHardwareOnce()
                if (_uiState.value.autoRefresh) {
                    startPolling()
                    startHardwarePolling()
                }
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(isLoading = false, connected = false, errorMessage = e.message ?: e.toString())
                }
            }
        }
    }

    private fun startPolling() {
        pollingJob?.cancel()
        pollingJob = viewModelScope.launch {
            while (isActive) {
                fetchOnce()
                if (_uiState.value.autoRefresh) {
                    val live = _uiState.value.trainStatus.status in LIVE_TRAIN_STATUSES
                    delay(if (live) 1_000L.milliseconds else 3_000L.milliseconds)
                } else {
                    break
                }
            }
        }
    }

    private fun startHardwarePolling() {
        hardwareJob?.cancel()
        hardwareJob = viewModelScope.launch {
            while (isActive) {
                fetchHardwareOnce()
                if (_uiState.value.autoRefresh) {
                    delay(1_000L.milliseconds)
                } else {
                    break
                }
            }
        }
    }

    private suspend fun fetchHardwareOnce() {
        try {
            val snapshot = withContext(IoDispatcher) { ipc.hardwareStatus() }
            _uiState.update { state ->
                val step = hardwareStep
                hardwareStep += 1
                state.copy(
                    hardware = snapshot,
                    hardwareHistory = appendHardwareHistory(state.hardwareHistory, snapshot, step),
                )
            }
        } catch (e: Exception) {
            _uiState.update {
                it.copy(
                    hardware = it.hardware.copy(
                        available = false,
                        error = e.message ?: e.toString(),
                    ),
                )
            }
        }
    }

    private suspend fun fetchOnce() {
        val firstLoad = !_uiState.value.connected && _uiState.value.latestStats.isEmpty()
        if (firstLoad) {
            _uiState.update { it.copy(isLoading = true, errorMessage = null) }
        }
        try {
            val pinned = _uiState.value.selectedRun
            val pending = _uiState.value.pendingRunId
            val runs = withContext(IoDispatcher) { ipc.listRuns() }.runs
            val selected = resolvePinnedRun(runs, pinned, pending)
            val dashboard = withContext(IoDispatcher) {
                ipc.getDashboard(name = selected?.outputName, runId = selected?.runId)
            }
            val samples = withContext(IoDispatcher) {
                ipc.listSamples(name = selected?.outputName, runId = selected?.runId)
            }
            // The Checkpoints section follows the run the page shows, so it needs that run's own
            // name even when the trainer is on a run created under a different `output_name`.
            // Neither list may take the dashboard down: a broken checkpoint file or a job record
            // the generator is rewriting leaves the charts and controls alone.
            val shown = displayedRun(runs, selected, dashboard.runId)
            val shownName = shown?.outputName ?: selected?.outputName
            val checkpoints = runCatching {
                withContext(IoDispatcher) { ipc.listCheckpoints(name = shownName) }.checkpoints
            }.getOrDefault(emptyList())
            val generatedJobs = runCatching {
                withContext(IoDispatcher) {
                    ipc.listGeneratedSamples(name = shownName, runId = shown?.runId)
                }.jobs
            }.getOrDefault(emptyList())
            // The pins live in the shown run's own log directory, so they follow the same run the
            // checkpoints do. A read is skipped while a pin write is in flight: the write's own
            // reply is the newer truth, and a read that started first would put the old list back.
            val pins = if (pinsWriteInFlight) {
                null
            } else {
                runCatching {
                    withContext(IoDispatcher) {
                        ipc.checkpointPins(name = shownName, runId = shown?.runId)
                    }
                }.getOrNull()
            }
            val trainStatus = withContext(IoDispatcher) { ipc.trainStatus() }
            if (_uiState.value.runId != null && _uiState.value.runId != dashboard.runId) {
                chartInteractions.resetAll()
            }
            _uiState.update { state ->
                val generated = generatedJobs
                // The preview list is the section's own images in the section's own order, so the
                // path a click opened is looked up in the very rows the page drew.
                val shownCheckpoints = if (shown == null) {
                    emptyList()
                } else {
                    checkpointsForRun(checkpoints, shown.runId)
                }
                val previewPath = state.previewIndex?.let { index ->
                    previewList(state.checkpoints, state.samples, state.generatedJobs).getOrNull(index)?.path
                }
                val newList = previewList(shownCheckpoints, samples.samples, generated)
                val newPreview = previewPath?.let { path ->
                    newList.indexOfFirst { it.path == path }.takeIf { it >= 0 }
                }
                state.copy(
                    isLoading = false,
                    errorMessage = null,
                    connected = true,
                    config = dashboard.config,
                    runId = dashboard.runId,
                    runs = runs,
                    selectedRun = selected,
                    // Only the id this fetch consumed: a click that landed while it was in flight
                    // has its own, newer id waiting for the refresh that click started.
                    pendingRunId = if (state.pendingRunId == pending) null else state.pendingRunId,
                    latestStats = dashboard.latestStats,
                    metrics = dashboard.metrics,
                    stepsPerEpoch = dashboard.stepsPerEpoch,
                    runSaveEveryNSteps = dashboard.saveEveryNSteps,
                    samples = samples.samples,
                    checkpoints = shownCheckpoints,
                    checkpointPins = pins?.pins ?: state.checkpointPins,
                    checkpointPinsFile = pins?.file ?: state.checkpointPinsFile,
                    generatedJobs = generated,
                    // A replaced sample keeps its old bytes in the client cache: the id of the job
                    // that rewrote it is what makes the thumbnail fetch the picture again.
                    sampleRevisions = sampleRevisions(generated),
                    previewIndex = newPreview,
                    trainStatus = trainStatus,
                    pendingCommand = resolvedPending(state.pendingCommand, trainStatus.status),
                )
            }
        } catch (e: Exception) {
            _uiState.update {
                it.copy(
                    isLoading = false,
                    connected = false,
                    errorMessage = e.message ?: e.toString(),
                )
            }
        }
    }

    private fun movePreview(delta: Int) {
        val list = currentPreviewList()
        if (list.isEmpty()) {
            closePreview()
            return
        }
        val current = _uiState.value.previewIndex ?: return
        val next = (current + delta).mod(list.size)
        _uiState.update { it.copy(previewIndex = next) }
    }

    /** What the fullscreen preview cycles through: every image the Checkpoints section shows. */
    private fun currentPreviewList(state: DashboardUiState = _uiState.value): List<SampleItem> =
        previewList(state.checkpoints, state.samples, state.generatedJobs)
}

/**
 * The run the page shows, against the run history just read: the pin, re-read from the list so its
 * badge and figures follow it — or, when a click arrived before the page had a list at all, the id
 * that click carried, looked up in the list. A pin whose run left the list keeps the entry the page
 * picked; an id that is not in the list resolves to nothing, which follows the live run again.
 */
internal fun resolvePinnedRun(
    runs: List<RunSummary>,
    pinned: RunSummary?,
    pendingRunId: String?,
): RunSummary? =
    pinned?.let { run -> runs.firstOrNull { it.runId == run.runId } ?: run }
        ?: pendingRunId?.let { id -> runs.firstOrNull { it.runId == id } }

/**
 * Start / Pause / Resume / Early Stop / Reset act on the run `state.json` is on. A run the
 * user pinned in the history list is a past run: the page shows it, the controls stay off.
 */
internal fun trainingControlsEnabled(state: DashboardUiState): Boolean {
    val pinned = state.selectedRun ?: return true
    return pinned.runId == state.trainStatus.runId
}

/**
 * The live cadence / sampling controls act on the run the trainer is on, so they follow the same
 * rule as the Start/Pause buttons and additionally need a run that is actually live.
 */
internal fun liveSettingsEnabled(state: DashboardUiState): Boolean =
    trainingControlsEnabled(state) && state.trainStatus.status in LIVE_TRAIN_STATUSES

/**
 * Whether a one-off generation may start: only while no live trainer is using the GPU. A paused
 * run is fine (pause has offloaded every module); api.py enforces the same rule.
 */
internal fun generationAllowed(status: TrainStatus): Boolean =
    !status.alive || status.status !in GPU_BUSY_STATUSES

/** Statuses in which the trainer holds the GPU (a paused one has given it back). */
internal val GPU_BUSY_STATUSES = setOf(
    "starting",
    "encoding",
    "training",
    "sampling",
    "pausing",
    "resuming",
    "stopping",
)

internal fun resolvedPending(pending: String?, status: String): String? {
    if (status in setOf("idle", "finished", "error")) return null
    return when (pending) {
        "pause" -> if (status == "pausing" || status == "paused") null else pending
        "resume" -> if (status in setOf("resuming", "encoding", "training", "sampling")) null else pending
        "stop" -> if (status == "stopping") null else pending
        else -> pending
    }
}

internal val LIVE_TRAIN_STATUSES = setOf(
    "starting",
    "encoding",
    "training",
    "sampling",
    "pausing",
    "paused",
    "resuming",
    "stopping",
)

/**
 * The images a page state shows, in the order the Checkpoints section shows them. `openPreview`
 * resolves a clicked thumbnail in this same list, so anything the section draws can be opened.
 */
internal fun previewList(
    checkpoints: List<CheckpointItem>,
    samples: Map<String, List<SampleItem>>,
    jobs: List<GeneratedSampleJob>,
): List<SampleItem> = sectionImages(checkpointRows(checkpoints, samples, jobs))

private const val HARDWARE_HISTORY_CAP = 360
private const val BYTES_PER_GIB = 1024.0 * 1024.0 * 1024.0
private const val GENERATED_POLL_MILLIS = 1_500L

/** How many polls a finished pinned round may stay missing from the list before the sequence stops. */
private const val GENERATED_JOB_MISS_LIMIT = 40

internal fun appendHardwareHistory(
    history: HardwareHistory,
    snapshot: HardwareStatus,
    step: Int,
): HardwareHistory {
    val gpu = snapshot.gpus.firstOrNull()
    val cpu = snapshot.cpu
    val vramGiB = gpu?.memUsedBytes?.let { it.toDouble() / BYTES_PER_GIB }
    val ramGiB = cpu.memUsedBytes?.let { it.toDouble() / BYTES_PER_GIB }
    return HardwareHistory(
        gpuUtil = appendHardwarePoint(history.gpuUtil, step, gpu?.gpuUtilPct),
        vramGiB = appendHardwarePoint(history.vramGiB, step, vramGiB),
        powerW = appendHardwarePoint(history.powerW, step, gpu?.powerW),
        tempEdge = appendHardwarePoint(history.tempEdge, step, gpu?.tempEdgeC ?: gpu?.tempC),
        tempJunction = appendHardwarePoint(history.tempJunction, step, gpu?.tempJunctionC),
        cpuUtil = appendHardwarePoint(history.cpuUtil, step, cpu.utilPct),
        cpuTemp = appendHardwarePoint(history.cpuTemp, step, cpu.tempC),
        ramGiB = appendHardwarePoint(history.ramGiB, step, ramGiB),
    )
}

private fun appendHardwarePoint(points: List<MetricPoint>, step: Int, value: Double?): List<MetricPoint> {
    if (value == null || !value.isFinite()) return points
    return (points + MetricPoint(step = step, value = value.toFloat())).takeLast(HARDWARE_HISTORY_CAP)
}
