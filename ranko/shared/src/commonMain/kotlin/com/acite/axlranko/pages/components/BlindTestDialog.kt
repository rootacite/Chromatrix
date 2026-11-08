package com.acite.axlranko.pages.components

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.focusable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsHoveredAsState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.key.Key
import androidx.compose.ui.input.key.KeyEventType
import androidx.compose.ui.input.key.key
import androidx.compose.ui.input.key.onPreviewKeyEvent
import androidx.compose.ui.input.key.type
import androidx.compose.ui.input.pointer.pointerHoverIcon
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil3.compose.AsyncImage
import com.acite.axlranko.data.BlobRef
import com.acite.axlranko.data.BlindTestState
import com.acite.axlranko.data.LocalThumbnailQuality
import com.acite.axlranko.model.CheckpointItem
import com.acite.axlranko.model.GeneratedSampleJob
import com.acite.axlranko.ui.components.CapsuleButton
import com.acite.axlranko.ui.components.PorcelainCard
import com.acite.axlranko.ui.pointerIconHand
import com.acite.axlranko.ui.theme.rankoColors
import kotlinx.coroutines.delay
import kotlin.math.roundToInt

private const val COUNTDOWN_TOTAL_MS = 6000L

@Composable
internal fun BlindTestDialog(
    pinnedCheckpoints: List<CheckpointItem>,
    jobs: List<GeneratedSampleJob>,
    state: BlindTestState?,
    loading: Boolean,
    error: String?,
    maxWidth: Dp = Dp.Unspecified,
    maxHeight: Dp = Dp.Unspecified,
    onChoice: (leftCp: String, rightCp: String, chosen: String, profileIndex: Int, isCanary: Boolean) -> Unit,
    onReset: () -> Unit,
    onDismiss: () -> Unit,
) {
    val colors = rankoColors
    val pool = remember(pinnedCheckpoints, jobs) { resolveBlindTestPool(pinnedCheckpoints, jobs) }
    val summary = remember(pinnedCheckpoints, pool) { summarizeBlindTestPool(pinnedCheckpoints, pool) }
    val leaderboard = remember(pinnedCheckpoints, state) { computeLeaderboard(pinnedCheckpoints, state) }
    val topConfidence = remember(leaderboard) { computeTopConfidence(leaderboard) }

    var currentPair by remember { mutableStateOf<BlindTestPair?>(null) }
    var roundNumber by remember { mutableStateOf(1) }
    var confirmReset by remember { mutableStateOf(false) }
    var showStandings by remember { mutableStateOf(false) }

    fun pickNextPair() {
        currentPair = sampleBlindTestPair(pinnedCheckpoints, pool, state)
        roundNumber++
    }

    LaunchedEffect(summary.eligibleProfilesCount, pinnedCheckpoints) {
        if (currentPair == null && summary.eligibleProfilesCount > 0) {
            currentPair = sampleBlindTestPair(pinnedCheckpoints, pool, state)
        }
    }

    // 6-second countdown timer for each pair
    var timerRemainingMs by remember(currentPair) { mutableStateOf(COUNTDOWN_TOTAL_MS) }
    LaunchedEffect(currentPair) {
        timerRemainingMs = COUNTDOWN_TOTAL_MS
        val stepMs = 50L
        while (timerRemainingMs > 0L) {
            delay(stepMs)
            timerRemainingMs = (timerRemainingMs - stepMs).coerceAtLeast(0L)
        }
    }

    val focusRequester = remember { FocusRequester() }
    LaunchedEffect(roundNumber, showStandings) {
        focusRequester.requestFocus()
    }

    // Full-screen edge-to-edge overlay
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(colors.bgApp.copy(alpha = 0.97f))
            .focusRequester(focusRequester)
            .focusable()
            .onPreviewKeyEvent { event ->
                if (event.type != KeyEventType.KeyDown) return@onPreviewKeyEvent false
                val pair = currentPair
                when (event.key) {
                    Key.Escape -> {
                        if (showStandings) {
                            showStandings = false
                        } else {
                            onDismiss()
                        }
                        true
                    }
                    Key.S, Key.Tab -> {
                        showStandings = !showStandings
                        true
                    }
                    Key.A, Key.DirectionLeft -> {
                        if (!showStandings && pair != null) {
                            onChoice(pair.leftCheckpoint, pair.rightCheckpoint, "left", pair.profileIndex, pair.isCanary)
                            pickNextPair()
                            true
                        } else false
                    }
                    Key.D, Key.DirectionRight -> {
                        if (!showStandings && pair != null) {
                            onChoice(pair.leftCheckpoint, pair.rightCheckpoint, "right", pair.profileIndex, pair.isCanary)
                            pickNextPair()
                            true
                        } else false
                    }
                    Key.Spacebar -> {
                        if (!showStandings && pair != null) {
                            onChoice(pair.leftCheckpoint, pair.rightCheckpoint, "skip", pair.profileIndex, pair.isCanary)
                            pickNextPair()
                            true
                        } else false
                    }
                    else -> false
                }
            }
            .clickable(
                interactionSource = remember { MutableInteractionSource() },
                indication = null,
                onClick = {} // Absorb clicks
            ),
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 20.dp, vertical = 14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            // 1. Sleek Single-Line Header Bar
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(colors.bgCard, shape = RoundedCornerShape(10.dp))
                    .border(1.dp, colors.stroke, shape = RoundedCornerShape(10.dp))
                    .padding(horizontal = 16.dp, vertical = 10.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                // Left: Title and Active Pool Badge
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    Text(
                        text = "Blind Test: Pinned Checkpoints",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                        color = colors.text,
                        maxLines = 1,
                    )
                    Surface(
                        color = colors.bgPanel,
                        shape = CircleShape,
                    ) {
                        Text(
                            text = "${summary.eligibleCheckpointsCount} active / ${summary.pinnedCheckpointsCount} pinned",
                            modifier = Modifier.padding(horizontal = 10.dp, vertical = 3.dp),
                            style = MaterialTheme.typography.labelSmall,
                            color = colors.textDim,
                            maxLines = 1,
                        )
                    }
                    currentPair?.let { pair ->
                        Surface(
                            color = colors.accentBlue.copy(alpha = 0.15f),
                            shape = RoundedCornerShape(6.dp),
                        ) {
                            Text(
                                text = "Profile Set: #${pair.profileIndex + 1}",
                                modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp),
                                style = MaterialTheme.typography.labelSmall,
                                color = colors.accentBlue,
                                fontWeight = FontWeight.SemiBold,
                                maxLines = 1,
                            )
                        }
                    }
                }

                // Center: Metrics Badges
                val metrics = state?.metrics
                val overallConsistency = metrics?.overallConsistency ?: 1.0
                val crossProfileConsistency = metrics?.crossProfileConsistency ?: 1.0
                val canaryConsistency = metrics?.canaryConsistency ?: 1.0
                val totalComparisons = metrics?.totalComparisons ?: 0

                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    // Consistency Badge
                    Surface(
                        color = if (overallConsistency >= 0.75) colors.bgPanel else colors.qualityYellow.copy(alpha = 0.15f),
                        shape = RoundedCornerShape(8.dp),
                    ) {
                        Row(
                            modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(6.dp),
                        ) {
                            Text(
                                text = "User Consistency: ${(overallConsistency * 100).roundToInt()}%",
                                style = MaterialTheme.typography.labelSmall,
                                fontWeight = FontWeight.SemiBold,
                                color = if (overallConsistency >= 0.75) colors.qualityGreen else colors.qualityYellow,
                                maxLines = 1,
                                softWrap = false,
                            )
                            Text(
                                text = "(Cross-Profile: ${(crossProfileConsistency * 100).roundToInt()}% · Canary: ${(canaryConsistency * 100).roundToInt()}%)",
                                style = MaterialTheme.typography.labelSmall,
                                color = colors.textDim,
                                maxLines = 1,
                                softWrap = false,
                            )
                        }
                    }

                    // Top-1 Dominance Badge
                    if (topConfidence != null) {
                        val confPercent = (topConfidence.confidence * 100).roundToInt()
                        Surface(
                            color = if (topConfidence.isConfirmed) colors.qualityGreen.copy(alpha = 0.15f) else colors.qualityYellow.copy(alpha = 0.15f),
                            shape = RoundedCornerShape(8.dp),
                        ) {
                            Text(
                                text = if (topConfidence.isConfirmed) "Top-1 Confirmed: $confPercent%" else "Top-1 Contested: $confPercent%",
                                modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp),
                                style = MaterialTheme.typography.labelSmall,
                                fontWeight = FontWeight.SemiBold,
                                color = if (topConfidence.isConfirmed) colors.qualityGreen else colors.qualityYellow,
                                maxLines = 1,
                                softWrap = false,
                            )
                        }
                    }

                    Text(
                        text = "Total Matches: $totalComparisons",
                        style = MaterialTheme.typography.labelSmall,
                        color = colors.textDim,
                        maxLines = 1,
                        softWrap = false,
                    )
                }

                // Right: Action buttons (Reset & Close)
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    CapsuleButton(
                        text = if (confirmReset) "Confirm Reset?" else "Reset Evaluation",
                        onClick = {
                            if (!confirmReset) {
                                confirmReset = true
                            } else {
                                confirmReset = false
                                onReset()
                                currentPair = null
                            }
                        },
                        compact = true,
                    )
                    IconButton(onClick = onDismiss, modifier = Modifier.size(30.dp)) {
                        Icon(
                            imageVector = Icons.Default.Close,
                            contentDescription = "Close",
                            tint = colors.textDim,
                            modifier = Modifier.size(18.dp),
                        )
                    }
                }
            }

            // 2. Imbalance Warning (if any)
            if (summary.isImbalanced && summary.warningMessage != null) {
                Surface(
                    color = colors.qualityYellow.copy(alpha = 0.12f),
                    shape = RoundedCornerShape(8.dp),
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Row(
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        Icon(
                            imageVector = Icons.Default.Warning,
                            contentDescription = null,
                            tint = colors.qualityYellow,
                            modifier = Modifier.size(16.dp),
                        )
                        Text(
                            text = summary.warningMessage,
                            style = MaterialTheme.typography.bodySmall,
                            color = colors.qualityYellow,
                            maxLines = 1,
                        )
                    }
                }
            }

            // 3. 6s Countdown Progress & Gut Reminder Bar
            val progress = timerRemainingMs.toFloat() / COUNTDOWN_TOTAL_MS.toFloat()
            val isExpired = timerRemainingMs == 0L

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 4.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(14.dp),
            ) {
                LinearProgressIndicator(
                    progress = { progress },
                    modifier = Modifier
                        .weight(1f)
                        .height(4.dp)
                        .clip(RoundedCornerShape(2.dp)),
                    color = if (isExpired) colors.qualityYellow else colors.accentBlue,
                    trackColor = colors.stroke,
                )

                if (isExpired) {
                    Text(
                        text = "⏱️ Time's up! Trust your initial gut impression rather than overanalyzing fine details.",
                        style = MaterialTheme.typography.labelSmall,
                        color = colors.qualityYellow,
                        fontWeight = FontWeight.Medium,
                        maxLines = 1,
                        softWrap = false,
                    )
                } else {
                    Text(
                        text = "⏱️ Quick Triage Timer: ${(timerRemainingMs / 1000f).coerceAtLeast(0f).toString().take(3)}s",
                        style = MaterialTheme.typography.labelSmall,
                        color = colors.textDim,
                        maxLines = 1,
                        softWrap = false,
                    )
                }
            }

            // 4. Hero Visual Comparison Area (occupies all remaining vertical space!)
            val pair = currentPair
            if (summary.eligibleCheckpointsCount < 2 || summary.eligibleProfilesCount == 0) {
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth()
                        .background(colors.bgCard, shape = RoundedCornerShape(12.dp))
                        .border(1.dp, colors.stroke, shape = RoundedCornerShape(12.dp)),
                    contentAlignment = Alignment.Center,
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(10.dp),
                    ) {
                        Text(
                            text = "No eligible pinned checkpoint pairs found",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold,
                            color = colors.textDim,
                        )
                        Text(
                            text = "Blind Test requires at least 2 pinned checkpoints with samples generated via 'Generate pinned samples'.",
                            style = MaterialTheme.typography.bodyMedium,
                            color = colors.textDim,
                            textAlign = TextAlign.Center,
                            modifier = Modifier.padding(horizontal = 32.dp),
                        )
                    }
                }
            } else if (pair == null) {
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth(),
                    contentAlignment = Alignment.Center,
                ) {
                    CircularProgressIndicator(color = colors.accentBlue)
                }
            } else {
                Row(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(14.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    // Left Image Hero Card
                    BlindHeroImageCard(
                        badgeLabel = "Image A",
                        buttonLabel = "👈 Left is Better [A / ←]",
                        shortcutHint = "Click image or press [A / ←]",
                        imagePath = pair.leftImage.path,
                        modifier = Modifier
                            .weight(1f)
                            .fillMaxHeight(),
                        onSelect = {
                            onChoice(pair.leftCheckpoint, pair.rightCheckpoint, "left", pair.profileIndex, pair.isCanary)
                            pickNextPair()
                        },
                    )

                    // Center VS & Skip Control
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.Center,
                        modifier = Modifier.width(90.dp),
                    ) {
                        Surface(
                            color = colors.bgCard,
                            shape = CircleShape,
                            border = androidx.compose.foundation.BorderStroke(1.dp, colors.stroke),
                            modifier = Modifier.size(46.dp),
                        ) {
                            Box(contentAlignment = Alignment.Center) {
                                Text(
                                    text = "VS",
                                    style = MaterialTheme.typography.titleMedium,
                                    fontWeight = FontWeight.Black,
                                    color = colors.textDim,
                                )
                            }
                        }
                        Spacer(Modifier.height(18.dp))
                        CapsuleButton(
                            text = "Skip [Space]",
                            onClick = {
                                onChoice(pair.leftCheckpoint, pair.rightCheckpoint, "skip", pair.profileIndex, pair.isCanary)
                                pickNextPair()
                            },
                            compact = true,
                        )
                    }

                    // Right Image Hero Card
                    BlindHeroImageCard(
                        badgeLabel = "Image B",
                        buttonLabel = "👉 Right is Better [D / →]",
                        shortcutHint = "Click image or press [D / →]",
                        imagePath = pair.rightImage.path,
                        modifier = Modifier
                            .weight(1f)
                            .fillMaxHeight(),
                        onSelect = {
                            onChoice(pair.leftCheckpoint, pair.rightCheckpoint, "right", pair.profileIndex, pair.isCanary)
                            pickNextPair()
                        },
                    )
                }
            }

            // 5. Sleek Bottom Bar (Leaderboard Summary & Expand Toggle)
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(44.dp)
                    .background(colors.bgCard, shape = RoundedCornerShape(10.dp))
                    .border(1.dp, colors.stroke, shape = RoundedCornerShape(10.dp))
                    .padding(horizontal = 16.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                val leader = leaderboard.firstOrNull()
                if (leader != null && leader.matches > 0) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        Text(
                            text = "🏆 Current Leader:",
                            style = MaterialTheme.typography.labelMedium,
                            fontWeight = FontWeight.SemiBold,
                            color = colors.qualityGreen,
                            maxLines = 1,
                        )
                        Text(
                            text = if (leader.step != null) "Step ${leader.step}" else leader.checkpointPath.substringAfterLast('/'),
                            style = MaterialTheme.typography.labelMedium,
                            fontWeight = FontWeight.Bold,
                            color = colors.text,
                            maxLines = 1,
                        )
                        Text(
                            text = "(Elo ${leader.rating.roundToInt()} ± ${leader.uncertainty.roundToInt()} · ${(leader.winRate * 100).roundToInt()}% Win Rate · ${leader.matches} Matches)",
                            style = MaterialTheme.typography.labelSmall,
                            color = colors.textDim,
                            maxLines = 1,
                        )
                    }
                } else {
                    Text(
                        text = "Pair comparison in progress · Click preferred image or press [A] / [D]",
                        style = MaterialTheme.typography.labelSmall,
                        color = colors.textDim,
                        maxLines = 1,
                    )
                }

                CapsuleButton(
                    text = if (showStandings) "Hide Standings [S]" else "📊 View Standings (${leaderboard.size}) [S]",
                    onClick = { showStandings = !showStandings },
                    compact = true,
                )
            }
        }

        // 6. Floating Standings Overlay (when toggled via button or S key)
        if (showStandings) {
            val maxRating = remember(leaderboard) { leaderboard.maxOfOrNull { it.rating } ?: 1500.0 }
            val minRating = remember(leaderboard) { leaderboard.minOfOrNull { it.rating } ?: 1500.0 }
            val quality = LocalThumbnailQuality.current

            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color.Black.copy(alpha = 0.7f))
                    .clickable(
                        interactionSource = remember { MutableInteractionSource() },
                        indication = null,
                        onClick = { showStandings = false }
                    ),
                contentAlignment = Alignment.Center,
            ) {
                PorcelainCard {
                    Column(
                        modifier = Modifier
                            .widthIn(min = 960.dp, max = 1240.dp)
                            .fillMaxWidth(0.92f)
                            .heightIn(max = 680.dp)
                            .clickable(
                                interactionSource = remember { MutableInteractionSource() },
                                indication = null,
                                onClick = {} // absorb click inside card
                            )
                            .padding(22.dp),
                        verticalArrangement = Arrangement.spacedBy(14.dp),
                    ) {
                        // 1. Standings Header
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(12.dp),
                            ) {
                                Text(
                                    text = "Checkpoint Standings (Bayesian Elo)",
                                    style = MaterialTheme.typography.titleMedium,
                                    fontWeight = FontWeight.Bold,
                                    color = colors.text,
                                )
                                Surface(
                                    color = colors.bgPanel,
                                    shape = CircleShape,
                                ) {
                                    Text(
                                        text = "${leaderboard.size} Checkpoints Evaluated",
                                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 3.dp),
                                        style = MaterialTheme.typography.labelSmall,
                                        color = colors.textDim,
                                        fontWeight = FontWeight.SemiBold,
                                    )
                                }
                                if (topConfidence != null) {
                                    val confPercent = (topConfidence.confidence * 100).roundToInt()
                                    Surface(
                                        color = if (topConfidence.isConfirmed) colors.qualityGreen.copy(alpha = 0.15f) else colors.qualityYellow.copy(alpha = 0.15f),
                                        shape = RoundedCornerShape(6.dp),
                                    ) {
                                        Text(
                                            text = if (topConfidence.isConfirmed) "Top-1 Confirmed: $confPercent%" else "Top-1 Contested: $confPercent%",
                                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp),
                                            style = MaterialTheme.typography.labelSmall,
                                            fontWeight = FontWeight.SemiBold,
                                            color = if (topConfidence.isConfirmed) colors.qualityGreen else colors.qualityYellow,
                                        )
                                    }
                                }
                            }

                            IconButton(onClick = { showStandings = false }, modifier = Modifier.size(30.dp)) {
                                Icon(
                                    imageVector = Icons.Default.Close,
                                    contentDescription = "Close Standings",
                                    tint = colors.textDim,
                                    modifier = Modifier.size(20.dp),
                                )
                            }
                        }

                        // 2. Scrollable Checkpoint Cards List
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .weight(1f, fill = false)
                                .verticalScroll(rememberScrollState()),
                            verticalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            leaderboard.forEach { row ->
                                val isLeader = row.rank == 1 && row.matches > 0
                                val cpSamples = pool[row.checkpointPath]?.values?.flatten().orEmpty()
                                val totalSamples = cpSamples.size
                                val distinctProfiles = pool[row.checkpointPath]?.keys?.size ?: 0

                                val eloColor = when {
                                    row.rating >= 1520.0 -> colors.qualityGreen
                                    row.rating >= 1500.0 -> colors.accentBlue
                                    row.rating >= 1485.0 -> colors.text
                                    else -> colors.qualityYellow
                                }
                                val ratingSpan = (maxRating - minRating).coerceAtLeast(1.0)
                                val ratingFraction = ((row.rating - minRating) / ratingSpan).toFloat().coerceIn(0.12f, 1.0f)
                                val winPercent = (row.winRate * 100).roundToInt()

                                Surface(
                                    color = if (isLeader) colors.qualityGreen.copy(alpha = 0.08f) else colors.bgPanel,
                                    shape = RoundedCornerShape(10.dp),
                                    border = BorderStroke(
                                        width = if (isLeader) 1.5.dp else 1.dp,
                                        color = if (isLeader) colors.qualityGreen.copy(alpha = 0.6f) else colors.stroke,
                                    ),
                                    modifier = Modifier.fillMaxWidth(),
                                ) {
                                    Row(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(horizontal = 14.dp, vertical = 10.dp),
                                        verticalAlignment = Alignment.CenterVertically,
                                        horizontalArrangement = Arrangement.SpaceBetween,
                                    ) {
                                        // Rank & Checkpoint Name & Sample Count
                                        Row(
                                            verticalAlignment = Alignment.CenterVertically,
                                            horizontalArrangement = Arrangement.spacedBy(12.dp),
                                            modifier = Modifier.widthIn(min = 200.dp, max = 260.dp),
                                        ) {
                                            Surface(
                                                color = when (row.rank) {
                                                    1 -> colors.qualityGreen.copy(alpha = 0.2f)
                                                    2 -> Color(0xFF64B5F6).copy(alpha = 0.2f)
                                                    3 -> Color(0xFFFFB74D).copy(alpha = 0.2f)
                                                    else -> colors.bgCard
                                                },
                                                shape = RoundedCornerShape(8.dp),
                                                modifier = Modifier.size(width = 44.dp, height = 36.dp),
                                            ) {
                                                Box(contentAlignment = Alignment.Center) {
                                                    Text(
                                                        text = when (row.rank) {
                                                            1 -> "🏆 #1"
                                                            2 -> "🥈 #2"
                                                            3 -> "🥉 #3"
                                                            else -> "#${row.rank}"
                                                        },
                                                        style = MaterialTheme.typography.labelMedium,
                                                        fontWeight = if (row.rank <= 3) FontWeight.Bold else FontWeight.SemiBold,
                                                        color = when (row.rank) {
                                                            1 -> colors.qualityGreen
                                                            2 -> Color(0xFF90CAF9)
                                                            3 -> Color(0xFFFFCC80)
                                                            else -> colors.textDim
                                                        },
                                                    )
                                                }
                                            }

                                            Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                                                Text(
                                                    text = if (row.step != null) "Step ${row.step}" else row.checkpointPath.substringAfterLast('/'),
                                                    style = MaterialTheme.typography.titleSmall,
                                                    fontWeight = FontWeight.Bold,
                                                    color = if (isLeader) colors.qualityGreen else colors.text,
                                                    maxLines = 1,
                                                    overflow = TextOverflow.Ellipsis,
                                                )
                                                Surface(
                                                    color = colors.bgCard,
                                                    shape = RoundedCornerShape(4.dp),
                                                ) {
                                                    Text(
                                                        text = if (totalSamples == 1) "1 sample" else "$totalSamples samples ($distinctProfiles sets)",
                                                        modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp),
                                                        style = MaterialTheme.typography.labelSmall,
                                                        color = colors.textDim,
                                                        maxLines = 1,
                                                    )
                                                }
                                            }
                                        }

                                        // Thumbnails Strip
                                        Row(
                                            verticalAlignment = Alignment.CenterVertically,
                                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                                            modifier = Modifier
                                                .weight(1f)
                                                .padding(horizontal = 14.dp),
                                        ) {
                                            if (cpSamples.isEmpty()) {
                                                Text(
                                                    text = "No samples generated",
                                                    style = MaterialTheme.typography.labelSmall,
                                                    color = colors.textDim,
                                                )
                                            } else {
                                                cpSamples.take(5).forEach { sample ->
                                                    Box(
                                                        modifier = Modifier
                                                            .size(46.dp)
                                                            .clip(RoundedCornerShape(6.dp))
                                                            .border(1.dp, colors.stroke, RoundedCornerShape(6.dp))
                                                            .background(Color.Black.copy(alpha = 0.4f)),
                                                        contentAlignment = Alignment.Center,
                                                    ) {
                                                        AsyncImage(
                                                            model = BlobRef(sample.path, maxEdge = 128, quality = quality),
                                                            contentDescription = "Profile #${sample.setIndex + 1}",
                                                            contentScale = ContentScale.Crop,
                                                            modifier = Modifier.fillMaxSize(),
                                                        )
                                                        Surface(
                                                            color = Color.Black.copy(alpha = 0.65f),
                                                            shape = RoundedCornerShape(topStart = 4.dp),
                                                            modifier = Modifier.align(Alignment.BottomEnd),
                                                        ) {
                                                            Text(
                                                                text = "#${sample.setIndex + 1}",
                                                                modifier = Modifier.padding(horizontal = 3.dp, vertical = 1.dp),
                                                                style = MaterialTheme.typography.labelSmall.copy(fontSize = 9.sp),
                                                                color = Color.White,
                                                                fontWeight = FontWeight.Bold,
                                                            )
                                                        }
                                                    }
                                                }
                                                if (cpSamples.size > 5) {
                                                    Surface(
                                                        color = colors.bgCard,
                                                        shape = CircleShape,
                                                        modifier = Modifier.size(28.dp),
                                                    ) {
                                                        Box(contentAlignment = Alignment.Center) {
                                                            Text(
                                                                text = "+${cpSamples.size - 5}",
                                                                style = MaterialTheme.typography.labelSmall,
                                                                color = colors.textDim,
                                                                fontWeight = FontWeight.Bold,
                                                            )
                                                        }
                                                    }
                                                }
                                            }
                                        }

                                        // Elo Rating & Visual Progress Gauge
                                        Column(
                                            horizontalAlignment = Alignment.End,
                                            verticalArrangement = Arrangement.spacedBy(4.dp),
                                            modifier = Modifier.width(170.dp).padding(horizontal = 8.dp),
                                        ) {
                                            Row(
                                                verticalAlignment = Alignment.CenterVertically,
                                                horizontalArrangement = Arrangement.spacedBy(6.dp),
                                            ) {
                                                Text(
                                                    text = "${row.rating.roundToInt()}",
                                                    style = MaterialTheme.typography.titleSmall,
                                                    fontWeight = FontWeight.Bold,
                                                    color = eloColor,
                                                )
                                                Text(
                                                    text = "± ${row.uncertainty.roundToInt()} σ",
                                                    style = MaterialTheme.typography.labelSmall,
                                                    color = colors.textDim,
                                                )
                                            }

                                            LinearProgressIndicator(
                                                progress = { ratingFraction },
                                                modifier = Modifier
                                                    .width(130.dp)
                                                    .height(5.dp)
                                                    .clip(RoundedCornerShape(3.dp)),
                                                color = eloColor,
                                                trackColor = colors.bgCard,
                                            )
                                        }

                                        // Win Rate & Matches
                                        Column(
                                            horizontalAlignment = Alignment.End,
                                            verticalArrangement = Arrangement.spacedBy(4.dp),
                                            modifier = Modifier.width(150.dp),
                                        ) {
                                            Surface(
                                                color = if (winPercent >= 60) colors.qualityGreen.copy(alpha = 0.15f) else colors.bgCard,
                                                shape = RoundedCornerShape(6.dp),
                                            ) {
                                                Text(
                                                    text = "$winPercent% (${row.wins}W / ${row.losses}L)",
                                                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.dp),
                                                    style = MaterialTheme.typography.labelSmall,
                                                    fontWeight = FontWeight.SemiBold,
                                                    color = if (winPercent >= 60) colors.qualityGreen else colors.text,
                                                    maxLines = 1,
                                                )
                                            }

                                            Text(
                                                text = "${row.matches} match${if (row.matches == 1) "" else "es"}",
                                                style = MaterialTheme.typography.labelSmall,
                                                color = colors.textDim,
                                                maxLines = 1,
                                            )
                                        }
                                    }
                                }
                            }
                        }

                        // 3. Bottom Action Row
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Text(
                                text = "💡 Real-time Bayesian Elo: dynamically calibrated after every blind pair comparison.",
                                style = MaterialTheme.typography.labelSmall,
                                color = colors.textDim,
                            )
                            CapsuleButton(
                                text = "Return to Comparison [S / Esc]",
                                onClick = { showStandings = false },
                                compact = true,
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun BlindHeroImageCard(
    badgeLabel: String,
    buttonLabel: String,
    shortcutHint: String,
    imagePath: String,
    modifier: Modifier = Modifier,
    onSelect: () -> Unit,
) {
    val colors = rankoColors
    val quality = LocalThumbnailQuality.current
    val interactionSource = remember { MutableInteractionSource() }
    val isHovered by interactionSource.collectIsHoveredAsState()

    Column(
        modifier = modifier
            .background(colors.bgCard, shape = RoundedCornerShape(12.dp))
            .border(
                width = if (isHovered) 2.dp else 1.dp,
                color = if (isHovered) colors.accentBlue else colors.stroke,
                shape = RoundedCornerShape(12.dp),
            )
            .clip(RoundedCornerShape(12.dp))
            .clickable(
                interactionSource = interactionSource,
                indication = null,
                onClick = onSelect,
            )
            .pointerHoverIcon(pointerIconHand)
            .padding(12.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        // Top Card Badge & Shortcut
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Surface(
                color = if (isHovered) colors.accentBlue.copy(alpha = 0.2f) else colors.bgPanel,
                shape = RoundedCornerShape(6.dp),
            ) {
                Text(
                    text = badgeLabel,
                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 3.dp),
                    style = MaterialTheme.typography.labelSmall,
                    fontWeight = FontWeight.Bold,
                    color = colors.accentBlue,
                )
            }

            Text(
                text = shortcutHint,
                style = MaterialTheme.typography.labelSmall,
                color = colors.textDim,
            )
        }

        // Hero Image Viewport: Takes all available height in this card!
        Box(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .clip(RoundedCornerShape(8.dp))
                .background(Color.Black.copy(alpha = 0.45f)),
            contentAlignment = Alignment.Center,
        ) {
            AsyncImage(
                model = BlobRef(imagePath, maxEdge = 2048, quality = quality),
                contentDescription = badgeLabel,
                contentScale = ContentScale.Fit,
                modifier = Modifier.fillMaxSize(),
            )
        }

        // Choice Button
        CapsuleButton(
            text = buttonLabel,
            onClick = onSelect,
            compact = false,
            modifier = Modifier
                .fillMaxWidth()
                .height(44.dp),
        )
    }
}
