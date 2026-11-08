package com.acite.axlranko.pages.components

import com.acite.axlranko.data.BlindTestCheckpointRating
import com.acite.axlranko.data.BlindTestState
import com.acite.axlranko.model.CheckpointItem
import com.acite.axlranko.model.GeneratedSampleJob
import com.acite.axlranko.model.SampleItem
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.random.Random

/** One side-by-side presentation in the Blind Test dialog. */
internal data class BlindTestPair(
    val profileIndex: Int,
    val leftCheckpoint: String,
    val leftStep: Int?,
    val leftImage: SampleItem,
    val rightCheckpoint: String,
    val rightStep: Int?,
    val rightImage: SampleItem,
    val isCanary: Boolean = false,
)

/** One row in the Blind Test live leaderboard. */
internal data class BlindTestLeaderboardRow(
    val rank: Int,
    val checkpointPath: String,
    val step: Int?,
    val rating: Double,
    val uncertainty: Double,
    val conservativeScore: Double,
    val wins: Int,
    val losses: Int,
    val matches: Int,
    val winRate: Double,
)

/** Top-1 confidence summary. */
internal data class BlindTestTopConfidence(
    val leaderCheckpoint: String,
    val leaderStep: Int?,
    val runnerUpCheckpoint: String?,
    val runnerUpStep: Int?,
    val confidence: Double,
    val isConfirmed: Boolean,
)

/** Summary of sample counts and balance warning across pinned checkpoints. */
internal data class BlindTestPoolSummary(
    val pinnedCheckpointsCount: Int,
    val eligibleCheckpointsCount: Int,
    val eligibleProfilesCount: Int,
    val minImagesPerCheckpoint: Int,
    val maxImagesPerCheckpoint: Int,
    val isImbalanced: Boolean,
    val warningMessage: String?,
)

/**
 * Approximate standard normal CDF: Phi(z) = 0.5 * (1 + erf(z / sqrt(2))).
 * Using Abramowitz & Stegun 7.1.26 with maximum error < 1.5e-7.
 */
internal fun standardNormalCdf(z: Double): Double {
    val x = z / sqrt(2.0)
    val sign = if (x < 0) -1.0 else 1.0
    val absX = abs(x)
    val p = 0.3275911
    val t = 1.0 / (1.0 + p * absX)
    val poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    val erf = sign * (1.0 - poly * exp(-absX * absX))
    return (0.5 * (1.0 + erf)).coerceIn(0.0, 1.0)
}

/**
 * Inspect pinned checkpoints and their generated samples (from jobs with selection == "pinned").
 * Collects only valid set images (setIndex >= 0).
 */
internal fun resolveBlindTestPool(
    pinnedCheckpoints: List<CheckpointItem>,
    jobs: List<GeneratedSampleJob>,
): Map<String, Map<Int, List<SampleItem>>> {
    // 1. Identify all batch job IDs generated via "Generate pinned samples"
    val pinnedBatchIds = jobs
        .filter { (it.mode == "batch" && it.selection == "pinned") || it.id.contains("_pinned_") }
        .map { it.id }
        .toSet()

    // 2. Identify all completed child jobs that belong to a pinned batch OR were marked pinned
    val pinnedJobs = jobs.filter { job ->
        job.state == "done" && (
            (job.batchId != null && (job.batchId in pinnedBatchIds || job.batchId.contains("_pinned_"))) ||
            (job.selection == "pinned" && job.mode == "sets")
        )
    }

    fun norm(p: String): String = p.trim().trimEnd('/').replace('\\', '/')

    // Map: checkpointPath -> (profileIndex -> list of sample items)
    val pool = mutableMapOf<String, MutableMap<Int, MutableList<SampleItem>>>()

    for (cp in pinnedCheckpoints) {
        val cpPath = norm(cp.path)
        val ownJobs = pinnedJobs.filter {
            val jobCp = norm(it.checkpoint)
            jobCp == cpPath || (it.checkpoint.isBlank() && it.step != null && it.step == cp.step) || (cp.step != null && it.step == cp.step)
        }
        val profileMap = mutableMapOf<Int, MutableList<SampleItem>>()

        for (job in ownJobs) {
            val items = generatedSampleItems(job)
            for (item in items) {
                if (item.setIndex >= 0) {
                    profileMap.getOrPut(item.setIndex) { mutableListOf() }.add(item)
                }
            }
        }
        pool[cpPath] = profileMap
    }
    return pool
}

/** Check sample count balance across pinned checkpoints. */
internal fun summarizeBlindTestPool(
    pinnedCheckpoints: List<CheckpointItem>,
    pool: Map<String, Map<Int, List<SampleItem>>>,
): BlindTestPoolSummary {
    val pinnedCount = pinnedCheckpoints.size
    if (pinnedCount == 0) {
        return BlindTestPoolSummary(0, 0, 0, 0, 0, false, null)
    }

    val countsPerCp = pinnedCheckpoints.map { cp ->
        pool[cp.path]?.values?.sumOf { it.size } ?: 0
    }
    val minCount = countsPerCp.minOrNull() ?: 0
    val maxCount = countsPerCp.maxOrNull() ?: 0

    val eligibleProfiles = mutableSetOf<Int>()
    for (profileMap in pool.values) {
        eligibleProfiles.addAll(profileMap.keys)
    }
    val commonProfiles = eligibleProfiles.filter { p ->
        pool.values.count { (it[p]?.size ?: 0) > 0 } >= 2
    }

    val eligibleCps = pinnedCheckpoints.filter { cp ->
        (pool[cp.path]?.values?.sumOf { it.size } ?: 0) > 0
    }

    val isImbalanced = maxCount > 0 && (minCount == 0 || (minCount.toDouble() / maxCount.toDouble() < 0.5))
    val warning = if (isImbalanced) {
        "Pinned sample counts vary significantly (min: $minCount, max: $maxCount images). Running another round of \"Generate pinned samples\" is recommended for balanced evaluation."
    } else {
        null
    }

    return BlindTestPoolSummary(
        pinnedCheckpointsCount = pinnedCount,
        eligibleCheckpointsCount = eligibleCps.size,
        eligibleProfilesCount = commonProfiles.size,
        minImagesPerCheckpoint = minCount,
        maxImagesPerCheckpoint = maxCount,
        isImbalanced = isImbalanced,
        warningMessage = warning,
    )
}

/** Sample the next blind test pair using active weighted sampling. */
internal fun sampleBlindTestPair(
    pinnedCheckpoints: List<CheckpointItem>,
    pool: Map<String, Map<Int, List<SampleItem>>>,
    state: BlindTestState?,
    random: Random = Random.Default,
): BlindTestPair? {
    val stepByPath = pinnedCheckpoints.associate { it.path to it.step }

    // Find all profiles with at least 2 checkpoints holding samples
    val allProfiles = pool.values.flatMap { it.keys }.distinct()
    val validProfiles = allProfiles.filter { p ->
        pool.entries.count { (it.value[p]?.size ?: 0) > 0 } >= 2
    }
    if (validProfiles.isEmpty()) return null

    val history = state?.history.orEmpty()
    val roll = random.nextDouble()

    // 1. Canary re-test (15% chance): re-test a previously answered pair
    if (roll < 0.15 && history.isNotEmpty()) {
        val answeredHistory = history.filter { it.chosen in setOf("left", "right") && !it.isCanary }
        if (answeredHistory.isNotEmpty()) {
            val past = answeredHistory[random.nextInt(answeredHistory.size)]
            val p = past.profileIndex
            val imgsA = pool[past.leftCheckpoint]?.get(p).orEmpty()
            val imgsB = pool[past.rightCheckpoint]?.get(p).orEmpty()
            if (imgsA.isNotEmpty() && imgsB.isNotEmpty()) {
                val flip = random.nextBoolean()
                val cpL = if (flip) past.rightCheckpoint else past.leftCheckpoint
                val cpR = if (flip) past.leftCheckpoint else past.rightCheckpoint
                val imgL = if (flip) imgsB[random.nextInt(imgsB.size)] else imgsA[random.nextInt(imgsA.size)]
                val imgR = if (flip) imgsA[random.nextInt(imgsA.size)] else imgsB[random.nextInt(imgsB.size)]
                return BlindTestPair(
                    profileIndex = p,
                    leftCheckpoint = cpL,
                    leftStep = stepByPath[cpL],
                    leftImage = imgL,
                    rightCheckpoint = cpR,
                    rightStep = stepByPath[cpR],
                    rightImage = imgR,
                    isCanary = true,
                )
            }
        }
    }

    // Pick profile: randomly or weighted
    val profileIndex = validProfiles[random.nextInt(validProfiles.size)]
    val candidates = pool.entries.filter { (it.value[profileIndex]?.size ?: 0) > 0 }.map { it.key }
    if (candidates.size < 2) return null

    // Pick 2 distinct checkpoints
    val ratings = state?.ratings.orEmpty()
    val shuffled = candidates.shuffled(random)

    // Matchmaking: prefer candidates with fewer direct comparisons on this profile
    var bestPair = Pair(shuffled[0], shuffled[1])
    var minMatches = Int.MAX_VALUE

    for (i in 0 until min(shuffled.size, 4)) {
        for (j in i + 1 until min(shuffled.size, 4)) {
            val c1 = shuffled[i]
            val c2 = shuffled[j]
            val matchesOnProfile = history.count { h ->
                h.profileIndex == profileIndex &&
                    ((h.leftCheckpoint == c1 && h.rightCheckpoint == c2) ||
                        (h.leftCheckpoint == c2 && h.rightCheckpoint == c1))
            }
            if (matchesOnProfile < minMatches) {
                minMatches = matchesOnProfile
                bestPair = Pair(c1, c2)
            }
        }
    }

    val cpA = bestPair.first
    val cpB = bestPair.second
    val imgsA = pool[cpA]?.get(profileIndex).orEmpty()
    val imgsB = pool[cpB]?.get(profileIndex).orEmpty()
    if (imgsA.isEmpty() || imgsB.isEmpty()) return null

    val flip = random.nextBoolean()
    val leftCp = if (flip) cpB else cpA
    val rightCp = if (flip) cpA else cpB
    val leftImg = if (flip) imgsB[random.nextInt(imgsB.size)] else imgsA[random.nextInt(imgsA.size)]
    val rightImg = if (flip) imgsA[random.nextInt(imgsA.size)] else imgsB[random.nextInt(imgsB.size)]

    return BlindTestPair(
        profileIndex = profileIndex,
        leftCheckpoint = leftCp,
        leftStep = stepByPath[leftCp],
        leftImage = leftImg,
        rightCheckpoint = rightCp,
        rightStep = stepByPath[rightCp],
        rightImage = rightImg,
        isCanary = false,
    )
}

/** Compute leaderboard rows for all pinned checkpoints. */
internal fun computeLeaderboard(
    pinnedCheckpoints: List<CheckpointItem>,
    state: BlindTestState?,
): List<BlindTestLeaderboardRow> {
    val ratings = state?.ratings.orEmpty()

    val rows = pinnedCheckpoints.map { cp ->
        val record = ratings[cp.path] ?: BlindTestCheckpointRating()
        val r = record.rating
        val u = record.uncertainty
        val conservative = r - 1.0 * u
        val total = record.wins + record.losses
        val winRate = if (total > 0) record.wins.toDouble() / total.toDouble() else 0.0

        BlindTestLeaderboardRow(
            rank = 1,
            checkpointPath = cp.path,
            step = cp.step,
            rating = r,
            uncertainty = u,
            conservativeScore = conservative,
            wins = record.wins,
            losses = record.losses,
            matches = record.matches,
            winRate = winRate,
        )
    }

    // Sorted descending by Elo rating, then by win rate, then by lowest uncertainty
    val sorted = rows.sortedWith(
        compareByDescending<BlindTestLeaderboardRow> { it.rating }
            .thenByDescending { it.winRate }
            .thenBy { it.uncertainty }
    )

    return sorted.mapIndexed { index, row -> row.copy(rank = index + 1) }
}

/** Calculate Top-1 Dominance Confidence. */
internal fun computeTopConfidence(leaderboard: List<BlindTestLeaderboardRow>): BlindTestTopConfidence? {
    if (leaderboard.size < 2) return null
    val first = leaderboard[0]
    val second = leaderboard[1]

    val beta = 175.0
    val diff = first.rating - second.rating
    val denom = sqrt(first.uncertainty * first.uncertainty + second.uncertainty * second.uncertainty + 2.0 * beta * beta)
    val z = diff / max(1.0, denom)
    val confidence = standardNormalCdf(z)

    return BlindTestTopConfidence(
        leaderCheckpoint = first.checkpointPath,
        leaderStep = first.step,
        runnerUpCheckpoint = second.checkpointPath,
        runnerUpStep = second.step,
        confidence = confidence,
        isConfirmed = confidence >= 0.85,
    )
}
