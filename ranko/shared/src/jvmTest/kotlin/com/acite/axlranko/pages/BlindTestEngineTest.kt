package com.acite.axlranko.pages

import com.acite.axlranko.data.BlindTestCheckpointRating
import com.acite.axlranko.data.BlindTestState
import com.acite.axlranko.model.CheckpointItem
import com.acite.axlranko.model.GeneratedSampleJob
import com.acite.axlranko.pages.components.computeLeaderboard
import com.acite.axlranko.pages.components.computeTopConfidence
import com.acite.axlranko.pages.components.resolveBlindTestPool
import com.acite.axlranko.pages.components.sampleBlindTestPair
import com.acite.axlranko.pages.components.summarizeBlindTestPool
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class BlindTestEngineTest {

    @Test
    fun testPoolResolutionAndImbalanceWarning() {
        val cp1 = CheckpointItem(path = "/models/s1000.safetensors", step = 1000)
        val cp2 = CheckpointItem(path = "/models/s2000.safetensors", step = 2000)
        val pinned = listOf(cp1, cp2)

        // Job with selection = "pinned" for cp1: 2 files for set 0, 1 for set 1
        val job1 = GeneratedSampleJob(
            id = "job1",
            state = "done",
            mode = "sets",
            selection = "pinned",
            checkpoint = cp1.path,
            files = listOf(
                "/output/job1_p0_0.png",
                "/output/job1_p0_1.png",
                "/output/job1_p1_0.png",
            ),
        )

        // Job with selection = "range" (should be ignored!)
        val jobRange = GeneratedSampleJob(
            id = "jobRange",
            state = "done",
            mode = "sets",
            selection = "range",
            checkpoint = cp2.path,
            files = listOf("/output/jobRange_p0_0.png"),
        )

        val batchJob = GeneratedSampleJob(
            id = "test_pinned_batch_001",
            state = "done",
            mode = "batch",
            selection = "pinned",
        )

        // Child job tied via batchId to the pinned batch
        val job2 = GeneratedSampleJob(
            id = "job2",
            state = "done",
            mode = "sets",
            batchId = "test_pinned_batch_001",
            checkpoint = cp2.path,
            files = listOf("/output/job2_p0_0.png"),
        )

        val pool = resolveBlindTestPool(pinned, listOf(batchJob, job1, jobRange, job2))

        assertEquals(2, pool[cp1.path]?.get(0)?.size)
        assertEquals(1, pool[cp1.path]?.get(1)?.size)
        assertEquals(1, pool[cp2.path]?.get(0)?.size)
        // jobRange should not be included for cp2!
        assertEquals(null, pool[cp2.path]?.get(1))

        val summary = summarizeBlindTestPool(pinned, pool)
        assertEquals(2, summary.pinnedCheckpointsCount)
        assertEquals(2, summary.eligibleCheckpointsCount)
        assertEquals(1, summary.eligibleProfilesCount) // Profile 0 is held by both!
        assertEquals(1, summary.minImagesPerCheckpoint)
        assertEquals(3, summary.maxImagesPerCheckpoint)
        assertTrue(summary.isImbalanced) // 1 / 3 < 0.5
        assertNotNull(summary.warningMessage)
    }

    @Test
    fun testSamplePairWithinSameProfile() {
        val cp1 = CheckpointItem(path = "/models/s1000.safetensors", step = 1000)
        val cp2 = CheckpointItem(path = "/models/s2000.safetensors", step = 2000)
        val pinned = listOf(cp1, cp2)

        val job1 = GeneratedSampleJob(
            id = "job1",
            state = "done",
            mode = "sets",
            selection = "pinned",
            checkpoint = cp1.path,
            files = listOf("/output/job1_p0_0.png"),
        )
        val job2 = GeneratedSampleJob(
            id = "job2",
            state = "done",
            mode = "sets",
            selection = "pinned",
            checkpoint = cp2.path,
            files = listOf("/output/job2_p0_0.png"),
        )

        val pool = resolveBlindTestPool(pinned, listOf(job1, job2))
        val pair = sampleBlindTestPair(pinned, pool, state = null, random = Random(42))

        assertNotNull(pair)
        assertEquals(0, pair.profileIndex)
        // Must pair cp1 and cp2
        val paired = setOf(pair.leftCheckpoint, pair.rightCheckpoint)
        assertEquals(setOf(cp1.path, cp2.path), paired)
        // Images must match profile 0
        assertEquals(0, pair.leftImage.setIndex)
        assertEquals(0, pair.rightImage.setIndex)
    }

    @Test
    fun testLeaderboardAndTopConfidence() {
        val cp1 = CheckpointItem(path = "/models/s1000.safetensors", step = 1000)
        val cp2 = CheckpointItem(path = "/models/s2000.safetensors", step = 2000)
        val pinned = listOf(cp1, cp2)

        val state = BlindTestState(
            ratings = mapOf(
                cp1.path to BlindTestCheckpointRating(rating = 1750.0, uncertainty = 40.0, wins = 15, losses = 3, matches = 18),
                cp2.path to BlindTestCheckpointRating(rating = 1420.0, uncertainty = 45.0, wins = 3, losses = 15, matches = 18),
            )
        )

        val leaderboard = computeLeaderboard(pinned, state)
        assertEquals(2, leaderboard.size)
        assertEquals(1, leaderboard[0].rank)
        assertEquals(cp1.path, leaderboard[0].checkpointPath)
        assertEquals(2, leaderboard[1].rank)
        assertEquals(cp2.path, leaderboard[1].checkpointPath)

        val conf = computeTopConfidence(leaderboard)
        assertNotNull(conf)
        assertTrue(conf.isConfirmed)
        assertTrue(conf.confidence >= 0.85)
    }
}
