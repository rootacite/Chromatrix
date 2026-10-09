package com.acite.axlranko.model

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * The Quick page's character trigger word: one tag written into the first tag of every validation
 * sample set, and readable back only while the sets agree on that tag.
 */
class CharacterTriggerTest {

    private fun form(vararg prompts: String) = TrainingConfigForm(
        sampleSets = prompts.map { SampleSetForm(prompt = it) },
    )

    @Test
    fun theSharedTagIsTheFirstTagEverySetOpensWith() {
        assertEquals(
            "alice",
            form("alice, 1girl, solo", "alice, close-up").sharedLeadingTag(),
        )
    }

    @Test
    fun setsWhoseFirstTagsDifferHaveNoSharedTag() {
        assertNull(form("alice, 1girl", "bob, 1girl").sharedLeadingTag())
        // A blank prompt is not a tag either: it cannot be swapped for the word, so the field stays
        // off until the set has one.
        assertNull(form("alice, 1girl", "").sharedLeadingTag())
    }

    @Test
    fun oneSetIsItsOwnSharedTag() {
        assertEquals("alice", form("alice").sharedLeadingTag())
        // The tag is read trimmed, so a prompt typed with padding still agrees with a tidy one.
        assertEquals("alice", form("  alice , 1girl", "alice, close-up").sharedLeadingTag())
    }

    @Test
    fun writingTheTriggerReplacesTheFirstTagAndKeepsTheRest() {
        val written = form("alice, 1girl, solo", "alice, close-up").withSharedLeadingTag("bob")

        assertEquals(listOf("bob, 1girl, solo", "bob, close-up"), written.sampleSets.map { it.prompt })
        assertEquals("bob", written.sharedLeadingTag())
    }

    @Test
    fun writingTheTriggerIntoASingleTagPromptReplacesIt() {
        val written = form("alice", "alice, smile").withSharedLeadingTag("bob")

        assertEquals(listOf("bob", "bob, smile"), written.sampleSets.map { it.prompt })
    }

    @Test
    fun theWordIsTrimmedOnTheWayIn() {
        val written = form("alice, 1girl").withSharedLeadingTag("  bob  ")

        assertEquals("bob, 1girl", written.sampleSets.single().prompt)
    }

    @Test
    fun clearingTheTriggerDropsTheTagAndPromotesTheNext() {
        val cleared = form("alice, 1girl, solo").withSharedLeadingTag("")

        assertEquals("1girl, solo", cleared.sampleSets.single().prompt)
        // The prompts still agree, so the field follows the tag that is now first.
        assertEquals("1girl", cleared.sharedLeadingTag())
    }

    @Test
    fun clearingNeverEmptiesAPrompt() {
        val cleared = form("alice").withSharedLeadingTag("")

        assertEquals("alice", cleared.sampleSets.single().prompt)
        assertEquals("alice", cleared.sharedLeadingTag())
    }

    @Test
    fun theQuickSectionOwnsTheFieldsItShows() {
        // The rail's error badge and the jump-to-error both go through `owns`, so a field shown on
        // Quick has to be owned there as well as in its home section.
        for (key in listOf(
            "pretrained_model_name_or_path",
            "output_name",
            "epoch",
            "train_batch_size",
            "gradient_accumulation_steps",
            "save_every_n_steps",
            "sampling_enabled",
            "unet_learning_rate",
            "te_learning_rate",
            "network_type",
            "network_dim",
            "network_alpha",
            "conv_dim",
            "conv_alpha",
            "network_dropout",
        )) {
            assertTrue(ConfigSection.Quick.owns(key), "Quick does not own $key")
        }
        assertTrue(ConfigSection.Quick.owns(TRAIN_DATA_ERROR_PREFIX + "0.path"))
        assertFalse(ConfigSection.Quick.owns("train_resolution"))
        // The dataset folders and the run length are on this page, so the estimate is too.
        assertTrue(ConfigSection.Quick.showsStepEstimate)
        assertTrue(ConfigSection.Training.showsStepEstimate)
        assertFalse(ConfigSection.Network.showsStepEstimate)
    }
}
