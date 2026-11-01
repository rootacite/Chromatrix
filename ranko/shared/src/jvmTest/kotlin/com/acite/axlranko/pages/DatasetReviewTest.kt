package com.acite.axlranko.pages

import com.acite.axlranko.model.ImageItem
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class DatasetReviewTest {
    private fun item(stem: String, trashId: String? = null) = ImageItem(
        directory = "/data",
        stem = stem,
        imagePath = if (trashId == null) "/data/$stem.png" else "/trash/$trashId/$stem.png",
        txtPath = "/data/$stem.txt",
        tags = "tag",
        maskPath = "/data/$stem.mask.png",
        trashId = trashId,
    )

    @Test
    fun reviewListsTrashAfterActiveImagesAndCanHideIt() {
        val active = listOf(item("a"), item("b"))
        val trashed = listOf(item("c", "c" + "0".repeat(31)))

        assertEquals(listOf("a", "b", "c"), reviewItems(active, trashed, true).map { it.stem })
        assertEquals(listOf("a", "b"), reviewItems(active, trashed, false).map { it.stem })
        assertTrue(trashed.single().isTrashed)
        assertFalse(active.first().isTrashed)
    }

    @Test
    fun rescanKeepsTheOpenReviewImage() {
        val items = listOf(item("a"), item("b"))
        assertEquals(1, reviewIndexAfterRescan(items, "/data/b.png", previousIndex = 0))
    }

    @Test
    fun aTrashedImageFallsBackToTheSamePosition() {
        val items = listOf(item("a"), item("b"))
        val index = reviewIndexAfterRescan(items, "/data/gone.png", previousIndex = 1)
        assertEquals(1, index)
    }

    @Test
    fun steppingClampsAtBothEnds() {
        assertEquals(0, reviewStep(index = 0, delta = -1, count = 3))
        assertEquals(2, reviewStep(index = 2, delta = 1, count = 3))
        assertEquals(1, reviewStep(index = 0, delta = 1, count = 3))
        assertEquals(0, reviewStep(index = 0, delta = 1, count = 0))
    }
}
