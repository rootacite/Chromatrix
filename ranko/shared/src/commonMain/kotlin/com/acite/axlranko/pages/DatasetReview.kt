package com.acite.axlranko.pages

import com.acite.axlranko.model.ImageItem

/**
 * The Images review list: active images in scan order, then recoverable Trash items. The Trash
 * section is optional so a long review can focus on what a run would still use.
 */
internal fun reviewItems(
    active: List<ImageItem>,
    trashed: List<ImageItem>,
    showTrashed: Boolean,
): List<ImageItem> = active + if (showTrashed) trashed else emptyList()

/** Keeps the open review image across a rescan; a removed image falls back to its old index. */
internal fun reviewIndexAfterRescan(
    items: List<ImageItem>,
    previousImagePath: String?,
    previousIndex: Int,
): Int {
    if (items.isEmpty()) return 0
    val remembered = previousImagePath?.let { path ->
        items.indexOfFirst { it.imagePath == path }
    } ?: -1
    return if (remembered >= 0) remembered else previousIndex.coerceIn(0, items.lastIndex)
}

/** One step through the review list, clamped at both ends. */
internal fun reviewStep(index: Int, delta: Int, count: Int): Int {
    if (count <= 0) return 0
    return (index + delta).coerceIn(0, count - 1)
}

/** Index of [imagePath], or [fallback] when it is not in the current list. */
internal fun reviewIndexOf(items: List<ImageItem>, imagePath: String?, fallback: Int = 0): Int {
    val found = imagePath?.let { path -> items.indexOfFirst { it.imagePath == path } } ?: -1
    return if (found >= 0) found else fallback.coerceIn(0, (items.size - 1).coerceAtLeast(0))
}
