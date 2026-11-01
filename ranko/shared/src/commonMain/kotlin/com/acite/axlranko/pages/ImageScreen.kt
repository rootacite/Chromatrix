// ImageScreen.kt
package com.acite.axlranko.pages

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.focusable
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.gestures.detectVerticalDragGestures
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.input.key.Key
import androidx.compose.ui.input.key.KeyEventType
import androidx.compose.ui.input.key.key
import androidx.compose.ui.input.key.onPreviewKeyEvent
import androidx.compose.ui.input.key.type
import androidx.compose.ui.input.pointer.PointerIcon
import androidx.compose.ui.input.pointer.pointerHoverIcon
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import coil3.compose.AsyncImage
import com.acite.axlranko.model.ImageItem
import com.acite.axlranko.model.ImageScreenState
import com.acite.axlranko.pages.components.AspectLockedAsyncImage
import com.acite.axlranko.pages.components.MaskPreview
import com.acite.axlranko.util.BRUSH_RADIUS_MAX
import com.acite.axlranko.util.BRUSH_RADIUS_MIN
import com.acite.axlranko.ui.components.CapsuleButton
import com.acite.axlranko.ui.components.rankoFieldColors
import com.acite.axlranko.ui.theme.rankoColors
import com.acite.axlranko.ui.theme.rankoTokens
import dev.zacsweers.metrox.viewmodel.metroViewModel
import com.acite.axlranko.pages.components.DatasetDirBar
import com.acite.axlranko.pages.components.datasetDirLabel
import com.acite.axlranko.data.BlobRef
import com.acite.axlranko.data.LocalThumbnailQuality
import com.acite.axlranko.ui.isPortrait
import com.acite.axlranko.ui.wheelScrollsHorizontally
import com.acite.axlranko.ui.pointerIconHorizontalResize
import com.acite.axlranko.ui.pointerIconVerticalResize

@Composable
public fun ImagesScreen(
    viewModel: ImageScreenViewModel = metroViewModel()
) {
    val uiState by viewModel.uiState.collectAsState()

    val colors = rankoColors
    BoxWithConstraints(
        modifier = Modifier.fillMaxSize()
    ) {
        val totalWidthPx = constraints.maxWidth.toFloat()
        val portrait = isPortrait(maxWidth, maxHeight)

        Column(modifier = Modifier.fillMaxSize()) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(start = 8.dp, end = 8.dp, top = 6.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Box(modifier = Modifier.weight(1f)) {
                    DatasetDirBar(
                        labels = uiState.datasetDirs.map { datasetDirLabel(it.path, it.repeat) },
                        selected = uiState.datasetDirIndex,
                        onSelect = viewModel::selectDatasetDir,
                        modifier = Modifier.fillMaxWidth(),
                    )
                }
                CapsuleButton(
                    text = if (uiState.reviewMode) "Exit review" else "Review",
                    onClick = {
                        if (uiState.reviewMode) viewModel.exitReview() else viewModel.enterReview()
                    },
                    enabled = uiState.dataDir.isNotEmpty(),
                    compact = true,
                    emphasized = uiState.reviewMode,
                )
            }

            if (uiState.reviewMode) {
                DatasetReviewPane(
                    uiState = uiState,
                    viewModel = viewModel,
                    modifier = Modifier.weight(1f),
                )
            } else if (portrait) {
                BoxWithConstraints(modifier = Modifier.fillMaxWidth().weight(1f)) {
                    val totalHeightPx = constraints.maxHeight.toFloat()
                    Column(modifier = Modifier.fillMaxSize()) {
                        ImageThumbStrip(
                            uiState = uiState,
                            horizontal = true,
                            onSelect = viewModel::selectItem,
                            modifier = Modifier.fillMaxWidth().weight(uiState.portraitStripWeight),
                        )
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(8.dp)
                                .pointerHoverIcon(pointerIconVerticalResize)
                                .pointerInput(totalHeightPx) {
                                    detectVerticalDragGestures { _, dragAmount ->
                                        if (totalHeightPx > 0f) {
                                            viewModel.updatePortraitStripWeight(
                                                uiState.portraitStripWeight + dragAmount / totalHeightPx,
                                            )
                                        }
                                    }
                                },
                            contentAlignment = Alignment.Center,
                        ) {
                            HorizontalDivider(thickness = 1.dp, color = colors.stroke.copy(alpha = 0.55f))
                        }
                        ImageEditorPane(
                            uiState = uiState,
                            viewModel = viewModel,
                            modifier = Modifier.fillMaxWidth().weight(1f - uiState.portraitStripWeight),
                        )
                    }
                }
            } else Row(modifier = Modifier.fillMaxWidth().weight(1f)) {

            ImageThumbStrip(
                uiState = uiState,
                horizontal = false,
                onSelect = viewModel::selectItem,
                modifier = Modifier.fillMaxHeight().weight(uiState.leftWeight),
            )

            Box(
                modifier = Modifier
                    .fillMaxHeight()
                    .width(8.dp)
                    .pointerHoverIcon(pointerIconHorizontalResize)
                    .pointerInput(totalWidthPx) {
                        detectHorizontalDragGestures { _, dragAmount ->
                            if (totalWidthPx > 0) {
                                val fraction = dragAmount / totalWidthPx
                                viewModel.updateLeftWeight(uiState.leftWeight + fraction)
                            }
                        }
                    },
                contentAlignment = Alignment.Center
            ) {
                VerticalDivider(
                    thickness = 1.dp,
                    color = colors.stroke.copy(alpha = 0.55f)
                )
            }

            ImageEditorPane(
                uiState = uiState,
                viewModel = viewModel,
                modifier = Modifier.fillMaxHeight().weight(1f - uiState.leftWeight),
            )
        }
        }
    }
}

@Composable
private fun DatasetReviewPane(
    uiState: ImageScreenState,
    viewModel: ImageScreenViewModel,
    modifier: Modifier = Modifier,
) {
    val colors = rankoColors
    val items = reviewItems(uiState.imageItems, uiState.trashedItems, uiState.reviewShowTrashed)
    val index = uiState.reviewIndex.coerceIn(0, items.lastIndex.coerceAtLeast(0))
    val current = items.getOrNull(index)
    val focusRequester = remember { FocusRequester() }

    LaunchedEffect(Unit) { focusRequester.requestFocus() }

    BoxWithConstraints(modifier.fillMaxSize()) {
        val portrait = isPortrait(maxWidth, maxHeight)
        Column(
            modifier = Modifier
                .fillMaxSize()
                .focusRequester(focusRequester)
                .focusable()
                .onPreviewKeyEvent { event ->
                    if (event.type != KeyEventType.KeyDown) return@onPreviewKeyEvent false
                    when (event.key) {
                        Key.DirectionLeft, Key.PageUp -> {
                            viewModel.reviewStep(-1)
                            true
                        }
                        Key.DirectionRight, Key.PageDown -> {
                            viewModel.reviewStep(1)
                            true
                        }
                        Key.T -> {
                            if (!uiState.reviewBusy) viewModel.trashCurrentReviewItem()
                            true
                        }
                        Key.R -> {
                            if (!uiState.reviewBusy) viewModel.restoreCurrentReviewItem()
                            true
                        }
                        Key.Escape -> {
                            viewModel.exitReview()
                            true
                        }
                        else -> false
                    }
                },
        ) {
            DatasetReviewHeader(
                uiState = uiState,
                index = index,
                itemCount = items.size,
                viewModel = viewModel,
            )
            HorizontalDivider(color = colors.stroke.copy(alpha = 0.55f))
            if (portrait) {
                Column(modifier = Modifier.fillMaxWidth().weight(1f)) {
                    DatasetReviewImage(
                        item = current,
                        modifier = Modifier.fillMaxWidth().weight(0.66f),
                    )
                    DatasetReviewDetails(
                        item = current,
                        busy = uiState.reviewBusy,
                        viewModel = viewModel,
                        modifier = Modifier.fillMaxWidth().weight(0.34f),
                    )
                }
            } else {
                Row(modifier = Modifier.fillMaxWidth().weight(1f)) {
                    DatasetReviewImage(
                        item = current,
                        modifier = Modifier.fillMaxHeight().weight(1f),
                    )
                    VerticalDivider(color = colors.stroke.copy(alpha = 0.55f))
                    DatasetReviewDetails(
                        item = current,
                        busy = uiState.reviewBusy,
                        viewModel = viewModel,
                        modifier = Modifier
                            .fillMaxHeight()
                            .widthIn(min = 260.dp, max = 360.dp),
                    )
                }
            }
            DatasetReviewFooter(uiState = uiState, item = current)
        }
    }
}

@Composable
private fun DatasetReviewHeader(
    uiState: ImageScreenState,
    index: Int,
    itemCount: Int,
    viewModel: ImageScreenViewModel,
) {
    val colors = rankoColors
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 10.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(
                text = "Dataset review",
                color = colors.text,
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
            )
            Text(
                text = "${uiState.imageItems.size} active · ${uiState.trashedItems.size} trashed",
                color = colors.textDim,
                style = MaterialTheme.typography.labelSmall,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.weight(1f),
            )
            CapsuleButton(
                text = "Close",
                onClick = viewModel::exitReview,
                compact = true,
            )
        }
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .horizontalScroll(rememberScrollState()),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            CapsuleButton(
                text = "Previous (←)",
                onClick = { viewModel.reviewStep(-1) },
                enabled = itemCount > 0,
                compact = true,
            )
            Text(
                text = if (itemCount == 0) "0 / 0" else "${index + 1} / $itemCount",
                color = colors.text,
                style = MaterialTheme.typography.labelLarge,
            )
            CapsuleButton(
                text = "Next (→)",
                onClick = { viewModel.reviewStep(1) },
                enabled = itemCount > 0,
                compact = true,
            )
            CapsuleButton(
                text = if (uiState.reviewShowTrashed) {
                    "Trashed shown"
                } else {
                    "Trashed hidden"
                },
                onClick = viewModel::toggleReviewShowTrashed,
                compact = true,
                emphasized = uiState.reviewShowTrashed,
            )
        }
    }
}

@Composable
private fun DatasetReviewImage(
    item: ImageItem?,
    modifier: Modifier = Modifier,
) {
    val colors = rankoColors
    Box(
        modifier = modifier
            .background(Color.Black.copy(alpha = 0.72f))
            .padding(8.dp),
        contentAlignment = Alignment.Center,
    ) {
        if (item == null) {
            Text(
                text = "No images to review",
                color = colors.textDim,
                style = MaterialTheme.typography.titleMedium,
            )
        } else {
            AsyncImage(
                model = BlobRef(
                    item.imagePath,
                    maxEdge = maxOf(item.width, item.height, 512).coerceIn(256, 2048),
                    quality = LocalThumbnailQuality.current,
                ),
                contentDescription = item.stem,
                contentScale = ContentScale.Fit,
                filterQuality = FilterQuality.High,
                modifier = Modifier.fillMaxSize(),
            )
            if (item.isTrashed) {
                Box(
                    modifier = Modifier
                        .align(Alignment.TopStart)
                        .clip(rankoTokens.capsule)
                        .background(colors.qualityRed)
                        .padding(horizontal = 10.dp, vertical = 5.dp),
                ) {
                    Text(
                        text = "TRASHED",
                        color = Color.White,
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.Bold,
                    )
                }
            }
        }
    }
}

@Composable
private fun DatasetReviewDetails(
    item: ImageItem?,
    busy: Boolean,
    viewModel: ImageScreenViewModel,
    modifier: Modifier = Modifier,
) {
    val colors = rankoColors
    Column(
        modifier = modifier
            .verticalScroll(rememberScrollState())
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Text(
            text = item?.stem ?: "—",
            color = colors.text,
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.SemiBold,
            maxLines = 2,
            overflow = TextOverflow.Ellipsis,
        )
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            if (item?.isTrashed == true) ReviewBadge("Trashed", colors.qualityRed)
            if (item?.hasMask == true) ReviewBadge("Mask", colors.accentPink)
            if (item != null) {
                Text(
                    text = "${item.width} × ${item.height}",
                    color = colors.textDim,
                    style = MaterialTheme.typography.labelSmall,
                )
            }
        }
        if (item?.isTrashed == true) {
            Text(
                text = "${item.trashFileCount} files are kept together in Trash. " +
                    "Restore puts the whole group back.",
                color = colors.textDim,
                style = MaterialTheme.typography.bodySmall,
            )
        }
        HorizontalDivider(color = colors.stroke.copy(alpha = 0.35f))
        Text(
            text = "Tags",
            color = colors.textDim,
            style = MaterialTheme.typography.labelSmall,
        )
        Text(
            text = item?.currentTags?.ifBlank { "(no caption)" } ?: "—",
            color = colors.text,
            style = MaterialTheme.typography.bodyMedium,
        )
        if (item != null) {
            if (item.isTrashed) {
                CapsuleButton(
                    text = "Restore (R)",
                    onClick = viewModel::restoreCurrentReviewItem,
                    enabled = !busy && item.trashId != null,
                    compact = true,
                    emphasized = true,
                )
            } else {
                CapsuleButton(
                    text = "Move to Trash (T)",
                    onClick = viewModel::trashCurrentReviewItem,
                    enabled = !busy,
                    compact = true,
                    danger = true,
                )
                Text(
                    text = "The next run will not use this image while it is in Trash.",
                    color = colors.textDim,
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }
    }
}

@Composable
private fun ReviewBadge(text: String, color: Color) {
    Box(
        modifier = Modifier
            .clip(rankoTokens.capsule)
            .background(color.copy(alpha = 0.22f))
            .border(1.dp, color.copy(alpha = 0.55f), rankoTokens.capsule)
            .padding(horizontal = 8.dp, vertical = 3.dp),
    ) {
        Text(
            text = text,
            color = color,
            style = MaterialTheme.typography.labelSmall,
            fontWeight = FontWeight.SemiBold,
        )
    }
}

@Composable
private fun DatasetReviewFooter(uiState: ImageScreenState, item: ImageItem?) {
    val colors = rankoColors
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 10.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text(
            text = uiState.reviewError
                ?: uiState.reviewStatus
                ?: "← / → page · T move to Trash · R restore · Esc close",
            color = if (uiState.reviewError != null) colors.qualityRed else colors.textDim,
            style = MaterialTheme.typography.labelSmall,
            maxLines = 2,
            overflow = TextOverflow.Ellipsis,
            modifier = Modifier.weight(1f),
        )
        if (uiState.reviewBusy) {
            CircularProgressIndicator(
                modifier = Modifier.size(16.dp),
                strokeWidth = 2.dp,
                color = colors.accentPink,
            )
        } else {
            Text(
                text = when {
                    item == null -> "No selection"
                    item.isTrashed -> "Trashed"
                    else -> "Active"
                },
                color = if (item?.isTrashed == true) colors.qualityRed else colors.textDim,
                style = MaterialTheme.typography.labelSmall,
            )
        }
    }
}

@Composable
private fun ImageThumbStrip(
    uiState: ImageScreenState,
    horizontal: Boolean,
    onSelect: (ImageItem) -> Unit,
    modifier: Modifier = Modifier,
) {
    if (horizontal) {
        val listState = rememberLazyListState()
        LazyRow(
            state = listState,
            modifier = modifier.wheelScrollsHorizontally(listState),
            contentPadding = PaddingValues(8.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            items(uiState.imageItems, key = { it.imagePath }) { item ->
                val ratio = if (item.height > 0) item.width.toFloat() / item.height.toFloat() else 1f
                ImageThumbCell(
                    item = item,
                    selected = uiState.selectedItem?.imagePath == item.imagePath,
                    maskDirty = uiState.maskDirty,
                    onSelect = { onSelect(item) },
                    modifier = Modifier
                        .fillMaxHeight()
                        .aspectRatio(ratio, matchHeightConstraintsFirst = true),
                    scale = ContentScale.FillHeight,
                )
            }
        }
    } else {
        LazyColumn(
            modifier = modifier,
            contentPadding = PaddingValues(8.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            items(uiState.imageItems, key = { it.imagePath }) { item ->
                ImageThumbCell(
                    item = item,
                    selected = uiState.selectedItem?.imagePath == item.imagePath,
                    maskDirty = uiState.maskDirty,
                    onSelect = { onSelect(item) },
                    modifier = Modifier.fillMaxWidth(),
                    scale = ContentScale.FillWidth,
                )
            }
        }
    }
}

@Composable
private fun ImageThumbCell(
    item: ImageItem,
    selected: Boolean,
    maskDirty: Boolean,
    onSelect: () -> Unit,
    modifier: Modifier,
    scale: ContentScale,
) {
    val colors = rankoColors
    val tokens = rankoTokens
    val borderColor = when {
        item.isDirty -> colors.qualityRed
        selected -> colors.accentPink.copy(alpha = 0.28f)
        else -> Color.White.copy(alpha = 0.08f)
    }
    Box(
        modifier = modifier
            .clip(tokens.panel)
            .background(
                if (selected) colors.accentPink.copy(alpha = 0.16f)
                else colors.bgCard.copy(alpha = 0.55f),
            )
            .border(1.dp, borderColor, tokens.panel)
            .clickable(onClick = onSelect),
    ) {
        AspectLockedAsyncImage(
            path = item.imagePath,
            width = item.width,
            height = item.height,
            contentScale = scale,
            filterQuality = FilterQuality.High,
            modifier = if (scale == ContentScale.FillWidth) Modifier.fillMaxWidth() else Modifier.fillMaxSize(),
        )
        if (item.hasMask || (selected && maskDirty)) {
            Box(
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(6.dp)
                    .size(8.dp)
                    .clip(CircleShape)
                    .background(if (selected && maskDirty) colors.qualityRed else colors.accentPink),
            )
        }
    }
}

@Composable
private fun ImageEditorPane(
    uiState: ImageScreenState,
    viewModel: ImageScreenViewModel,
    modifier: Modifier = Modifier,
) {
    val colors = rankoColors
    val tokens = rankoTokens
    BoxWithConstraints(modifier) {
        val totalHeightPx = constraints.maxHeight.toFloat()
        Column(modifier = Modifier.fillMaxSize()) {

                    val hasSelection = uiState.selectedItem != null
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .heightIn(min = 36.dp)
                            .padding(start = 8.dp, end = 8.dp, top = 8.dp)
                            .horizontalScroll(rememberScrollState()),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        CapsuleButton(
                            text = "Mask",
                            onClick = { viewModel.setMaskEditEnabled(!uiState.maskEditEnabled) },
                            enabled = hasSelection,
                            compact = true,
                            emphasized = uiState.maskEditEnabled,
                        )
                        CapsuleButton(
                            text = "Mask only",
                            onClick = { viewModel.setMaskOnly(!uiState.maskOnly) },
                            enabled = hasSelection,
                            compact = true,
                            emphasized = uiState.maskOnly,
                        )
                        Text(
                            text = "Brush ${uiState.brushRadiusPx.toInt()}",
                            color = colors.textDim,
                            style = MaterialTheme.typography.labelSmall,
                        )
                        Slider(
                            value = uiState.brushRadiusPx,
                            onValueChange = { viewModel.setBrushRadius(it) },
                            valueRange = BRUSH_RADIUS_MIN..BRUSH_RADIUS_MAX,
                            enabled = hasSelection,
                            modifier = Modifier.width(120.dp),
                        )
                        Text(
                            text = "Feather ${(uiState.brushFeather * 100).toInt()}%",
                            color = colors.textDim,
                            style = MaterialTheme.typography.labelSmall,
                        )
                        Slider(
                            value = uiState.brushFeather,
                            onValueChange = { viewModel.setBrushFeather(it) },
                            valueRange = 0f..1f,
                            enabled = hasSelection,
                            modifier = Modifier.width(100.dp),
                        )
                        Text(
                            text = "Strength ${(uiState.brushStrength * 100).toInt()}%",
                            color = colors.textDim,
                            style = MaterialTheme.typography.labelSmall,
                        )
                        Slider(
                            value = uiState.brushStrength,
                            onValueChange = { viewModel.setBrushStrength(it) },
                            valueRange = 0.05f..1f,
                            enabled = hasSelection,
                            modifier = Modifier.width(100.dp),
                        )
                        CapsuleButton(
                            text = "Invert",
                            onClick = { viewModel.invertMask() },
                            enabled = hasSelection,
                            compact = true,
                        )
                        CapsuleButton(
                            text = "Fill white",
                            onClick = { viewModel.fillMask(white = true) },
                            enabled = hasSelection,
                            compact = true,
                        )
                        CapsuleButton(
                            text = "Fill black",
                            onClick = { viewModel.fillMask(white = false) },
                            enabled = hasSelection,
                            compact = true,
                        )
                        CapsuleButton(
                            text = "Clear",
                            onClick = { viewModel.clearMask() },
                            enabled = hasSelection && (
                                uiState.selectedItem?.hasSidecarMask == true || uiState.maskDirty
                            ),
                            compact = true,
                            danger = true,
                        )
                        CapsuleButton(
                            text = "Reset mask",
                            onClick = { viewModel.resetMask() },
                            enabled = hasSelection && uiState.maskDirty,
                            compact = true,
                        )
                        CapsuleButton(
                            text = "Save mask",
                            onClick = { viewModel.saveMask() },
                            enabled = hasSelection && uiState.maskDirty,
                            compact = true,
                            emphasized = true,
                        )
                    }

                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .weight(uiState.topWeight)
                            .padding(8.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        val previewRevision = uiState.maskPreviewRevision
                        val previewBitmap = remember(previewRevision) { viewModel.previewBitmap() }
                        if (uiState.selectedItem != null && previewBitmap != null) {
                            MaskPreview(
                                bitmap = previewBitmap,
                                sourceWidth = uiState.sourceWidth,
                                sourceHeight = uiState.sourceHeight,
                                editing = uiState.maskEditEnabled,
                                brushRadius = uiState.brushRadiusPx,
                                brushFeather = uiState.brushFeather,
                                onStrokeStart = { x, y, erase -> viewModel.beginMaskStroke(x, y, erase) },
                                onStrokeMove = { x, y -> viewModel.continueMaskStroke(x, y) },
                                onStrokeLeaveImage = { viewModel.leaveMaskImage() },
                                onStrokeEnd = { viewModel.endMaskStroke() },
                                onBrushResize = { viewModel.resizeBrushBy(it) },
                                modifier = Modifier.fillMaxSize(),
                            )
                        } else if (uiState.selectedItem != null) {
                            AsyncImage(
                                model = BlobRef(
                                    uiState.selectedItem!!.imagePath,
                                    maxEdge = maxOf(uiState.sourceWidth, uiState.sourceHeight, 256).coerceIn(32, 4096),
                                    quality = LocalThumbnailQuality.current,
                                ),
                                contentDescription = null,
                                contentScale = ContentScale.Fit,
                                filterQuality = FilterQuality.High,
                                modifier = Modifier.fillMaxSize()
                            )
                        } else {
                            Text(
                                text = "Select an image to edit tags",
                                color = colors.textDim
                            )
                        }
                    }

                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(8.dp)
                            .pointerHoverIcon(pointerIconVerticalResize)
                            .pointerInput(totalHeightPx) {
                                detectVerticalDragGestures { _, dragAmount ->
                                    if (totalHeightPx > 0) {
                                        val fraction = dragAmount / totalHeightPx
                                        viewModel.updateTopWeight(uiState.topWeight + fraction)
                                    }
                                }
                            },
                        contentAlignment = Alignment.Center
                    ) {
                        HorizontalDivider(
                            thickness = 1.dp,
                            color = colors.stroke.copy(alpha = 0.55f)
                        )
                    }

                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .weight(1f - uiState.topWeight)
                            .padding(8.dp)
                    ) {
                        OutlinedTextField(
                            value = uiState.editorText,
                            onValueChange = { viewModel.updateEditorText(it) },
                            modifier = Modifier.weight(1f).fillMaxWidth(),
                            enabled = uiState.selectedItem != null,
                            label = { Text("Tags") },
                            shape = tokens.panel,
                            colors = rankoFieldColors(),
                        )
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(top = 8.dp),
                            horizontalArrangement = Arrangement.End
                        ) {
                            CapsuleButton(
                                text = "Reset",
                                onClick = { viewModel.resetEditorText() },
                                enabled = uiState.selectedItem?.isDirty == true,
                                compact = true,
                                modifier = Modifier.padding(end = 8.dp),
                            )
                            CapsuleButton(
                                text = "Save",
                                onClick = { viewModel.saveTags() },
                                enabled = uiState.selectedItem?.isDirty == true,
                                compact = true,
                                emphasized = true,
                            )
                        }
                    }
                }
            }
}
