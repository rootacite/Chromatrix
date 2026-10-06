package com.acite.axlranko.pages.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuAnchorType
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import com.acite.axlranko.data.ComfyCheckpointList
import com.acite.axlranko.model.SampleBackend
import com.acite.axlranko.model.SampleBackendChoice
import com.acite.axlranko.model.SamplePassRequest
import com.acite.axlranko.model.SAMPLE_BACKEND_BUILTIN
import com.acite.axlranko.model.SAMPLE_BACKEND_COMFY
import com.acite.axlranko.ui.components.CapsuleButton
import com.acite.axlranko.ui.components.CapsuleChoice
import com.acite.axlranko.ui.components.PorcelainCard
import com.acite.axlranko.ui.components.rankoFieldColors
import com.acite.axlranko.ui.theme.rankoColors

private val BACKEND_DIALOG_WIDTH = 520.dp

/**
 * How a Checkpoints pass should be rendered, asked before it starts.
 *
 * It exists because the same four entries can draw either way: locally, with the run's checkpoint
 * loaded into this machine's pipeline, or through a listening ComfyUI on the bundled
 * `beta/Sampling.json`. Only the ComfyUI path needs anything from the user — the base model its
 * graph loads and the LoRA strength — because the prompts, sizes, steps, CFG, RescaleCFG multiplier
 * and seeds are the run's own either way.
 *
 * The entry's own controls stay where they are (the range row's steps, the pinned row's rounds), so
 * this dialog says what it is about to start rather than asking again.
 */
@Composable
internal fun SampleBackendDialog(
    request: SamplePassRequest,
    choice: SampleBackendChoice,
    checkpoints: ComfyCheckpointList?,
    checkpointsLoading: Boolean,
    checkpointsError: String?,
    starting: Boolean,
    error: String?,
    onChoiceChange: (SampleBackendChoice) -> Unit,
    onRefreshCheckpoints: () -> Unit,
    onConfirm: (SampleBackend?) -> Unit,
    onDismiss: () -> Unit,
    maxWidth: Dp = BACKEND_DIALOG_WIDTH,
) {
    val colors = rankoColors
    val problem = sampleBackendError(choice, checkpoints, checkpointsLoading)
    Dialog(onDismissRequest = onDismiss) {
        PorcelainCard {
            Column(
                modifier = Modifier
                    .width(minOf(BACKEND_DIALOG_WIDTH, maxWidth))
                    .heightIn(max = 640.dp)
                    .verticalScroll(rememberScrollState())
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                Text(
                    text = samplePassTitle(request.kind),
                    color = colors.text,
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 14.sp,
                )
                Text(text = request.detail, color = colors.textDim, fontSize = 11.sp)
                BackendChoices(choice = choice, onChoiceChange = onChoiceChange)
                if (choice.isComfy) {
                    ComfyBackendFields(
                        choice = choice,
                        checkpoints = checkpoints,
                        loading = checkpointsLoading,
                        error = checkpointsError,
                        onChoiceChange = onChoiceChange,
                        onRefresh = onRefreshCheckpoints,
                    )
                }
                (problem ?: error)?.let { message ->
                    Text(text = message, color = colors.qualityRed, fontSize = 12.sp)
                }
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    CapsuleButton(
                        text = if (starting) "Starting…" else "Start",
                        onClick = { onConfirm(sampleBackendPayload(choice)) },
                        emphasized = true,
                        // One pass at a time, and the ComfyUI path has to be usable: the helper
                        // would refuse a request for a machine it cannot find.
                        enabled = problem == null && !starting,
                    )
                    CapsuleButton(text = "Cancel", onClick = onDismiss, enabled = !starting)
                }
            }
        }
    }
}

/** The two renderers, as one row of choices. */
@Composable
internal fun BackendChoices(
    choice: SampleBackendChoice,
    onChoiceChange: (SampleBackendChoice) -> Unit,
) {
    val colors = rankoColors
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        CapsuleChoice(
            text = "Built-in",
            selected = choice.backend == SAMPLE_BACKEND_BUILTIN,
            onClick = { onChoiceChange(choice.copy(backend = SAMPLE_BACKEND_BUILTIN)) },
        )
        CapsuleChoice(
            text = "ComfyUI",
            selected = choice.isComfy,
            onClick = { onChoiceChange(choice.copy(backend = SAMPLE_BACKEND_COMFY)) },
        )
    }
    Text(
        text = "Built-in renders here with this run's checkpoint; ComfyUI queues the bundled " +
            "Sampling.json workflow on the ComfyUI listening on this machine.",
        color = colors.textDim,
        fontSize = 11.sp,
    )
}

/**
 * What only the ComfyUI path needs: the base model its graph loads and the LoRA strength.
 *
 * The base model is a dropdown over that instance's own `models/checkpoints` — the same list the
 * Automation page offers — because ComfyUI can only load a model by name from its own folder. The
 * first entry keeps the workflow's own base model.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
internal fun ComfyBackendFields(
    choice: SampleBackendChoice,
    checkpoints: ComfyCheckpointList?,
    loading: Boolean,
    error: String?,
    onChoiceChange: (SampleBackendChoice) -> Unit,
    onRefresh: () -> Unit,
) {
    val colors = rankoColors
    var expanded by remember { mutableStateOf(false) }
    val options = buildList {
        val current = choice.comfyCheckpoint
        if (current.isNotBlank() && current !in (checkpoints?.checkpoints ?: emptyList())) add(current)
        addAll(checkpoints?.checkpoints ?: emptyList())
    }
    Text(text = "Base model", color = colors.text, fontSize = 12.sp)
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        ExposedDropdownMenuBox(
            expanded = expanded && options.isNotEmpty(),
            onExpandedChange = { if (options.isNotEmpty()) expanded = it },
            modifier = Modifier.weight(1f),
        ) {
            OutlinedTextField(
                value = choice.comfyCheckpoint,
                onValueChange = {},
                readOnly = true,
                singleLine = true,
                placeholder = { Text("Workflow default", fontSize = 12.sp) },
                trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
                colors = rankoFieldColors(),
                modifier = Modifier
                    .fillMaxWidth()
                    .menuAnchor(ExposedDropdownMenuAnchorType.PrimaryNotEditable),
            )
            ExposedDropdownMenu(
                expanded = expanded && options.isNotEmpty(),
                onDismissRequest = { expanded = false },
                modifier = Modifier.heightIn(max = 320.dp),
            ) {
                DropdownMenuItem(
                    text = { Text("Workflow default", fontSize = 12.sp) },
                    onClick = {
                        onChoiceChange(choice.copy(comfyCheckpoint = ""))
                        expanded = false
                    },
                )
                options.forEach { name ->
                    DropdownMenuItem(
                        text = {
                            Text(name, fontSize = 12.sp, maxLines = 1, overflow = TextOverflow.Ellipsis)
                        },
                        onClick = {
                            onChoiceChange(choice.copy(comfyCheckpoint = name))
                            expanded = false
                        },
                    )
                }
            }
        }
        if (loading) {
            CircularProgressIndicator(modifier = Modifier.size(14.dp))
        }
        CapsuleButton(text = "Refresh", onClick = onRefresh, enabled = !loading, compact = true)
    }
    val notice = checkpoints?.error.orEmpty().ifBlank { error.orEmpty() }
    if (notice.isNotBlank()) {
        Text(text = notice, color = colors.qualityRed, fontSize = 11.sp)
    }
    checkpoints?.root?.takeIf { it.isNotBlank() }?.let { root ->
        Text(text = root, color = colors.textDim, fontSize = 10.sp, maxLines = 1, overflow = TextOverflow.Ellipsis)
    }
    OutlinedTextField(
        value = choice.comfyLoraStrength,
        onValueChange = { onChoiceChange(choice.copy(comfyLoraStrength = it)) },
        singleLine = true,
        label = { Text("LoRA strength") },
        supportingText = { Text("Applies to the model and the CLIP, like the workflow's own value") },
        isError = loraStrengthError(choice.comfyLoraStrength) != null,
        colors = rankoFieldColors(),
        modifier = Modifier.fillMaxWidth(),
    )
    Text(
        text = "Steps, CFG, RescaleCFG, size and seed come from this run's prompts. The checkpoint " +
            "is copied into ComfyUI's LoRA folder for the pass and removed when it ends.",
        color = colors.textDim,
        fontSize = 11.sp,
    )
}
