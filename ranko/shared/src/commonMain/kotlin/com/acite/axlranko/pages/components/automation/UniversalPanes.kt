package com.acite.axlranko.pages.components.automation

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuAnchorType
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.acite.axlranko.model.AutomationUiState
import com.acite.axlranko.pages.AutomationScreenViewModel
import com.acite.axlranko.ui.components.CapsuleButton
import com.acite.axlranko.ui.components.PorcelainCard
import com.acite.axlranko.ui.components.rankoFieldColors
import com.acite.axlranko.ui.theme.rankoColors
import dev.zacsweers.metrox.viewmodel.metroViewModel

/** The bundled-workflow section: same server and batch as ComfyUI, plus a LoRA and a trigger. */
@Composable
fun UniversalPane(
    state: AutomationUiState,
    viewModel: AutomationScreenViewModel = metroViewModel(),
    portrait: Boolean = false,
) {
    Column(
        modifier = Modifier.fillMaxWidth().verticalScroll(rememberScrollState()).padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        ServerCard(state, viewModel)
        CheckpointCard(state, viewModel)
        LoraCard(state, viewModel)
        BatchCard(
            state = state,
            viewModel = viewModel,
            portrait = portrait,
            source = state.universalSource,
            onSource = viewModel::setUniversalSource,
            setName = state.universalSetName,
            onSetName = viewModel::setUniversalSetName,
            manual = state.universalManual,
            onManual = viewModel::setUniversalManual,
            onStart = viewModel::startUniversalJob,
            onSaveSet = { name -> viewModel.savePromptSet(name, state.universalManual) },
        )
        JobLogCard(state, viewModel)
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun CheckpointCard(state: AutomationUiState, viewModel: AutomationScreenViewModel) {
    val colors = rankoColors
    val lang = state.language
    var expanded by remember { mutableStateOf(false) }
    val options = buildList {
        val current = state.settings.universalCheckpoint
        if (current.isNotBlank() && current !in state.checkpoints) add(current)
        addAll(state.checkpoints)
    }
    PorcelainCard {
        Column(
            modifier = Modifier.fillMaxWidth().padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(
                text = uiText(lang, "checkpoint"),
                color = colors.text,
                fontWeight = FontWeight.SemiBold,
                fontSize = 13.sp,
            )
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                ExposedDropdownMenuBox(
                    expanded = expanded && options.isNotEmpty(),
                    onExpandedChange = { if (options.isNotEmpty()) expanded = it },
                    modifier = Modifier.weight(1f),
                ) {
                    OutlinedTextField(
                        value = state.settings.universalCheckpoint,
                        onValueChange = {},
                        readOnly = true,
                        singleLine = true,
                        placeholder = { Text(uiText(lang, "checkpoint_default"), fontSize = 12.sp) },
                        trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
                        colors = rankoFieldColors(),
                        modifier = Modifier.fillMaxWidth().menuAnchor(ExposedDropdownMenuAnchorType.PrimaryNotEditable),
                    )
                    ExposedDropdownMenu(
                        expanded = expanded && options.isNotEmpty(),
                        onDismissRequest = { expanded = false },
                        modifier = Modifier.heightIn(max = 320.dp),
                    ) {
                        DropdownMenuItem(
                            text = {
                                Text(
                                    text = uiText(lang, "checkpoint_default"),
                                    fontSize = 12.sp,
                                    maxLines = 1,
                                    overflow = TextOverflow.Ellipsis,
                                )
                            },
                            onClick = {
                                viewModel.setUniversalCheckpoint("")
                                expanded = false
                            },
                        )
                        options.forEach { name ->
                            DropdownMenuItem(
                                text = {
                                    Text(
                                        text = name,
                                        fontSize = 12.sp,
                                        maxLines = 1,
                                        overflow = TextOverflow.Ellipsis,
                                    )
                                },
                                onClick = {
                                    viewModel.setUniversalCheckpoint(name)
                                    expanded = false
                                },
                            )
                        }
                    }
                }
                if (state.checkpointsLoading) {
                    CircularProgressIndicator(modifier = Modifier.size(14.dp))
                }
                CapsuleButton(
                    text = uiText(lang, "refresh"),
                    onClick = viewModel::refreshCheckpoints,
                    enabled = !state.checkpointsLoading,
                    compact = true,
                )
            }
            Text(text = uiText(lang, "checkpoint_hint"), color = colors.textDim, fontSize = 11.sp)
            if (state.checkpointRoot.isNotBlank()) {
                Text(text = state.checkpointRoot, color = colors.textDim, fontSize = 10.sp, maxLines = 1, overflow = TextOverflow.Ellipsis)
            }
            state.checkpointsError?.let { Text(it, color = colors.qualityRed, fontSize = 11.sp) }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun LoraCard(state: AutomationUiState, viewModel: AutomationScreenViewModel) {
    val colors = rankoColors
    val lang = state.language
    var expanded by remember { mutableStateOf(false) }
    val options = buildList {
        val current = state.settings.universalLora
        if (current.isNotBlank() && current !in state.loras) add(current)
        addAll(state.loras)
    }
    PorcelainCard {
        Column(
            modifier = Modifier.fillMaxWidth().padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(
                text = uiText(lang, "lora"),
                color = colors.text,
                fontWeight = FontWeight.SemiBold,
                fontSize = 13.sp,
            )
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                ExposedDropdownMenuBox(
                    expanded = expanded && options.isNotEmpty(),
                    onExpandedChange = { if (options.isNotEmpty()) expanded = it },
                    modifier = Modifier.weight(1f),
                ) {
                    OutlinedTextField(
                        value = state.settings.universalLora,
                        onValueChange = {},
                        readOnly = true,
                        singleLine = true,
                        placeholder = { Text(uiText(lang, "lora_empty"), fontSize = 12.sp) },
                        trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
                        colors = rankoFieldColors(),
                        modifier = Modifier.fillMaxWidth().menuAnchor(ExposedDropdownMenuAnchorType.PrimaryNotEditable),
                    )
                    ExposedDropdownMenu(
                        expanded = expanded && options.isNotEmpty(),
                        onDismissRequest = { expanded = false },
                        modifier = Modifier.heightIn(max = 320.dp),
                    ) {
                        options.forEach { name ->
                            DropdownMenuItem(
                                text = {
                                    Text(
                                        text = name,
                                        fontSize = 12.sp,
                                        maxLines = 1,
                                        overflow = TextOverflow.Ellipsis,
                                    )
                                },
                                onClick = {
                                    viewModel.setUniversalLora(name)
                                    expanded = false
                                },
                            )
                        }
                    }
                }
                if (state.lorasLoading) {
                    CircularProgressIndicator(modifier = Modifier.size(14.dp))
                }
                CapsuleButton(
                    text = uiText(lang, "refresh"),
                    onClick = viewModel::refreshLoras,
                    enabled = !state.lorasLoading,
                    compact = true,
                )
            }
            Text(text = uiText(lang, "lora_hint"), color = colors.textDim, fontSize = 11.sp)
            if (state.loraRoot.isNotBlank()) {
                Text(text = state.loraRoot, color = colors.textDim, fontSize = 10.sp, maxLines = 1, overflow = TextOverflow.Ellipsis)
            }
            state.lorasError?.let { Text(it, color = colors.qualityRed, fontSize = 11.sp) }
            Text(
                text = uiText(lang, "trigger"),
                color = colors.text,
                fontWeight = FontWeight.SemiBold,
                fontSize = 13.sp,
            )
            OutlinedTextField(
                value = state.settings.universalTrigger,
                onValueChange = viewModel::setUniversalTrigger,
                singleLine = true,
                placeholder = { Text("(yui_character:1.1)", fontSize = 12.sp) },
                colors = rankoFieldColors(),
                modifier = Modifier.fillMaxWidth(),
            )
            Text(text = uiText(lang, "trigger_hint"), color = colors.textDim, fontSize = 11.sp)
        }
    }
}
