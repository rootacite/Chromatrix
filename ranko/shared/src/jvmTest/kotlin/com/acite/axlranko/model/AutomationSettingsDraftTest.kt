package com.acite.axlranko.model

import com.acite.axlranko.data.AutomationSettings
import kotlin.test.Test
import kotlin.test.assertEquals

/** The Universal (Beta) LoRA strength the slider edits, on its way to `automation/settings.json`. */
class AutomationSettingsDraftTest {

    @Test
    fun theSavedStrengthComesBackOntoTheSlider() {
        val saved = AutomationSettings(universalLoraStrength = 0.85)
        val draft = AutomationSettingsDraft.of(saved)
        assertEquals(0.85f, draft.universalLoraStrength)
        assertEquals(0.85, draft.toSettings().universalLoraStrength)
    }

    @Test
    fun aSliderValueIsSentRoundedToTwoDecimals() {
        // 0.85f is 0.8500000238418579 as a Double; the file should hold 0.85.
        assertEquals(0.85, AutomationSettingsDraft(universalLoraStrength = 0.85f).toSettings().universalLoraStrength)
    }

    @Test
    fun theDefaultIsTheWorkflowsOwnValue() {
        val draft = AutomationSettingsDraft()
        assertEquals(UNIVERSAL_LORA_STRENGTH_DEFAULT, draft.universalLoraStrength)
        assertEquals(1.0, draft.toSettings().universalLoraStrength)
    }

    @Test
    fun aStrengthOutsideTheWindowIsClampedOnTheWayOut() {
        // The slider cannot leave 0–2; a settings file that did is corrected when it is written back.
        assertEquals(2.0, AutomationSettingsDraft(universalLoraStrength = 5f).toSettings().universalLoraStrength)
        assertEquals(0.0, AutomationSettingsDraft(universalLoraStrength = -1f).toSettings().universalLoraStrength)
    }
}
