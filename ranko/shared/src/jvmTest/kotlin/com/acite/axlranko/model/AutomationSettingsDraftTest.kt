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
        assertEquals(0.9, draft.toSettings().universalLoraStrength)
    }

    @Test
    fun aStrengthOutsideTheWindowIsClampedOnTheWayOut() {
        // The slider cannot leave 0–2; a settings file that did is corrected when it is written back.
        assertEquals(2.0, AutomationSettingsDraft(universalLoraStrength = 5f).toSettings().universalLoraStrength)
        assertEquals(0.0, AutomationSettingsDraft(universalLoraStrength = -1f).toSettings().universalLoraStrength)
    }

    @Test
    fun universalGenerationParametersRoundTrip() {
        val saved = AutomationSettings(
            universalCfg = 6.0,
            universalWidth = 1152,
            universalHeight = 768,
            universalStep = 58,
        )
        val draft = AutomationSettingsDraft.of(saved)
        assertEquals("6", draft.universalCfg)
        assertEquals("1152", draft.universalWidth)
        assertEquals("768", draft.universalHeight)
        assertEquals("58", draft.universalStep)

        val converted = draft.toSettings()
        assertEquals(6.0, converted.universalCfg)
        assertEquals(1152, converted.universalWidth)
        assertEquals(768, converted.universalHeight)
        assertEquals(58, converted.universalStep)
    }

    @Test
    fun universalGenerationParametersEmptyByDefault() {
        val draft = AutomationSettingsDraft()
        assertEquals("", draft.universalCfg)
        assertEquals("", draft.universalWidth)
        assertEquals("", draft.universalHeight)
        assertEquals("", draft.universalStep)

        val converted = draft.toSettings()
        assertEquals(null, converted.universalCfg)
        assertEquals(null, converted.universalWidth)
        assertEquals(null, converted.universalHeight)
        assertEquals(null, converted.universalStep)
    }
}
