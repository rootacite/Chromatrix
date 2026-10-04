package com.acite.axlranko

import com.acite.axlranko.data.TomlDocumentPatcher
import com.acite.axlranko.data.loadTrainerConfig
import com.acite.axlranko.model.ConfigSection
import com.acite.axlranko.model.TrainingConfigForm
import com.acite.axlranko.model.VAL_INTERVAL_RANGE
import com.acite.axlranko.model.VAL_SAMPLE_COUNT_RANGE
import com.acite.axlranko.model.VAL_SPLIT_PERCENT_RANGE
import com.acite.axlranko.model.validationEnabled
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import okio.Path.Companion.toPath

/**
 * The validation-set split's three `[training]` keys: the sectional model decodes them (with
 * defaults when a config predates them), the Utils form shows and writes them in place through the
 * line-preserving patcher, and each range is refused under its own key.
 */
class ValSplitConfigTest {

    private fun writeConfig(training: String = ""): java.io.File {
        val file = Files.createTempFile("axl-val-split", ".toml").toFile()
        file.deleteOnExit()
        file.writeText(CONFIG.replace("TRAINING_PLACEHOLDER", training))
        return file
    }

    private fun load(training: String = "") =
        assertNotNull(loadTrainerConfig(writeConfig(training).absolutePath.toPath()))

    private fun formOf(training: String = "") = TrainingConfigForm.from(load(training))

    private fun save(source: String, form: TrainingConfigForm): String =
        TomlDocumentPatcher.apply(source, form.toTomlSections())

    @Test
    fun aConfigWithoutTheKeysUsesTheDefaults() {
        val config = load()
        assertEquals(10.0, config.training.valSplitPercent)
        assertEquals(8, config.training.valSampleCount)
        assertEquals(5, config.training.valInterval)

        val form = TrainingConfigForm.from(config)
        assertEquals("10", form.valSplitPercent)
        assertEquals("8", form.valSampleCount)
        assertEquals("5", form.valInterval)
        assertTrue(form.validate().isEmpty())
    }

    @Test
    fun theKeysAreReadFromTheFile() {
        val form = formOf(
            """
            val_split_percent = 25.5
            val_sample_count = 2
            val_interval = 0
            """.trimIndent(),
        )
        assertEquals("25.5", form.valSplitPercent)
        assertEquals("2", form.valSampleCount)
        assertEquals("0", form.valInterval)
        assertTrue(form.validate().isEmpty())
    }

    @Test
    fun savingWritesThemIntoTheTrainingTableWithoutDuplicatingThem() {
        val source = writeConfig().readText()
        val form = formOf().copy(valSplitPercent = "12.5", valSampleCount = "3", valInterval = "20")

        val once = save(source, form)
        // Written inside [training], before the next table, as a float for the percent.
        assertTrue(once.indexOf("val_split_percent = 12.5") < once.indexOf("[network]"))
        assertEquals(1, once.lines().count { it.trim().startsWith("val_split_percent") })
        assertEquals(1, once.lines().count { it.trim().startsWith("val_sample_count") })
        assertEquals(1, once.lines().count { it.trim().startsWith("val_interval") })

        val twice = save(once, form)
        assertEquals(1, twice.lines().count { it.trim().startsWith("val_split_percent") })

        val reloaded = reload(twice)
        assertEquals(12.5, reloaded.training.valSplitPercent)
        assertEquals(3, reloaded.training.valSampleCount)
        assertEquals(20, reloaded.training.valInterval)
    }

    @Test
    fun aWholeNumberPercentIsWrittenAsAFloat() {
        // ktoml refuses an integer literal for the Double field, so a hand-typed "10" must come
        // back as `10.0` - the same rule `TomlIntegerLiterals` repairs on read.
        val form = formOf().copy(valSplitPercent = "10")
        val patched = save(writeConfig().readText(), form)
        assertTrue(patched.contains("val_split_percent = 10.0"))
        assertEquals(10.0, reload(patched).training.valSplitPercent)
    }

    @Test
    fun eachRangeIsRefusedUnderItsOwnKey() {
        assertEquals("Max 90.0", formOf().copy(valSplitPercent = "91").validate()["val_split_percent"])
        assertEquals("Min 0.0", formOf().copy(valSplitPercent = "-1").validate()["val_split_percent"])
        assertEquals("Enter a number", formOf().copy(valSplitPercent = "half").validate()["val_split_percent"])
        assertEquals("Min 1", formOf().copy(valSampleCount = "0").validate()["val_sample_count"])
        assertEquals("Max 64", formOf().copy(valSampleCount = "65").validate()["val_sample_count"])
        assertEquals("Min 0", formOf().copy(valInterval = "-1").validate()["val_interval"])
        val low = VAL_SPLIT_PERCENT_RANGE.start
        val high = VAL_SPLIT_PERCENT_RANGE.endInclusive
        assertEquals(0.0, low)
        assertEquals(90.0, high)
        assertEquals(1, VAL_SAMPLE_COUNT_RANGE.first)
        assertEquals(0, VAL_INTERVAL_RANGE.first)
    }

    @Test
    fun zeroPercentIsTheOffSwitch() {
        // 0 % disables the held-out set: Utils greys the other two fields (`validationEnabled`) and
        // their numbers stop being range-checked, because neither pass would read them.
        val off = formOf().copy(valSplitPercent = "0")
        assertFalse(validationEnabled(off))
        assertTrue(validationEnabled(off.copy(valSplitPercent = "0.5")))
        assertTrue(validationEnabled(formOf()))
        assertTrue(off.copy(valSampleCount = "0", valInterval = "-4").validate().isEmpty())
        assertFalse(off.copy(valSplitPercent = "10", valSampleCount = "0").validate().isEmpty())
        assertEquals("Min 0.0", formOf().copy(valSplitPercent = "-1").validate()["val_split_percent"])
    }

    @Test
    fun theKeysBelongToTheTrainingSection() {
        for (key in listOf("val_split_percent", "val_sample_count", "val_interval")) {
            assertTrue(ConfigSection.Training.owns(key), key)
            assertFalse(ConfigSection.Environment.owns(key), key)
            assertFalse(ConfigSection.Validation.owns(key), key)
        }
    }

    private fun reload(configText: String) =
        assertNotNull(loadTrainerConfig(writeConfig().also { it.writeText(configText) }.absolutePath.toPath()))

    private companion object {
        val CONFIG = """
            [environment]
            pretrained_model_name_or_path = "/opt/models/sdxl"
            output_dir = "/tmp/out"
            logging_dir = "/tmp/logs"
            train_data_dir = "/tmp/data"
            output_name = "haruko"

            [model_spec]
            base_model_version = "sdxl_base_v1-0"
            modelspec_architecture = "stable-diffusion-xl-v1-base/lora"
            modelspec_implementation = "https://github.com/Stability-AI/generative-models"
            modelspec_sai_model_spec = "1.0.0"

            [training]
            min_snr_gamma = 5.0
            seed = 1145141919
            mixed_precision = "bf16"
            train_batch_size = 3
            gradient_accumulation_steps = 1
            learning_rate = 1.0
            epoch = 16
            save_every_n_epochs = 1
            save_every_n_steps = 100
            TRAINING_PLACEHOLDER

            [network]
            network_dim = 48
            network_alpha = 24
            network_dropout = 0.15
            clip_skip = 1
            max_token_length = 225

            [bucketing]
            enable_bucket = true
            bucket_no_upscale = true
            train_resolution = 1024
            bucket_reso_steps = 128
            min_bucket_reso = 768
            max_bucket_reso = 1280

            [optimization]
            cache_latents = true
            cache_latents_to_disk = true
            shuffle_caption = true
            keep_tokens = 2
            caption_extension = ".txt"
            noise_offset = 0.05

            [unet_optimizer]
            unet_learning_rate = 5.0E-5
            unet_weight_decay = 0.01
            unet_betas_1 = 0.9
            unet_betas_2 = 0.99
            unet_warmup_steps = 100
            unet_max_grad_norm = 1.0

            [te_optimizer]
            te_learning_rate = 5.0E-6
            te_weight_decay = 0.01
            te_betas_1 = 0.9
            te_betas_2 = 0.99
            te_max_grad_norm = 1.0
            te_warmup_steps = 100

            [infrastructure]
            max_data_loader_n_workers = 20
            persistent_workers = true

            [validation]
            sample_prompts = "flat prompt"
            sample_negative = "flat negative"
            sample_width = 1280
            sample_height = 720
            sample_steps = 35
            sample_seed = 0
            sample_repeat = 3
            guidance_scale = 6.0
        """.trimIndent()
    }
}
