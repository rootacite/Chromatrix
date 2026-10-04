package com.acite.axlranko.util

/**
 * WD-tagger English → Chinese map from `tagger/selected_tags.csv`.
 *
 * Chromatrix still stores captions as English; this is display (and search) only.
 */
class TagLexicon(
    private val chineseByEnglish: Map<String, String>,
    /** Every normalized `name` column of the CSV, whether it carries a translation or not. */
    private val englishNames: Set<String> = emptySet(),
) {

    fun chinese(tag: String): String? =
        chineseByEnglish[normalizeTag(tag)]?.takeIf { it.isNotEmpty() }

    /** True when the tag is a row of `selected_tags.csv` — a known Danbooru tag. */
    fun isKnownTag(tag: String): Boolean = normalizeTag(tag) in englishNames

    /** `english [chinese]` when a distinct translation exists; otherwise the tag as stored. */
    fun display(tag: String): String {
        val zh = chinese(tag) ?: return tag
        if (zh.equals(tag, ignoreCase = true)) return tag
        return "$tag [$zh]"
    }

    fun matchesQuery(tag: String, query: String): Boolean {
        if (query.isEmpty()) return true
        if (tag.contains(query, ignoreCase = true)) return true
        val zh = chinese(tag) ?: return false
        return zh.contains(query, ignoreCase = true)
    }

    companion object {
        fun parse(text: String): TagLexicon {
            val lines = text.replace("\uFEFF", "").lineSequence().filter { it.isNotEmpty() }
            val iterator = lines.iterator()
            if (!iterator.hasNext()) return TagLexicon(emptyMap())
            val header = parseCsvLine(iterator.next())
            val nameIdx = header.indexOf("name").takeIf { it >= 0 } ?: 1
            val zhIdx = header.indexOf("Translated")

            val names = linkedSetOf<String>()
            val map = linkedMapOf<String, String>()
            for (line in iterator) {
                val row = parseCsvLine(line)
                if (row.size <= nameIdx) continue
                val english = normalizeTag(row[nameIdx])
                if (english.isEmpty()) continue
                // A tag with no translation is still a known tag; only the display map skips it.
                names += english
                if (zhIdx < 0 || row.size <= zhIdx) continue
                val chinese = row[zhIdx].trim()
                if (chinese.isEmpty()) continue
                if (english !in map) map[english] = chinese
            }
            return TagLexicon(map, names)
        }

        fun load(text: String): TagLexicon = parse(text)
    }
}

object TagTranslations {
    var lexicon: TagLexicon = TagLexicon(emptyMap())
        private set

    fun install(text: String) {
        lexicon = TagLexicon.parse(text)
    }

    fun display(tag: String): String = lexicon.display(tag)

    fun matchesQuery(tag: String, query: String): Boolean = lexicon.matchesQuery(tag, query)
}

internal fun normalizeTag(tag: String): String = tag.trim().replace('_', ' ').lowercase()

/** One RFC4180 row: commas, quoted fields, doubled quotes. Newlines inside a field are not supported. */
internal fun parseCsvLine(line: String): List<String> {
    val fields = mutableListOf<String>()
    val field = StringBuilder()
    var i = 0
    var inQuotes = false
    while (i < line.length) {
        val c = line[i]
        when {
            c == '"' -> {
                if (inQuotes && i + 1 < line.length && line[i + 1] == '"') {
                    field.append('"')
                    i++
                } else {
                    inQuotes = !inQuotes
                }
            }
            c == ',' && !inQuotes -> {
                fields.add(field.toString())
                field.clear()
            }
            else -> field.append(c)
        }
        i++
    }
    fields.add(field.toString())
    return fields
}
