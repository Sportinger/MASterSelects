package com.terminalremote.theme

import android.graphics.Color

data class TerminalTheme(
    val name: String,
    val background: Int,
    val foreground: Int,
    val cursorColor: Int,
) {
    companion object {
        val DARK = TerminalTheme(
            name = "Dark",
            background = Color.parseColor("#1A1A2E"),
            foreground = Color.parseColor("#E0E0E0"),
            cursorColor = Color.parseColor("#FFFFFF"),
        )

        val MONOKAI = TerminalTheme(
            name = "Monokai",
            background = Color.parseColor("#272822"),
            foreground = Color.parseColor("#F8F8F2"),
            cursorColor = Color.parseColor("#F8F8F0"),
        )

        val SOLARIZED = TerminalTheme(
            name = "Solarized Dark",
            background = Color.parseColor("#002B36"),
            foreground = Color.parseColor("#839496"),
            cursorColor = Color.parseColor("#93A1A1"),
        )

        val ALL = listOf(DARK, MONOKAI, SOLARIZED)

        fun byName(name: String): TerminalTheme =
            ALL.find { it.name == name } ?: DARK
    }
}
