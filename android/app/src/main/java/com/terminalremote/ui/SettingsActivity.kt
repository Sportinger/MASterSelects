package com.terminalremote.ui

import android.content.Context
import android.os.Bundle
import android.widget.ArrayAdapter
import androidx.appcompat.app.AppCompatActivity
import com.terminalremote.databinding.ActivitySettingsBinding
import com.terminalremote.theme.TerminalTheme

class SettingsActivity : AppCompatActivity() {

    private lateinit var binding: ActivitySettingsBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivitySettingsBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val prefs = getSharedPreferences("terminal_settings", Context.MODE_PRIVATE)

        // Font size
        val fontSize = prefs.getFloat("font_size", 14f)
        binding.sliderFontSize.value = fontSize
        binding.tvFontSizeValue.text = "${fontSize.toInt()}sp"
        binding.sliderFontSize.addOnChangeListener { _, value, _ ->
            binding.tvFontSizeValue.text = "${value.toInt()}sp"
        }

        // Theme
        val themeNames = TerminalTheme.ALL.map { it.name }
        val currentTheme = prefs.getString("theme", "Dark") ?: "Dark"
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, themeNames)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        binding.spinnerTheme.adapter = adapter
        binding.spinnerTheme.setSelection(themeNames.indexOf(currentTheme).coerceAtLeast(0))

        // Toggles
        binding.switchKeepScreenOn.isChecked = prefs.getBoolean("keep_screen_on", true)
        binding.switchVibrate.isChecked = prefs.getBoolean("vibrate_on_key", true)

        binding.btnSaveSettings.setOnClickListener {
            prefs.edit()
                .putFloat("font_size", binding.sliderFontSize.value)
                .putString("theme", themeNames[binding.spinnerTheme.selectedItemPosition])
                .putBoolean("keep_screen_on", binding.switchKeepScreenOn.isChecked)
                .putBoolean("vibrate_on_key", binding.switchVibrate.isChecked)
                .apply()
            finish()
        }

        binding.btnBack.setOnClickListener { finish() }
    }
}
