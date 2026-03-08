package com.terminalremote.ui

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.view.KeyEvent
import android.view.inputmethod.EditorInfo
import android.widget.Toast
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.google.android.material.tabs.TabLayout
import com.terminalremote.R
import com.terminalremote.databinding.ActivityMainBinding
import com.terminalremote.network.ConnectionState
import com.terminalremote.terminal.TerminalViewModel
import com.terminalremote.theme.TerminalTheme
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val viewModel: TerminalViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val host = intent.getStringExtra("host")
        val port = intent.getIntExtra("port", 8765)
        val token = intent.getStringExtra("token")

        if (host == null || token == null) {
            startActivity(Intent(this, ConnectActivity::class.java))
            finish()
            return
        }

        applyTheme()
        setupTerminalInput()
        setupSpecialKeys()
        setupTabs()
        setupToolbar()
        observeState()

        viewModel.connect(host, port, token)
    }

    private fun applyTheme() {
        val prefs = getSharedPreferences("terminal_settings", Context.MODE_PRIVATE)
        val themeName = prefs.getString("theme", "Dark") ?: "Dark"
        val theme = TerminalTheme.byName(themeName)
        val fontSize = prefs.getFloat("font_size", 14f)

        binding.terminalOutput.apply {
            setBackgroundColor(theme.background)
            setTextColor(theme.foreground)
            textSize = fontSize
        }

        if (prefs.getBoolean("keep_screen_on", true)) {
            window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }
    }

    private fun setupTerminalInput() {
        binding.terminalOutput.apply {
            setTextIsSelectable(true)
            typeface = android.graphics.Typeface.MONOSPACE
        }

        binding.terminalInput.setOnEditorActionListener { _, actionId, event ->
            if (actionId == EditorInfo.IME_ACTION_SEND ||
                (event?.keyCode == KeyEvent.KEYCODE_ENTER && event.action == KeyEvent.ACTION_DOWN)
            ) {
                val text = binding.terminalInput.text.toString()
                viewModel.sendInput(text + "\n")
                binding.terminalInput.text?.clear()
                true
            } else false
        }
    }

    private fun setupSpecialKeys() {
        binding.btnTab.setOnClickListener { viewModel.sendInput("\t") }
        binding.btnEsc.setOnClickListener { viewModel.sendInput("\u001B") }
        binding.btnCtrlC.setOnClickListener { viewModel.sendInput("\u0003") }
        binding.btnUp.setOnClickListener { viewModel.sendInput("\u001B[A") }
        binding.btnDown.setOnClickListener { viewModel.sendInput("\u001B[B") }
        binding.btnLeft.setOnClickListener { viewModel.sendInput("\u001B[D") }
        binding.btnRight.setOnClickListener { viewModel.sendInput("\u001B[C") }
        binding.btnCtrl.setOnClickListener {
            it.isSelected = !it.isSelected
        }
    }

    private fun setupTabs() {
        binding.tabLayout.addOnTabSelectedListener(object : TabLayout.OnTabSelectedListener {
            override fun onTabSelected(tab: TabLayout.Tab?) {
                (tab?.tag as? String)?.let { viewModel.switchTab(it) }
            }
            override fun onTabUnselected(tab: TabLayout.Tab?) {}
            override fun onTabReselected(tab: TabLayout.Tab?) {}
        })
    }

    private fun setupToolbar() {
        binding.toolbar.setOnMenuItemClickListener { item ->
            when (item.itemId) {
                R.id.action_new_session -> { viewModel.createSession(); true }
                R.id.action_settings -> {
                    startActivity(Intent(this, SettingsActivity::class.java)); true
                }
                R.id.action_disconnect -> {
                    viewModel.disconnect()
                    startActivity(Intent(this, ConnectActivity::class.java))
                    finish(); true
                }
                else -> false
            }
        }
    }

    private fun observeState() {
        lifecycleScope.launch {
            viewModel.wsClient.connectionState.collectLatest { state ->
                binding.statusIndicator.text = when (state) {
                    ConnectionState.DISCONNECTED -> getString(R.string.disconnected)
                    ConnectionState.CONNECTING -> getString(R.string.connecting)
                    ConnectionState.AUTHENTICATING -> "Authenticating..."
                    ConnectionState.CONNECTED -> getString(R.string.connected)
                }
                binding.statusIndicator.setTextColor(when (state) {
                    ConnectionState.CONNECTED -> 0xFF4CAF50.toInt()
                    ConnectionState.CONNECTING, ConnectionState.AUTHENTICATING -> 0xFFFFC107.toInt()
                    ConnectionState.DISCONNECTED -> 0xFFF44336.toInt()
                })
                if (state == ConnectionState.CONNECTED && viewModel.tabs.value.isEmpty()) {
                    viewModel.createSession("Main")
                }
            }
        }

        lifecycleScope.launch {
            viewModel.terminalOutput.collectLatest { output ->
                binding.terminalOutput.text = output
                binding.terminalScrollView.post {
                    binding.terminalScrollView.fullScroll(android.view.View.FOCUS_DOWN)
                }
            }
        }

        lifecycleScope.launch {
            viewModel.tabs.collectLatest { tabs ->
                binding.tabLayout.removeAllTabs()
                tabs.forEach { tab ->
                    binding.tabLayout.addTab(
                        binding.tabLayout.newTab().setText(tab.name).setTag(tab.sessionId)
                    )
                }
            }
        }

        lifecycleScope.launch {
            viewModel.wsClient.errors.collect { error ->
                Toast.makeText(this@MainActivity, error, Toast.LENGTH_LONG).show()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        viewModel.disconnect()
    }
}
