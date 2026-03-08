package com.terminalremote.terminal

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.terminalremote.network.ConnectionState
import com.terminalremote.network.ServerMessage
import com.terminalremote.network.WebSocketClient
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch

data class TerminalTab(
    val sessionId: String,
    val name: String,
    val shell: String,
    val outputBuffer: StringBuilder = StringBuilder()
)

class TerminalViewModel : ViewModel() {

    val wsClient = WebSocketClient(viewModelScope)

    private val _tabs = MutableStateFlow<List<TerminalTab>>(emptyList())
    val tabs: StateFlow<List<TerminalTab>> = _tabs

    private val _activeTabId = MutableStateFlow<String?>(null)
    val activeTabId: StateFlow<String?> = _activeTabId

    private val _terminalOutput = MutableStateFlow("")
    val terminalOutput: StateFlow<String> = _terminalOutput

    init {
        viewModelScope.launch {
            wsClient.messages.collect { msg ->
                when (msg) {
                    is ServerMessage.SessionCreated -> {
                        val tab = TerminalTab(msg.sessionId, msg.name, msg.shell)
                        _tabs.value = _tabs.value + tab
                        _activeTabId.value = msg.sessionId
                    }
                    is ServerMessage.Output -> {
                        val current = _tabs.value.toMutableList()
                        val idx = current.indexOfFirst { it.sessionId == msg.sessionId }
                        if (idx >= 0) {
                            current[idx].outputBuffer.append(msg.data)
                            _tabs.value = current
                            if (_activeTabId.value == msg.sessionId) {
                                _terminalOutput.value = current[idx].outputBuffer.toString()
                            }
                        }
                    }
                    is ServerMessage.SessionClosed -> {
                        _tabs.value = _tabs.value.filter { it.sessionId != msg.sessionId }
                        if (_activeTabId.value == msg.sessionId) {
                            _activeTabId.value = _tabs.value.firstOrNull()?.sessionId
                        }
                    }
                    is ServerMessage.SessionList -> { /* Could sync tabs from server */ }
                }
            }
        }
    }

    fun connect(host: String, port: Int, token: String) {
        wsClient.connect(host, port, token)
    }

    fun disconnect() {
        wsClient.disconnect()
        _tabs.value = emptyList()
        _activeTabId.value = null
        _terminalOutput.value = ""
    }

    fun createSession(name: String? = null) {
        wsClient.createSession(name = name)
    }

    fun sendInput(text: String) {
        val activeId = _activeTabId.value ?: return
        wsClient.sendInput(activeId, text)
    }

    fun switchTab(sessionId: String) {
        _activeTabId.value = sessionId
        _terminalOutput.value = _tabs.value
            .find { it.sessionId == sessionId }
            ?.outputBuffer?.toString() ?: ""
    }

    fun closeSession(sessionId: String) {
        wsClient.closeSession(sessionId)
    }
}
