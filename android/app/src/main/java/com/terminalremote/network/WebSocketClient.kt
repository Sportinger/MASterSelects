package com.terminalremote.network

import android.util.Log
import com.google.gson.Gson
import com.google.gson.JsonObject
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import okhttp3.*
import java.time.Duration

enum class ConnectionState {
    DISCONNECTED, CONNECTING, AUTHENTICATING, CONNECTED
}

sealed class ServerMessage {
    data class Output(val sessionId: String, val data: String) : ServerMessage()
    data class SessionCreated(val sessionId: String, val name: String, val shell: String) : ServerMessage()
    data class SessionClosed(val sessionId: String) : ServerMessage()
    data class SessionList(val sessions: List<SessionInfo>) : ServerMessage()
}

data class SessionInfo(
    val id: String,
    val name: String,
    val shell: String,
    val createdAt: String
)

class WebSocketClient(private val scope: CoroutineScope) {

    private val TAG = "WebSocketClient"
    private val gson = Gson()
    private var webSocket: WebSocket? = null
    private val client = OkHttpClient.Builder()
        .pingInterval(Duration.ofSeconds(30))
        .build()

    private val _connectionState = MutableStateFlow(ConnectionState.DISCONNECTED)
    val connectionState: StateFlow<ConnectionState> = _connectionState

    private val _messages = MutableSharedFlow<ServerMessage>(extraBufferCapacity = 256)
    val messages: SharedFlow<ServerMessage> = _messages

    private val _errors = MutableSharedFlow<String>(extraBufferCapacity = 16)
    val errors: SharedFlow<String> = _errors

    private var authToken = ""

    fun connect(host: String, port: Int, token: String) {
        if (_connectionState.value != ConnectionState.DISCONNECTED) disconnect()

        authToken = token
        _connectionState.value = ConnectionState.CONNECTING

        val request = Request.Builder().url("ws://$host:$port").build()

        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.i(TAG, "Connected")
                _connectionState.value = ConnectionState.AUTHENTICATING
                val msg = JsonObject().apply {
                    addProperty("type", "auth")
                    addProperty("token", authToken)
                }
                webSocket.send(gson.toJson(msg))
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                try {
                    val json = gson.fromJson(text, JsonObject::class.java)
                    when (json.get("type")?.asString) {
                        "auth_result" -> {
                            if (json.get("success")?.asBoolean == true) {
                                _connectionState.value = ConnectionState.CONNECTED
                            } else {
                                scope.launch { _errors.emit(json.get("message")?.asString ?: "Auth failed") }
                                disconnect()
                            }
                        }
                        "output" -> scope.launch {
                            _messages.emit(ServerMessage.Output(
                                json.get("session_id")?.asString ?: "",
                                json.get("data")?.asString ?: ""
                            ))
                        }
                        "session_created" -> scope.launch {
                            _messages.emit(ServerMessage.SessionCreated(
                                json.get("session_id")?.asString ?: "",
                                json.get("name")?.asString ?: "",
                                json.get("shell")?.asString ?: ""
                            ))
                        }
                        "session_closed" -> scope.launch {
                            _messages.emit(ServerMessage.SessionClosed(
                                json.get("session_id")?.asString ?: ""
                            ))
                        }
                        "session_list" -> scope.launch {
                            val sessions = json.getAsJsonArray("sessions")?.map { e ->
                                val o = e.asJsonObject
                                SessionInfo(
                                    o.get("id")?.asString ?: "",
                                    o.get("name")?.asString ?: "",
                                    o.get("shell")?.asString ?: "",
                                    o.get("created_at")?.asString ?: ""
                                )
                            } ?: emptyList()
                            _messages.emit(ServerMessage.SessionList(sessions))
                        }
                        "error" -> scope.launch {
                            _errors.emit(json.get("message")?.asString ?: "Unknown error")
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Parse error: ${e.message}")
                }
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                _connectionState.value = ConnectionState.DISCONNECTED
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                scope.launch { _errors.emit("Connection failed: ${t.message}") }
                _connectionState.value = ConnectionState.DISCONNECTED
            }
        })
    }

    fun disconnect() {
        webSocket?.close(1000, "Disconnect")
        webSocket = null
        _connectionState.value = ConnectionState.DISCONNECTED
    }

    fun sendInput(sessionId: String, data: String) {
        send(JsonObject().apply {
            addProperty("type", "input")
            addProperty("session_id", sessionId)
            addProperty("data", data)
        })
    }

    fun createSession(shell: String? = null, name: String? = null) {
        send(JsonObject().apply {
            addProperty("type", "create_session")
            shell?.let { addProperty("shell", it) }
            name?.let { addProperty("name", it) }
        })
    }

    fun closeSession(sessionId: String) {
        send(JsonObject().apply {
            addProperty("type", "close_session")
            addProperty("session_id", sessionId)
        })
    }

    fun listSessions() {
        send(JsonObject().apply { addProperty("type", "list_sessions") })
    }

    fun resize(sessionId: String, cols: Int, rows: Int) {
        send(JsonObject().apply {
            addProperty("type", "resize")
            addProperty("session_id", sessionId)
            addProperty("cols", cols)
            addProperty("rows", rows)
        })
    }

    private fun send(msg: JsonObject) {
        webSocket?.send(gson.toJson(msg))
    }
}
