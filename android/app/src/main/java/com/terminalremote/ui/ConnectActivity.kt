package com.terminalremote.ui

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import com.terminalremote.databinding.ActivityConnectBinding

data class SavedConnection(
    val name: String,
    val host: String,
    val port: Int,
    val token: String
)

class ConnectActivity : AppCompatActivity() {

    private lateinit var binding: ActivityConnectBinding
    private val gson = Gson()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityConnectBinding.inflate(layoutInflater)
        setContentView(binding.root)

        loadLastConnection()

        binding.btnConnect.setOnClickListener {
            val host = binding.editHost.text.toString().trim()
            val portStr = binding.editPort.text.toString().trim()
            val token = binding.editToken.text.toString().trim()

            if (host.isEmpty() || portStr.isEmpty() || token.isEmpty()) {
                Toast.makeText(this, "Please fill all fields", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            val port = portStr.toIntOrNull()
            if (port == null || port !in 1..65535) {
                Toast.makeText(this, "Invalid port", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            saveLastConnection(host, port, token)

            startActivity(Intent(this, MainActivity::class.java).apply {
                putExtra("host", host)
                putExtra("port", port)
                putExtra("token", token)
            })
        }

        binding.btnSave.setOnClickListener {
            val name = binding.editConnectionName.text.toString().trim()
            val host = binding.editHost.text.toString().trim()
            val portStr = binding.editPort.text.toString().trim()
            val token = binding.editToken.text.toString().trim()

            if (name.isEmpty() || host.isEmpty() || portStr.isEmpty() || token.isEmpty()) {
                Toast.makeText(this, "Fill all fields including name", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            val port = portStr.toIntOrNull() ?: return@setOnClickListener
            saveConnection(SavedConnection(name, host, port, token))
            Toast.makeText(this, "Saved!", Toast.LENGTH_SHORT).show()
        }
    }

    private fun loadLastConnection() {
        val prefs = getSharedPreferences("terminal_remote", Context.MODE_PRIVATE)
        binding.editHost.setText(prefs.getString("last_host", ""))
        binding.editPort.setText(prefs.getInt("last_port", 8765).toString())
        binding.editToken.setText(prefs.getString("last_token", ""))
    }

    private fun saveLastConnection(host: String, port: Int, token: String) {
        getSharedPreferences("terminal_remote", Context.MODE_PRIVATE).edit()
            .putString("last_host", host)
            .putInt("last_port", port)
            .putString("last_token", token)
            .apply()
    }

    private fun saveConnection(conn: SavedConnection) {
        val prefs = getSharedPreferences("terminal_remote", Context.MODE_PRIVATE)
        val json = prefs.getString("saved_connections", "[]")
        val type = object : TypeToken<MutableList<SavedConnection>>() {}.type
        val list: MutableList<SavedConnection> = gson.fromJson(json, type)
        list.removeAll { it.name == conn.name }
        list.add(conn)
        prefs.edit().putString("saved_connections", gson.toJson(list)).apply()
    }
}
