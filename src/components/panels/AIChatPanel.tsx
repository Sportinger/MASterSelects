// AI Chat Panel - Simple chat interface using OpenAI API

import { useState, useCallback, useRef, useEffect } from 'react';
import { useSettingsStore } from '../../stores/settingsStore';
import './AIChatPanel.css';

// Available OpenAI models
const OPENAI_MODELS = [
  { id: 'gpt-4o', name: 'GPT-4o' },
  { id: 'gpt-4o-mini', name: 'GPT-4o Mini' },
  { id: 'gpt-4-turbo', name: 'GPT-4 Turbo' },
  { id: 'gpt-3.5-turbo', name: 'GPT-3.5 Turbo' },
];

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export function AIChatPanel() {
  const { apiKeys, openSettings } = useSettingsStore();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [model, setModel] = useState('gpt-4o-mini');
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Check if API key is available
  const hasApiKey = !!apiKeys.openai;

  // Send message to OpenAI
  const sendMessage = useCallback(async () => {
    if (!input.trim() || !hasApiKey || isLoading) return;

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setError(null);
    setIsLoading(true);

    try {
      const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKeys.openai}`,
        },
        body: JSON.stringify({
          model,
          messages: [
            ...messages.map(m => ({ role: m.role, content: m.content })),
            { role: 'user', content: userMessage.content },
          ],
          max_tokens: 2048,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error?.message || `API error: ${response.status}`);
      }

      const data = await response.json();
      const assistantContent = data.choices?.[0]?.message?.content || 'No response';

      const assistantMessage: Message = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: assistantContent,
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send message');
    } finally {
      setIsLoading(false);
    }
  }, [input, hasApiKey, isLoading, model, messages, apiKeys.openai]);

  // Handle key press
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }, [sendMessage]);

  // Clear chat
  const clearChat = useCallback(() => {
    setMessages([]);
    setError(null);
  }, []);

  // Render empty state if no API key
  if (!hasApiKey) {
    return (
      <div className="ai-chat-panel">
        <div className="ai-chat-header">
          <h2>AI Chat</h2>
        </div>
        <div className="ai-chat-empty">
          <div className="ai-chat-no-key">
            <span className="no-key-icon">🔑</span>
            <p>OpenAI API key required</p>
            <button className="btn-settings" onClick={openSettings}>
              Open Settings
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="ai-chat-panel">
      {/* Header */}
      <div className="ai-chat-header">
        <h2>AI Chat</h2>
        <div className="ai-chat-controls">
          <select
            className="model-select"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            disabled={isLoading}
          >
            {OPENAI_MODELS.map(m => (
              <option key={m.id} value={m.id}>{m.name}</option>
            ))}
          </select>
          <button
            className="btn-clear"
            onClick={clearChat}
            disabled={isLoading || messages.length === 0}
            title="Clear chat"
          >
            Clear
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="ai-chat-messages">
        {messages.length === 0 ? (
          <div className="ai-chat-welcome">
            <p>Start a conversation with AI</p>
            <span className="welcome-hint">Using {OPENAI_MODELS.find(m => m.id === model)?.name}</span>
          </div>
        ) : (
          messages.map(msg => (
            <div key={msg.id} className={`ai-chat-message ${msg.role}`}>
              <div className="message-header">
                <span className="message-role">{msg.role === 'user' ? 'You' : 'AI'}</span>
                <span className="message-time">
                  {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>
              </div>
              <div className="message-content">
                {msg.content.split('\n').map((line, i) => (
                  <p key={i}>{line || '\u00A0'}</p>
                ))}
              </div>
            </div>
          ))
        )}
        {isLoading && (
          <div className="ai-chat-message assistant loading">
            <div className="message-header">
              <span className="message-role">AI</span>
            </div>
            <div className="message-content">
              <span className="typing-indicator">
                <span></span><span></span><span></span>
              </span>
            </div>
          </div>
        )}
        {error && (
          <div className="ai-chat-error">
            <span className="error-icon">⚠️</span>
            {error}
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="ai-chat-input-area">
        <textarea
          ref={inputRef}
          className="ai-chat-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message... (Enter to send)"
          disabled={isLoading}
          rows={2}
        />
        <button
          className="btn-send"
          onClick={sendMessage}
          disabled={!input.trim() || isLoading}
        >
          {isLoading ? '...' : 'Send'}
        </button>
      </div>
    </div>
  );
}
