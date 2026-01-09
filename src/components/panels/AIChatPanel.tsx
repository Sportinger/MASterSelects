// AI Chat Panel - Chat interface with timeline editing tools using OpenAI API

import { useState, useCallback, useRef, useEffect } from 'react';
import { useSettingsStore } from '../../stores/settingsStore';
import { AI_TOOLS, executeAITool, getQuickTimelineSummary } from '../../services/aiTools';
import './AIChatPanel.css';

// Available OpenAI models
const OPENAI_MODELS = [
  // GPT-5.2 series (newest - Dec 2025)
  { id: 'gpt-5.2', name: 'GPT-5.2 (Thinking)' },
  { id: 'gpt-5.2-pro', name: 'GPT-5.2 Pro' },
  // GPT-5.1 series
  { id: 'gpt-5.1', name: 'GPT-5.1' },
  { id: 'gpt-5.1-codex', name: 'GPT-5.1 Codex' },
  { id: 'gpt-5.1-codex-mini', name: 'GPT-5.1 Codex Mini' },
  // GPT-5 series
  { id: 'gpt-5', name: 'GPT-5' },
  { id: 'gpt-5-mini', name: 'GPT-5 Mini' },
  { id: 'gpt-5-nano', name: 'GPT-5 Nano' },
  // Reasoning models
  { id: 'o3', name: 'o3 (Reasoning)' },
  { id: 'o4-mini', name: 'o4-mini (Reasoning)' },
  { id: 'o3-pro', name: 'o3-pro (Deep Reasoning)' },
  // GPT-4.1 series
  { id: 'gpt-4.1', name: 'GPT-4.1' },
  { id: 'gpt-4.1-mini', name: 'GPT-4.1 Mini' },
  { id: 'gpt-4.1-nano', name: 'GPT-4.1 Nano' },
  // GPT-4o series (legacy)
  { id: 'gpt-4o', name: 'GPT-4o' },
  { id: 'gpt-4o-mini', name: 'GPT-4o Mini' },
];

// System prompt for editor mode
const EDITOR_SYSTEM_PROMPT = `You are an AI video editing assistant with direct access to the timeline. You can:

- View and analyze the timeline state (tracks, clips, playhead position)
- Get detailed clip information including analysis data and transcripts
- Split, delete, move, and trim clips
- Create and manage video/audio tracks
- Capture frames from the composition
- Find silent sections in clips based on transcripts

CRITICAL RULES - FOLLOW EXACTLY:
1. ALWAYS assume the user means the CURRENTLY SELECTED CLIP. Never ask "which clip?" - just use the selected one.
2. ONLY work within the VISIBLE RANGE of the clip on the timeline (from clip.startTime to clip.startTime + clip.duration).
   - Analysis data covers the full source file, but you must FILTER to only the visible/trimmed portion.
   - If a clip is trimmed (inPoint/outPoint), only consider data within that range.
3. DO NOT ask for clarification. Make reasonable assumptions and proceed with the action.
4. When the user asks to "cut" something, use splitClip to divide clips and deleteClip to remove unwanted sections.
5. Be precise with time values - they are in seconds.
6. When making multiple edits, process them in order from END to START of timeline to avoid position shifts.
7. When using findLowQualitySections or similar tools, FILTER the results to only include sections within the clip's visible timeline range.

Current timeline summary: `;

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'tool';
  content: string;
  timestamp: Date;
  toolCalls?: ToolCall[];
  toolName?: string;
  isToolResult?: boolean;
}

interface ToolCall {
  id: string;
  name: string;
  arguments: string;
}

interface APIMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string | null;
  tool_calls?: Array<{
    id: string;
    type: 'function';
    function: { name: string; arguments: string };
  }>;
  tool_call_id?: string;
}

export function AIChatPanel() {
  const { apiKeys, openSettings } = useSettingsStore();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [model, setModel] = useState('gpt-5.1');
  const [error, setError] = useState<string | null>(null);
  const [editorMode, setEditorMode] = useState(true); // Enable tools by default
  const [currentToolAction, setCurrentToolAction] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, currentToolAction]);

  // Check if API key is available
  const hasApiKey = !!apiKeys.openai;

  // Build API messages from chat history
  const buildAPIMessages = useCallback((userContent: string): APIMessage[] => {
    const apiMessages: APIMessage[] = [];

    // Add system prompt in editor mode
    if (editorMode) {
      apiMessages.push({
        role: 'system',
        content: EDITOR_SYSTEM_PROMPT + getQuickTimelineSummary(),
      });
    }

    // Add conversation history
    for (const msg of messages) {
      if (msg.role === 'user') {
        apiMessages.push({ role: 'user', content: msg.content });
      } else if (msg.role === 'assistant') {
        if (msg.toolCalls && msg.toolCalls.length > 0) {
          apiMessages.push({
            role: 'assistant',
            content: msg.content || null,
            tool_calls: msg.toolCalls.map(tc => ({
              id: tc.id,
              type: 'function' as const,
              function: { name: tc.name, arguments: tc.arguments },
            })),
          });
        } else {
          apiMessages.push({ role: 'assistant', content: msg.content });
        }
      } else if (msg.role === 'tool' && msg.toolName) {
        apiMessages.push({
          role: 'tool',
          content: msg.content,
          tool_call_id: msg.id,
        });
      }
    }

    // Add new user message
    apiMessages.push({ role: 'user', content: userContent });

    return apiMessages;
  }, [messages, editorMode]);

  // Call OpenAI API
  const callOpenAI = useCallback(async (apiMessages: APIMessage[]): Promise<{
    content: string | null;
    toolCalls: ToolCall[];
  }> => {
    // Newer models (GPT-5.x, o3, o4) use max_completion_tokens instead of max_tokens
    const isNewerModel = model.startsWith('gpt-5') || model.startsWith('o3') || model.startsWith('o4');

    const requestBody: Record<string, unknown> = {
      model,
      messages: apiMessages,
      ...(isNewerModel
        ? { max_completion_tokens: 4096 }
        : { max_tokens: 4096 }),
    };

    // Add tools in editor mode
    if (editorMode) {
      requestBody.tools = AI_TOOLS;
      requestBody.tool_choice = 'auto';
    }

    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKeys.openai}`,
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error?.message || `API error: ${response.status}`);
    }

    const data = await response.json();
    const choice = data.choices?.[0];

    const toolCalls: ToolCall[] = (choice?.message?.tool_calls || []).map((tc: {
      id: string;
      function: { name: string; arguments: string };
    }) => ({
      id: tc.id,
      name: tc.function.name,
      arguments: tc.function.arguments,
    }));

    return {
      content: choice?.message?.content || null,
      toolCalls,
    };
  }, [model, editorMode, apiKeys.openai]);

  // Send message to OpenAI (with tool calling loop)
  const sendMessage = useCallback(async () => {
    if (!input.trim() || !hasApiKey || isLoading) return;

    const userContent = input.trim();
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: userContent,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setError(null);
    setIsLoading(true);

    try {
      const apiMessages = buildAPIMessages(userContent);
      let iterationCount = 0;
      const maxIterations = 10; // Prevent infinite loops

      while (iterationCount < maxIterations) {
        iterationCount++;

        const { content, toolCalls } = await callOpenAI(apiMessages);

        if (toolCalls.length === 0) {
          // No tool calls - add final assistant message
          if (content) {
            const assistantMessage: Message = {
              id: `assistant-${Date.now()}`,
              role: 'assistant',
              content,
              timestamp: new Date(),
            };
            setMessages(prev => [...prev, assistantMessage]);
          }
          break;
        }

        // Handle tool calls
        const assistantMessage: Message = {
          id: `assistant-${Date.now()}-${iterationCount}`,
          role: 'assistant',
          content: content || '',
          timestamp: new Date(),
          toolCalls,
        };
        setMessages(prev => [...prev, assistantMessage]);

        // Add assistant message to API messages
        apiMessages.push({
          role: 'assistant',
          content: content || null,
          tool_calls: toolCalls.map(tc => ({
            id: tc.id,
            type: 'function' as const,
            function: { name: tc.name, arguments: tc.arguments },
          })),
        });

        // Execute each tool call
        for (const toolCall of toolCalls) {
          setCurrentToolAction(`Executing: ${toolCall.name}`);

          let args: Record<string, unknown> = {};
          try {
            args = JSON.parse(toolCall.arguments);
          } catch {
            args = {};
          }

          const result = await executeAITool(toolCall.name, args);

          const toolResultMessage: Message = {
            id: toolCall.id,
            role: 'tool',
            content: JSON.stringify(result, null, 2),
            timestamp: new Date(),
            toolName: toolCall.name,
            isToolResult: true,
          };
          setMessages(prev => [...prev, toolResultMessage]);

          // Add tool result to API messages
          apiMessages.push({
            role: 'tool',
            content: JSON.stringify(result),
            tool_call_id: toolCall.id,
          });
        }

        setCurrentToolAction(null);
      }

      if (iterationCount >= maxIterations) {
        setError('Too many tool iterations - stopping to prevent infinite loop');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send message');
    } finally {
      setIsLoading(false);
      setCurrentToolAction(null);
    }
  }, [input, hasApiKey, isLoading, buildAPIMessages, callOpenAI]);

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
          <h2>AI Editor</h2>
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
        <h2>AI Editor</h2>
        <div className="ai-chat-controls">
          <label className="editor-mode-toggle" title="Enable timeline editing tools">
            <input
              type="checkbox"
              checked={editorMode}
              onChange={(e) => setEditorMode(e.target.checked)}
              disabled={isLoading}
            />
            <span className="toggle-label">Tools</span>
          </label>
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
            <p>{editorMode ? 'AI Editor Ready' : 'Start a conversation'}</p>
            <span className="welcome-hint">
              {editorMode
                ? 'Ask me to edit your timeline - cut clips, remove silence, etc.'
                : `Using ${OPENAI_MODELS.find(m => m.id === model)?.name}`}
            </span>
          </div>
        ) : (
          messages.map(msg => {
            // Tool result messages - show compact
            if (msg.isToolResult) {
              return (
                <div key={msg.id} className="ai-chat-message tool-result">
                  <div className="tool-result-header">
                    <span className="tool-icon">🔧</span>
                    <span className="tool-name">{msg.toolName}</span>
                  </div>
                  <pre className="tool-result-content">
                    {msg.content.length > 500
                      ? msg.content.substring(0, 500) + '...'
                      : msg.content}
                  </pre>
                </div>
              );
            }

            // Assistant message with tool calls
            if (msg.role === 'assistant' && msg.toolCalls && msg.toolCalls.length > 0) {
              return (
                <div key={msg.id} className="ai-chat-message assistant">
                  <div className="message-header">
                    <span className="message-role">AI</span>
                    <span className="message-time">
                      {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                  </div>
                  {msg.content && (
                    <div className="message-content">
                      {msg.content.split('\n').map((line, i) => (
                        <p key={i}>{line || '\u00A0'}</p>
                      ))}
                    </div>
                  )}
                  <div className="tool-calls">
                    {msg.toolCalls.map(tc => (
                      <div key={tc.id} className="tool-call">
                        <span className="tool-call-name">{tc.name}</span>
                        <span className="tool-call-args">
                          {tc.arguments.length > 100
                            ? tc.arguments.substring(0, 100) + '...'
                            : tc.arguments}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              );
            }

            // Regular user/assistant message
            return (
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
            );
          })
        )}
        {isLoading && (
          <div className="ai-chat-message assistant loading">
            <div className="message-header">
              <span className="message-role">AI</span>
            </div>
            <div className="message-content">
              {currentToolAction ? (
                <span className="tool-action">{currentToolAction}</span>
              ) : (
                <span className="typing-indicator">
                  <span></span><span></span><span></span>
                </span>
              )}
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
          placeholder={editorMode
            ? "e.g., 'Remove all silent parts' or 'Split clip at 5 seconds'"
            : "Type a message... (Enter to send)"}
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
