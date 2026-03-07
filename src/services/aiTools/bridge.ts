/**
 * AI Tools Bridge - connects browser to Vite dev server via HMR
 * so external agents (Claude CLI) can execute aiTools via HTTP POST.
 *
 * Flow: POST /api/ai-tools → Vite server → HMR → browser → aiTools.execute() → HMR → HTTP response
 */
if (import.meta.hot) {
  import.meta.hot.on('ai-tools:execute', async (data: { requestId: string; tool: string; args: Record<string, unknown> }) => {
    try {
      const aiTools = (window as any).aiTools;
      if (!aiTools) {
        import.meta.hot!.send('ai-tools:result', {
          requestId: data.requestId,
          result: { success: false, error: 'aiTools not initialized yet' },
        });
        return;
      }

      let result: unknown;
      if (data.tool === '_list') {
        result = { success: true, data: aiTools.list() };
      } else if (data.tool === '_status') {
        result = { success: true, data: aiTools.status() };
      } else {
        result = await aiTools.execute(data.tool, data.args);
      }

      import.meta.hot!.send('ai-tools:result', {
        requestId: data.requestId,
        result,
      });
    } catch (error: unknown) {
      import.meta.hot!.send('ai-tools:result', {
        requestId: data.requestId,
        result: { success: false, error: error instanceof Error ? error.message : String(error) },
      });
    }
  });
}
