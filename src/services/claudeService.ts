// Claude Service
// Interfaces with Claude API to generate edit decision lists

import type {
  MultiCamSource,
  MultiCamAnalysis,
  TranscriptEntry,
  EditDecision,
  EditStyle,
} from '../stores/multicamStore';
import { apiKeyManager } from './apiKeyManager';

// Style presets with editing instructions
const STYLE_PRESETS: Record<EditStyle, string> = {
  podcast: `Edit Style: Podcast
- Cut to the speaker when they start talking
- Use reaction shots sparingly (when listener has strong reaction)
- Prefer close-up shots when speaker makes important points
- Use wide shot for topic transitions or longer pauses
- Minimum cut duration: 3 seconds
- Avoid cutting mid-sentence unless absolutely necessary`,

  interview: `Edit Style: Interview
- Primarily show the person being asked questions
- Cut to interviewer only for questions or strong reactions
- Use wide shots to establish context
- Close-ups for emotional moments
- Minimum cut duration: 2 seconds
- Consider jump cuts for pacing when appropriate`,

  music: `Edit Style: Music Video
- Cut on the beat when possible
- Use motion to drive cut decisions
- Faster pacing, shorter cuts (1-2 seconds minimum)
- Prioritize visual interest over audio sync
- Use faces when singing, instruments when playing
- More dynamic camera switching`,

  documentary: `Edit Style: Documentary
- Let shots breathe - longer duration cuts (5+ seconds)
- B-roll for context and visual interest
- Cut on complete thoughts
- Use wide shots to establish scenes
- Close-ups for detail and emotion
- Follow the narrative arc`,

  custom: `Edit Style: Custom
Follow the user's custom instructions below.`,
};

interface GenerateEDLParams {
  cameras: MultiCamSource[];
  analysis: MultiCamAnalysis | null;
  transcript: TranscriptEntry[];
  editStyle: EditStyle;
  customPrompt?: string;
}

/**
 * Build a prompt for Claude to generate an EDL
 */
function buildPrompt(params: GenerateEDLParams): string {
  const { cameras, analysis, transcript, editStyle, customPrompt } = params;

  // Calculate duration
  const duration = analysis?.projectDuration ?? Math.max(...cameras.map(c => c.duration));
  const durationFormatted = formatDuration(duration);

  // Build camera info
  const cameraInfo = cameras
    .map((c, i) => `  Camera ${i + 1} (${c.id}): "${c.name}" - Role: ${c.role}`)
    .join('\n');

  // Build analysis summary
  let analysisSection = '';
  if (analysis) {
    analysisSection = `
ANALYSIS DATA:
Sample interval: ${analysis.sampleInterval}ms

Per-camera metrics (sampled data):
${analysis.cameras.map(cam => {
  const camera = cameras.find(c => c.id === cam.cameraId);
  const avgMotion = cam.frames.reduce((sum, f) => sum + f.motion, 0) / cam.frames.length;
  const avgSharpness = cam.frames.reduce((sum, f) => sum + f.sharpness, 0) / cam.frames.length;
  return `  ${camera?.name || cam.cameraId}:
    - Average motion: ${(avgMotion * 100).toFixed(1)}%
    - Average sharpness: ${(avgSharpness * 100).toFixed(1)}%
    - Frame count: ${cam.frames.length}`;
}).join('\n')}

Timeline data (motion/sharpness per camera at each timestamp):
${formatTimelineData(analysis, cameras).slice(0, 5000)}... [truncated]
`;
  }

  // Build transcript section
  let transcriptSection = '';
  if (transcript.length > 0) {
    transcriptSection = `
TRANSCRIPT:
${transcript.map(t =>
  `[${formatDuration(t.start)} - ${formatDuration(t.end)}] ${t.speaker}: "${t.text}"`
).join('\n')}
`;
  }

  // Build style section
  const styleInstructions = STYLE_PRESETS[editStyle];
  const customInstructions = customPrompt ? `\nCUSTOM INSTRUCTIONS:\n${customPrompt}` : '';

  return `You are an expert video editor. Generate an edit decision list (EDL) for a multicam video.

PROJECT INFORMATION:
- Total duration: ${durationFormatted}
- Number of cameras: ${cameras.length}

CAMERAS:
${cameraInfo}

${styleInstructions}
${customInstructions}
${analysisSection}
${transcriptSection}

TASK:
Generate an EDL (edit decision list) that specifies which camera to show at each moment.
Consider:
1. Speaker tracking (show who's talking)
2. Motion and sharpness (prefer sharper, less shaky shots)
3. Visual variety (don't stay on one camera too long)
4. The specified edit style

OUTPUT FORMAT:
Return a JSON array of edit decisions. Each decision has:
- start: timestamp in milliseconds
- end: timestamp in milliseconds
- cameraId: the camera ID to use (from the list above)
- reason: brief explanation for this cut decision

Example:
[
  {"start": 0, "end": 5000, "cameraId": "cam-123", "reason": "Opening wide shot"},
  {"start": 5000, "end": 12000, "cameraId": "cam-456", "reason": "Speaker A talking"},
  ...
]

IMPORTANT:
- Decisions must cover the entire duration without gaps
- Each decision's end must equal the next decision's start
- Use actual camera IDs from the provided list
- Return ONLY the JSON array, no other text

Generate the EDL now:`;
}

/**
 * Format timeline analysis data for the prompt
 */
function formatTimelineData(analysis: MultiCamAnalysis, cameras: MultiCamSource[]): string {
  const lines: string[] = [];
  const maxEntries = 100; // Limit to avoid token limits

  const step = Math.max(1, Math.floor(analysis.cameras[0]?.frames.length / maxEntries));

  for (let i = 0; i < (analysis.cameras[0]?.frames.length ?? 0); i += step) {
    const timestamp = i * analysis.sampleInterval;
    let line = `${formatDuration(timestamp)}: `;

    const cameraData = analysis.cameras.map(cam => {
      const frame = cam.frames[i];
      const camera = cameras.find(c => c.id === cam.cameraId);
      return `${camera?.name?.substring(0, 8) || 'Cam'}(M:${(frame?.motion * 100).toFixed(0)}%,S:${(frame?.sharpness * 100).toFixed(0)}%)`;
    });

    line += cameraData.join(' | ');

    // Add audio level
    const audioLevel = analysis.audioLevels[i]?.level ?? 0;
    line += ` | Audio: ${(audioLevel * 100).toFixed(0)}%`;

    lines.push(line);
  }

  return lines.join('\n');
}

/**
 * Format duration in milliseconds to MM:SS format
 */
function formatDuration(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

/**
 * Parse Claude's response into EditDecision array
 */
function parseEDLResponse(response: string, cameras: MultiCamSource[]): EditDecision[] {
  // Extract JSON from response
  const jsonMatch = response.match(/\[[\s\S]*\]/);
  if (!jsonMatch) {
    throw new Error('Could not parse EDL response - no JSON array found');
  }

  try {
    const parsed = JSON.parse(jsonMatch[0]);

    if (!Array.isArray(parsed)) {
      throw new Error('Response is not an array');
    }

    // Validate and convert to EditDecision
    return parsed.map((item: any, index: number) => {
      // Validate required fields
      if (typeof item.start !== 'number' || typeof item.end !== 'number') {
        throw new Error(`Invalid timestamps in decision ${index}`);
      }

      // Validate camera ID exists
      const validCameraId = cameras.some(c => c.id === item.cameraId);
      const cameraId = validCameraId ? item.cameraId : cameras[0]?.id;

      return {
        id: `edl-${index}`,
        start: item.start,
        end: item.end,
        cameraId,
        reason: item.reason || undefined,
        confidence: item.confidence,
      };
    });
  } catch (error) {
    throw new Error(`Failed to parse EDL JSON: ${error}`);
  }
}

class ClaudeService {
  private apiEndpoint = 'https://api.anthropic.com/v1/messages';

  /**
   * Generate an EDL using Claude
   */
  async generateEDL(params: GenerateEDLParams): Promise<EditDecision[]> {
    const apiKey = await apiKeyManager.getKey();
    if (!apiKey) {
      throw new Error('API key not configured');
    }

    const prompt = buildPrompt(params);
    console.log('[ClaudeService] Generating EDL...');

    try {
      const response = await fetch(this.apiEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': apiKey,
          'anthropic-version': '2023-06-01',
          'anthropic-dangerous-direct-browser-access': 'true',
        },
        body: JSON.stringify({
          model: 'claude-sonnet-4-20250514',
          max_tokens: 4096,
          messages: [
            {
              role: 'user',
              content: prompt,
            },
          ],
        }),
      });

      if (!response.ok) {
        const errorBody = await response.text();
        console.error('[ClaudeService] API error:', response.status, errorBody);

        if (response.status === 401) {
          throw new Error('Invalid API key. Please check your Claude API key in settings.');
        } else if (response.status === 429) {
          throw new Error('Rate limited. Please wait a moment and try again.');
        } else {
          throw new Error(`API request failed: ${response.status}`);
        }
      }

      const data = await response.json();
      const content = data.content?.[0]?.text;

      if (!content) {
        throw new Error('Empty response from Claude');
      }

      console.log('[ClaudeService] Received response, parsing EDL...');
      const edl = parseEDLResponse(content, params.cameras);

      console.log('[ClaudeService] Generated', edl.length, 'edit decisions');
      return edl;
    } catch (error) {
      console.error('[ClaudeService] Failed to generate EDL:', error);
      throw error;
    }
  }
}

// Singleton instance
export const claudeService = new ClaudeService();
