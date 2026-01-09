// Audio Panel - Per-clip volume and 10-band EQ controls with keyframe support

import { useEffect, useCallback } from 'react';
import { useTimelineStore } from '../../stores/timeline';
import { createEffectProperty } from '../../types';
import { EQ_FREQUENCIES } from '../../services/audioManager';
import './AudioPanel.css';

// EQ band parameter names
const EQ_BAND_PARAMS = ['band31', 'band62', 'band125', 'band250', 'band500', 'band1k', 'band2k', 'band4k', 'band8k', 'band16k'];

// Keyframe toggle button component
interface KeyframeToggleProps {
  clipId: string;
  effectId: string;
  paramName: string;
  value: number;
}

function KeyframeToggle({ clipId, effectId, paramName, value }: KeyframeToggleProps) {
  const { isRecording, toggleKeyframeRecording, hasKeyframes, addKeyframe } = useTimelineStore();

  const property = createEffectProperty(effectId, paramName);
  const recording = isRecording(clipId, property);
  const hasKfs = hasKeyframes(clipId, property);

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();

    if (!recording && !hasKfs) {
      // Turning ON for first time - create initial keyframe
      addKeyframe(clipId, property, value);
    }
    toggleKeyframeRecording(clipId, property);
  };

  return (
    <button
      className={`keyframe-toggle ${recording ? 'recording' : ''} ${hasKfs ? 'has-keyframes' : ''}`}
      onClick={handleClick}
      title={recording ? 'Stop recording keyframes' : hasKfs ? 'Enable keyframe recording' : 'Add keyframe'}
    >
      <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="13" r="7" />
        <line x1="12" y1="13" x2="12" y2="9" />
        <line x1="12" y1="2" x2="12" y2="5" />
        <line x1="9" y1="3" x2="15" y2="3" />
      </svg>
    </button>
  );
}

export function AudioPanel() {
  const {
    clips,
    tracks,
    selectedClipIds,
    addClipEffect,
    updateClipEffect,
    getInterpolatedEffects,
    playheadPosition,
  } = useTimelineStore();

  // Get first selected clip
  const selectedClipId = selectedClipIds.size > 0 ? [...selectedClipIds][0] : null;
  const selectedClip = clips.find((c) => c.id === selectedClipId);

  // Check if it's an audio clip
  const selectedTrack = selectedClip ? tracks.find(t => t.id === selectedClip.trackId) : null;
  const isAudioClip = selectedTrack?.type === 'audio';

  // Auto-add audio effects if they don't exist on an audio clip
  useEffect(() => {
    if (selectedClip && isAudioClip) {
      const hasVolumeEffect = selectedClip.effects?.some(e => e.type === 'audio-volume');
      const hasEQEffect = selectedClip.effects?.some(e => e.type === 'audio-eq');

      if (!hasVolumeEffect) {
        addClipEffect(selectedClip.id, 'audio-volume');
      }
      if (!hasEQEffect) {
        addClipEffect(selectedClip.id, 'audio-eq');
      }
    }
  }, [selectedClip?.id, isAudioClip, addClipEffect]);

  // Get interpolated effect values at current playhead
  const getEffectValues = useCallback(() => {
    if (!selectedClip) return { volume: 1, eqBands: EQ_BAND_PARAMS.map(() => 0) };

    const clipLocalTime = playheadPosition - selectedClip.startTime;
    const interpolatedEffects = getInterpolatedEffects(selectedClip.id, clipLocalTime);

    const volumeEffect = interpolatedEffects.find(e => e.type === 'audio-volume');
    const eqEffect = interpolatedEffects.find(e => e.type === 'audio-eq');

    return {
      volume: (volumeEffect?.params?.volume as number) ?? 1,
      eqBands: EQ_BAND_PARAMS.map(param => (eqEffect?.params?.[param] as number) ?? 0),
      volumeEffectId: volumeEffect?.id,
      eqEffectId: eqEffect?.id,
    };
  }, [selectedClip, playheadPosition, getInterpolatedEffects]);

  const effectValues = getEffectValues();

  // Format frequency label
  const formatFreq = (freq: number) => {
    return freq >= 1000 ? `${freq / 1000}k` : `${freq}`;
  };

  // Handle volume change
  const handleVolumeChange = (value: number) => {
    if (!selectedClip || !effectValues.volumeEffectId) return;
    updateClipEffect(selectedClip.id, effectValues.volumeEffectId, { volume: value });
  };

  // Handle EQ band change
  const handleEQChange = (bandIndex: number, value: number) => {
    if (!selectedClip || !effectValues.eqEffectId) return;
    const paramName = EQ_BAND_PARAMS[bandIndex];
    updateClipEffect(selectedClip.id, effectValues.eqEffectId, { [paramName]: value });
  };

  // Reset EQ to flat
  const handleResetEQ = () => {
    if (!selectedClip || !effectValues.eqEffectId) return;
    const resetParams: Record<string, number> = {};
    EQ_BAND_PARAMS.forEach(param => {
      resetParams[param] = 0;
    });
    updateClipEffect(selectedClip.id, effectValues.eqEffectId, resetParams);
  };

  // No audio clip selected
  if (!selectedClip || !isAudioClip) {
    return (
      <div className="audio-panel">
        <div className="audio-empty">
          <div className="audio-empty-icon">
            <svg viewBox="0 0 24 24" width="48" height="48" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M9 18V5l12-2v13" />
              <circle cx="6" cy="18" r="3" />
              <circle cx="18" cy="16" r="3" />
            </svg>
          </div>
          <div className="audio-empty-text">Select an audio clip to edit its audio properties</div>
        </div>
      </div>
    );
  }

  return (
    <div className="audio-panel">
      <div className="audio-clip-info">
        <span className="audio-clip-name">{selectedClip.name}</span>
      </div>

      {/* Volume Section */}
      <div className="audio-section">
        <div className="audio-section-header">
          <span className="audio-section-title">Volume</span>
          {effectValues.volumeEffectId && (
            <KeyframeToggle
              clipId={selectedClip.id}
              effectId={effectValues.volumeEffectId}
              paramName="volume"
              value={effectValues.volume}
            />
          )}
          <span className="audio-section-value">{Math.round(effectValues.volume * 100)}%</span>
        </div>
        <div className="volume-slider-container">
          <input
            type="range"
            className="volume-slider"
            min="0"
            max="2"
            step="0.01"
            value={effectValues.volume}
            onChange={(e) => handleVolumeChange(parseFloat(e.target.value))}
          />
        </div>
      </div>

      {/* 10-Band EQ Section */}
      <div className="audio-section">
        <div className="audio-section-header">
          <span className="audio-section-title">10-Band Equalizer</span>
          <button className="eq-reset-btn" onClick={handleResetEQ}>
            Reset
          </button>
        </div>

        <div className="eq-panel">
          {EQ_FREQUENCIES.map((freq, index) => (
            <div key={freq} className="eq-band-panel">
              <div className="eq-band-header">
                {effectValues.eqEffectId && (
                  <KeyframeToggle
                    clipId={selectedClip.id}
                    effectId={effectValues.eqEffectId}
                    paramName={EQ_BAND_PARAMS[index]}
                    value={effectValues.eqBands[index]}
                  />
                )}
              </div>
              <div className="eq-band-value">
                {effectValues.eqBands[index] > 0 ? '+' : ''}
                {effectValues.eqBands[index].toFixed(1)}
              </div>
              <div className="eq-band-slider-container">
                <input
                  type="range"
                  className="eq-band-slider"
                  min="-12"
                  max="12"
                  step="0.5"
                  value={effectValues.eqBands[index]}
                  onChange={(e) => handleEQChange(index, parseFloat(e.target.value))}
                  title={`${formatFreq(freq)}Hz: ${effectValues.eqBands[index].toFixed(1)}dB`}
                />
              </div>
              <div className="eq-band-label">{formatFreq(freq)}</div>
            </div>
          ))}
        </div>

        <div className="eq-scale">
          <span>+12dB</span>
          <span>0dB</span>
          <span>-12dB</span>
        </div>
      </div>
    </div>
  );
}
