// Volume Tab - Audio volume and EQ controls
import { useEffect } from 'react';
import { useTimelineStore } from '../../../stores/timeline';
import { createEffectProperty } from '../../../types';
import { EQ_FREQUENCIES } from '../../../services/audioManager';
import { EffectKeyframeToggle, EQKeyframeToggle, EQ_BAND_PARAMS } from './shared';

interface VolumeTabProps {
  clipId: string;
  effects: Array<{ id: string; name: string; type: string; params: Record<string, number | boolean | string> }>;
}

export function VolumeTab({ clipId, effects }: VolumeTabProps) {
  // Reactive data - subscribe to specific values only
  const playheadPosition = useTimelineStore(state => state.playheadPosition);
  const clips = useTimelineStore(state => state.clips);
  // Actions from getState() - stable, no subscription needed
  const { setPropertyValue, getInterpolatedEffects, addClipEffect, setClipPreservesPitch } = useTimelineStore.getState();
  const clip = clips.find(c => c.id === clipId);
  const clipLocalTime = clip ? playheadPosition - clip.startTime : 0;
  const interpolatedEffects = getInterpolatedEffects(clipId, clipLocalTime);
  const preservesPitch = clip?.preservesPitch !== false; // default true

  // Auto-add audio effects if they don't exist
  useEffect(() => {
    const hasVolumeEffect = effects.some(e => e.type === 'audio-volume');
    const hasEQEffect = effects.some(e => e.type === 'audio-eq');
    if (!hasVolumeEffect) addClipEffect(clipId, 'audio-volume');
    if (!hasEQEffect) addClipEffect(clipId, 'audio-eq');
  }, [clipId, effects, addClipEffect]);

  // Get current values
  const volumeEffect = interpolatedEffects.find(e => e.type === 'audio-volume');
  const eqEffect = interpolatedEffects.find(e => e.type === 'audio-eq');
  const volume = (volumeEffect?.params?.volume as number) ?? 1;
  const eqBands = EQ_BAND_PARAMS.map(param => (eqEffect?.params?.[param] as number) ?? 0);

  const formatFreq = (freq: number) => freq >= 1000 ? `${freq / 1000}k` : `${freq}`;

  const handleVolumeChange = (value: number) => {
    if (!volumeEffect) return;
    const property = createEffectProperty(volumeEffect.id, 'volume');
    setPropertyValue(clipId, property, value);
  };

  const handleEQChange = (bandIndex: number, value: number) => {
    if (!eqEffect) return;
    const property = createEffectProperty(eqEffect.id, EQ_BAND_PARAMS[bandIndex]);
    setPropertyValue(clipId, property, value);
  };

  const handleResetEQ = () => {
    if (!eqEffect) return;
    EQ_BAND_PARAMS.forEach(param => {
      const property = createEffectProperty(eqEffect.id, param);
      setPropertyValue(clipId, property, 0);
    });
  };

  return (
    <div className="properties-tab-content volume-tab">
      {/* Volume Section */}
      <div className="properties-section">
        <div className="section-header-row">
          <h4>Volume</h4>
        </div>
        <div className="control-row">
          {volumeEffect && (
            <EffectKeyframeToggle clipId={clipId} effectId={volumeEffect.id} paramName="volume" value={volume} />
          )}
          <input type="range" min="0" max="2" step="0.01" value={volume}
            onChange={(e) => handleVolumeChange(parseFloat(e.target.value))} />
          <span className="value">{Math.round(volume * 100)}%</span>
        </div>
      </div>

      {/* Pitch Preservation Section */}
      <div className="properties-section">
        <div className="section-header-row">
          <h4>Speed Settings</h4>
        </div>
        <div className="control-row checkbox-row">
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={preservesPitch}
              onChange={(e) => setClipPreservesPitch(clipId, e.target.checked)}
            />
            <span>Keep Pitch</span>
          </label>
          <span className="hint">When speed changes, maintain original pitch</span>
        </div>
      </div>

      {/* 10-Band EQ Section */}
      <div className="properties-section eq-section">
        <div className="section-header-row">
          <h4>10-Band Equalizer</h4>
          {eqEffect && (
            <EQKeyframeToggle clipId={clipId} effectId={eqEffect.id} eqBands={eqBands} />
          )}
          <button className="btn btn-sm" onClick={handleResetEQ}>Reset</button>
        </div>

        <div className="eq-bands">
          {EQ_FREQUENCIES.map((freq, index) => (
            <div key={freq} className="eq-band">
              <div className="eq-band-value">
                {eqBands[index] > 0 ? '+' : ''}{eqBands[index].toFixed(1)}
              </div>
              <input type="range" className="eq-slider" min="-12" max="12" step="0.5"
                value={eqBands[index]} onChange={(e) => handleEQChange(index, parseFloat(e.target.value))}
                title={`${formatFreq(freq)}Hz: ${eqBands[index].toFixed(1)}dB`} />
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
