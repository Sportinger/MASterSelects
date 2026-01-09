// Audio Panel - Master volume and 10-band EQ controls

import { useMixerStore } from '../../stores/mixerStore';
import { EQ_FREQUENCIES } from '../../services/audioManager';
import './AudioPanel.css';

export function AudioPanel() {
  const { masterVolume, setMasterVolume, eqBands, setEQBand, resetEQ } = useMixerStore();

  // Format frequency label
  const formatFreq = (freq: number) => {
    return freq >= 1000 ? `${freq / 1000}k` : `${freq}`;
  };

  return (
    <div className="audio-panel">
      {/* Master Volume Section */}
      <div className="audio-section">
        <div className="audio-section-header">
          <span className="audio-section-title">Master Volume</span>
          <span className="audio-section-value">{Math.round(masterVolume * 100)}%</span>
        </div>
        <div className="volume-slider-container">
          <input
            type="range"
            className="volume-slider"
            min="0"
            max="1"
            step="0.01"
            value={masterVolume}
            onChange={(e) => setMasterVolume(parseFloat(e.target.value))}
          />
        </div>
      </div>

      {/* 10-Band EQ Section */}
      <div className="audio-section">
        <div className="audio-section-header">
          <span className="audio-section-title">10-Band Equalizer</span>
          <button className="eq-reset-btn" onClick={resetEQ}>
            Reset
          </button>
        </div>

        <div className="eq-panel">
          {EQ_FREQUENCIES.map((freq, index) => (
            <div key={freq} className="eq-band-panel">
              <div className="eq-band-value">{eqBands[index] > 0 ? '+' : ''}{eqBands[index].toFixed(1)}</div>
              <div className="eq-band-slider-container">
                <input
                  type="range"
                  className="eq-band-slider"
                  min="-12"
                  max="12"
                  step="0.5"
                  value={eqBands[index]}
                  onChange={(e) => setEQBand(index, parseFloat(e.target.value))}
                  title={`${formatFreq(freq)}Hz: ${eqBands[index].toFixed(1)}dB`}
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

      {/* Audio Info */}
      <div className="audio-section">
        <div className="audio-info">
          <div className="audio-info-item">
            <span className="audio-info-label">Audio Engine</span>
            <span className="audio-info-value">Web Audio API</span>
          </div>
          <div className="audio-info-note">
            Volume and EQ affect all video playback in real-time.
            Connect audio sources to apply effects.
          </div>
        </div>
      </div>
    </div>
  );
}
