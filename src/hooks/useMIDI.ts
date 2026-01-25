// MIDI input hook for WebVJ Mixer

import { useEffect, useState, useCallback } from 'react';
import { useMixerStore } from '../stores/mixerStore';
import { Logger } from '../services/logger';

const log = Logger.create('MIDI');

interface MIDIDevice {
  id: string;
  name: string;
  manufacturer: string;
}

export function useMIDI() {
  const [midiAccess, setMidiAccess] = useState<MIDIAccess | null>(null);
  const [devices, setDevices] = useState<MIDIDevice[]>([]);
  const [isSupported, setIsSupported] = useState(false);
  const [lastMessage, setLastMessage] = useState<{
    channel: number;
    control: number;
    value: number;
  } | null>(null);

  const {
    midiEnabled,
    midiMappings,
    setMidiEnabled,
    layers,
    setLayerOpacity,
    setLayerVisibility,
  } = useMixerStore();

  // Check MIDI support and request access
  useEffect(() => {
    if (!navigator.requestMIDIAccess) {
      setIsSupported(false);
      return;
    }

    setIsSupported(true);

    if (!midiEnabled) return;

    navigator.requestMIDIAccess().then(
      (access) => {
        setMidiAccess(access);

        // Get input devices
        const inputDevices: MIDIDevice[] = [];
        access.inputs.forEach((input) => {
          inputDevices.push({
            id: input.id,
            name: input.name || 'Unknown',
            manufacturer: input.manufacturer || 'Unknown',
          });
        });
        setDevices(inputDevices);

        // Listen for device changes
        access.onstatechange = () => {
          const newDevices: MIDIDevice[] = [];
          access.inputs.forEach((input) => {
            newDevices.push({
              id: input.id,
              name: input.name || 'Unknown',
              manufacturer: input.manufacturer || 'Unknown',
            });
          });
          setDevices(newDevices);
        };
      },
      (error) => {
        log.error('MIDI access denied', error);
        setIsSupported(false);
      }
    );
  }, [midiEnabled]);

  // Handle MIDI messages
  useEffect(() => {
    if (!midiAccess || !midiEnabled) return;

    const handleMIDIMessage = (event: MIDIMessageEvent) => {
      const [status, control, value] = event.data as Uint8Array;
      const channel = status & 0x0f;
      const messageType = status & 0xf0;

      // Only handle Control Change messages (0xB0)
      if (messageType !== 0xb0) return;

      setLastMessage({ channel, control, value });

      // Apply MIDI mappings
      for (const mapping of midiMappings) {
        if (mapping.channel === channel && mapping.control === control) {
          const normalizedValue =
            mapping.min + (value / 127) * (mapping.max - mapping.min);

          // Parse target and apply
          const [targetType, layerIndex, property] = mapping.target.split('.');

          if (targetType === 'layer' && layerIndex !== undefined) {
            const layer = layers[parseInt(layerIndex, 10)];
            if (layer) {
              switch (property) {
                case 'opacity':
                  setLayerOpacity(layer.id, normalizedValue);
                  break;
                case 'visible':
                  setLayerVisibility(layer.id, value > 63);
                  break;
              }
            }
          }
        }
      }
    };

    // Attach listeners to all inputs
    midiAccess.inputs.forEach((input) => {
      input.onmidimessage = handleMIDIMessage;
    });

    return () => {
      midiAccess.inputs.forEach((input) => {
        input.onmidimessage = null;
      });
    };
  }, [midiAccess, midiEnabled, midiMappings, layers, setLayerOpacity, setLayerVisibility]);

  const enableMIDI = useCallback(() => {
    setMidiEnabled(true);
  }, [setMidiEnabled]);

  const disableMIDI = useCallback(() => {
    setMidiEnabled(false);
    setMidiAccess(null);
    setDevices([]);
  }, [setMidiEnabled]);

  return {
    isSupported,
    isEnabled: midiEnabled,
    devices,
    lastMessage,
    enableMIDI,
    disableMIDI,
  };
}
