// MIDI input hook - simplified version (VJ layer control removed)

import { useEffect, useState, useCallback } from 'react';
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
  const [isEnabled, setIsEnabled] = useState(false);
  const [lastMessage, setLastMessage] = useState<{
    channel: number;
    control: number;
    value: number;
  } | null>(null);

  // Check MIDI support and request access
  useEffect(() => {
    if (!navigator.requestMIDIAccess) {
      setIsSupported(false);
      return;
    }

    setIsSupported(true);

    if (!isEnabled) return;

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
  }, [isEnabled]);

  // Handle MIDI messages (just log them for now - can be extended later)
  useEffect(() => {
    if (!midiAccess || !isEnabled) return;

    const handleMIDIMessage = (event: MIDIMessageEvent) => {
      const [status, control, value] = event.data as Uint8Array;
      const channel = status & 0x0f;
      const messageType = status & 0xf0;

      // Only handle Control Change messages (0xB0)
      if (messageType !== 0xb0) return;

      setLastMessage({ channel, control, value });
      // MIDI mappings can be implemented here in the future
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
  }, [midiAccess, isEnabled]);

  const enableMIDI = useCallback(() => {
    setIsEnabled(true);
  }, []);

  const disableMIDI = useCallback(() => {
    setIsEnabled(false);
    setMidiAccess(null);
    setDevices([]);
  }, []);

  return {
    isSupported,
    isEnabled,
    devices,
    lastMessage,
    enableMIDI,
    disableMIDI,
  };
}
