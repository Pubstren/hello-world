# Voice Temperature Remote

Python-based reference implementation for a **MFCC + DTW** speech command link driving a
black-box temperature plant with **PID + mode state machine** control.

## Module layout

```
voice_temp_remote/
  config.py          # system configuration constants
  tx_capture.py      # audio capture + simple VAD
  mfcc_feat.py       # MFCC + CMVN + quantization
  packet.py          # packet framing + CRC16
  channel_sim.py     # BPSK + AWGN channel
  rx_recognize.py    # DTW template matcher
  plant_temp.py      # temperature plant (black-box internal state)
  controller.py      # PID + mode/state machine
  main_tx.py         # sender entry point
  main_rx.py         # receiver entry point
  templates/         # pickled template features
```

## Quick start

1. Capture a command and build a packet:

```bash
python -m voice_temp_remote.main_tx --duration 1.0
```

2. Run the receiver with a saved packet and templates file:

```bash
python -m voice_temp_remote.main_rx packet.bin --templates templates/templates.pkl --ebn0 10
```

## Notes

- MFCC extraction uses `librosa`.
- Audio capture uses `sounddevice`.
- Templates are expected to be stored as a pickle with a dictionary mapping command names
  to lists of feature arrays.
