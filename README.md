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

## Binance realtime altcoin books and trades

This repository also includes an asyncio-based Binance Spot market data program that
loads altcoin symbols from a TXT file, maintains local order books from Binance diff
depth streams, and keeps raw trade events in a bounded cache.

Install the runtime dependency:

```bash
python3 -m pip install -r requirements.txt
```

Create a symbols file with one coin or pair per line:

```text
DOGE
PEPE
SOL/USDT
LINKUSDT
```

Run the collector:

```bash
python3 -m binance_realtime examples/altcoins.txt --output-dir data/binance
```

Important implementation details:

- A bare coin such as `DOGE` is normalized to `DOGEUSDT` by default. Use
  `--quote USDC` or provide an explicit pair such as `DOGE/USDT` to change this.
- For each symbol the client subscribes to `<symbol>@depth@100ms` and
  `<symbol>@trade` through Binance combined streams.
- Order books follow Binance's documented local book procedure: buffer websocket
  depth deltas, fetch `/api/v3/depth`, discard stale deltas, apply the first event
  that bridges `lastUpdateId + 1`, and reconnect/resync if an update gap is found.
- Symbols are split across multiple websocket connections with
  `--max-symbols-per-connection` and REST snapshots are throttled with
  `--snapshot-concurrency` for larger watchlists.
- `--output-dir` writes raw trades to `trades.jsonl` and periodic best bid/ask
  summaries to `book_tops.jsonl`. State maintenance continues even if the bounded
  writer queue drops output records under extreme disk pressure; dropped counts are
  included in periodic reports.
