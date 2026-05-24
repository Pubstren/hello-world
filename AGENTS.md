# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Voice Temperature Remote — a pure-Python simulation of a voice-commanded temperature control system. No external services, databases, or Docker required.

### Dependencies

- **Required**: `numpy` (core numerical engine used by all modules)
- **Optional**: `librosa` (MFCC extraction on TX side), `sounddevice` + system `portaudio` (live microphone capture on TX side)
- Install via: `pip install numpy librosa`

### Running the application

The RX (receiver) pipeline is the primary testable entry point in a headless environment (TX requires a microphone):

```bash
python3 -m voice_temp_remote.main_rx packet.bin --templates voice_temp_remote/templates/templates.pkl --ebn0 10
```

To run this, you need test data files (`packet.bin` and `templates/templates.pkl`). These can be generated synthetically — see the README for the expected format.

### Lint

No formal lint config exists in the repo. Use `pyflakes voice_temp_remote/` for basic checks. Pre-existing warnings: unused imports of `List` and `Tuple` in `rx_recognize.py`.

### Testing

No test framework or test suite exists in the repo. Validate by importing all modules and running the RX pipeline with synthetic data.

### Gotchas

- Use `python3`, not `python` (the latter may not be aliased in this environment).
- `sounddevice` and `librosa` imports are guarded with try/except — the package runs without them if you only use the RX path.
- The `templates.pkl` file must be a pickle containing a `Dict[str, List[np.ndarray]]` mapping command names to lists of feature arrays (shape: `(n_frames, dim)`).
