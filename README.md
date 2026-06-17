# ComfyUI-LTX-AudioCaptioner

A ComfyUI custom node that transcribes audio from video files and generates descriptive audio captions for dataset preparation (e.g. LTX-2.3).

It uses a local Whisper model for transcription, optionally detects whether the audio is singing vs speaking, and combines everything into a caption ready for text encoding.

## Installation

1. Place this repository in your ComfyUI custom nodes directory:

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/nerdydude364/ComfyUI-LTX-AudioCaptioner.git
   ```

2. Install the Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Restart ComfyUI.

## Usage

Use the **LTX-2.3 Audio-Video Captioner (Local)** node from the *LTX-2.3 Dataset Tools* category.

### Inputs

| Input | Type | Default | Description |
|---|---|---|-|
| `video_path` | STRING | — | Path to the video file (required) |
| `trigger_name` | STRING | `character` | Name of the subject in the caption |
| `whisper_model_type` | STRING | `character` | Whisper model to use. Options: `base`, `tiny`, `small`, `medium`, `character` |
| `overwrite_existing` | BOOLEAN | `True` | Whether to overwrite an existing `.txt` caption file |
| `detect_singing` | BOOLEAN | `False` | Enable singing-vs-speaking detection heuristic. Requires audio analysis pass |
| `avg_segment_length_threshold` | FLOAT | `1.8` | Average phrase duration threshold for singing detection |
| `energy_variance_threshold` | FLOAT | `0.005` | RMS variance threshold for singing detection |
| `audio_channels` | STRING | `1` | Mono (1) or stereo (2) for audio extraction |
| `audio_sample_rate` | STRING | `16000` | Sample rate for audio extraction (8000, 16000, 22050, 44100, 48000) |
| `silence_rms_threshold` | FLOAT | `0.001` | RMS threshold below which audio is considered silent |
| `low_rms_threshold` | FLOAT | `0.01` | RMS threshold for distinguishing quiet vs ambient audio |
| `vocal_window_rms_threshold` | FLOAT | `0.015` | Minimum RMS to count a frame as "vocal" |
| `audio_window_ms` | INT | `30` | Frame/window size in milliseconds for RMS analysis |
| `visual_caption` (optional) | STRING | — | Pre-written visual description to combine with audio caption |

### Output

| Output | Type | Description |
|---|---|---|
| `final_caption` | STRING | Combined visual + audio caption, also written to `{video_path}.txt` |

### Caption format

When a video contains speech, the output looks like:

```
<visual caption>. <trigger_name> says/sings, "<transcribed text>".
```

When there is no speech, an ambient description is appended:

```
A video of <trigger_name>. The audio is accompanied by ...
```

## Requirements

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [openai-whisper](https://github.com/openai/whisper) (local transcription)
- [numpy](https://numpy.org/) (audio analysis)
- [ffmpeg](https://ffmpeg.org/) (audio extraction — must be installed on the system)

## License

Apache-2.0 — see [LICENSE](LICENSE) for details.
