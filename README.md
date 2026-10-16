[![Pylint](https://github.com/formeo/audiobook-cleaner/actions/workflows/pylint.yml/badge.svg)](https://github.com/formeo/audiobook-cleaner/actions/workflows/pylint.yml)
![License](https://img.shields.io/badge/license-MIT-green)
[![Latest Release](https://img.shields.io/github/v/release/formeo/audiobook-cleaner?style=flat-square)](https://github.com/formeo/audiobook-cleaner/releases)
[![Language](https://img.shields.io/github/languages/top/formeo/audiobook-cleaner?style=flat-square)]()
[![Downloads](https://img.shields.io/github/downloads/formeo/audiobook-cleaner/total?style=flat-square&label=Downloads)](https://github.com/formeo/audiobook-cleaner/releases)
# Audiobook Cleaner

Clean audiobooks from background music and noise using MDX-Net, VR, and Roformer neural network models.

Python wrapper over [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) — the same models used by UVR5, but without issues with large files and 32-bit architecture.

## Features

- Separate narrator's voice from background music
- Remove noise (hiss, hum, interference)
- Remove echo and reverberation
- Automatic splitting of large files
- Batch processing of directories
- Works on CPU and GPU (NVIDIA CUDA)
- Supported formats: MP3, WAV, FLAC, M4A, OGG, OPUS

## Installation

### Basic Installation

```bash
pip install audio-separator[cpu]
```

### FFmpeg (required)

```bash
# Windows (winget)
winget install ffmpeg

# Windows (chocolatey)
choco install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

### GPU Installation (NVIDIA, recommended)

For 2-3x speed boost on NVIDIA GPUs:

```bash
# 1. PyTorch with CUDA
pip uninstall torch torchaudio torchvision -y
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. ONNX Runtime with CUDA
pip install onnxruntime-gpu==1.18.1

# 3. cuDNN (required for ONNX Runtime)
pip install nvidia-cudnn-cu12==8.9.7.29
```

### GPU Verification

```bash
# PyTorch sees CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# ONNX Runtime sees CUDA
python -c "import onnxruntime as ort; print('Providers:', ort.get_available_providers())"
```

Expected output:
- `CUDA: True NVIDIA GeForce ...`
- `Providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`

## Usage

### Basic

```bash
# Process single file
python audiobook_cleaner.py audiobook.mp3

# Process directory
python audiobook_cleaner.py --dir /path/to/audiobooks
```

### Large Files (recommended)

For files >100 MB, use chunking:

```bash
# Split into 5-minute chunks (recommended)
python audiobook_cleaner.py audiobook.mp3 --chunk-duration 300 --no-denoise

# Split into 10-minute chunks (if you have plenty of RAM)
python audiobook_cleaner.py audiobook.mp3 --chunk-duration 600 --no-denoise
```

The script will automatically split the file, process chunks, and merge the result.

### Model Selection

```bash
# List available models
python audiobook_cleaner.py --list-models

# Use specific model
python audiobook_cleaner.py audiobook.mp3 --model vocals_kim
```

### All Parameters

```
Arguments:
  input                   Input audio file
  --dir, -d               Process all files in directory
  --output, -o            Output directory
  --model, -m             Separation model (default: vocals)
  --no-denoise            Skip additional denoising
  --cpu                   Force CPU usage
  --format, -f            Output format: mp3, wav, flac (default: mp3)
  --sample-rate, -sr      Sample rate (default: 44100)
  --segment-size, -s      Segment size in seconds (default: 64)
  --chunk-duration, -c    Split large files by N seconds (default: 0 = off)
  --list-models, -l       Show available models
```

## Models

### Voice Separation from Music

| Shortcut | Model | Description |
|---------|--------|----------|
| `vocals` | UVR-MDX-NET-Voc_FT.onnx | Default, good balance |
| `vocals_kim` | Kim_Vocal_2.onnx | Alternative, sometimes better |
| `roformer` | BS-Roformer | Best quality, but slower and needs more VRAM |

### Noise Removal

| Shortcut | Model | Description |
|---------|--------|----------|
| `denoise` | UVR-DeNoise.pth | Standard denoising |
| `denoise_lite` | UVR-DeNoise-Lite.pth | Lightweight version |
| `roformer_denoise` | Mel-Roformer-Denoise | Best quality |

### Echo/Reverb Removal

| Shortcut | Model | Description |
|---------|--------|----------|
| `dereverb` | UVR-DeEcho-DeReverb.pth | Echo + reverb |
| `dereverb_normal` | UVR-De-Echo-Normal.pth | Echo only |

## Examples

```bash
# Quick processing without denoising
python audiobook_cleaner.py book.mp3 --no-denoise

# Large file — split into 5-minute chunks
python audiobook_cleaner.py book.mp3 --chunk-duration 300 --no-denoise

# Maximum quality (slow)
python audiobook_cleaner.py book.mp3 --model roformer --chunk-duration 300

# Processing on low-end machine
python audiobook_cleaner.py book.mp3 --chunk-duration 120 --segment-size 16 --cpu --no-denoise

# Batch processing to WAV
python audiobook_cleaner.py --dir ./books --output ./clean --format wav

# Clean cockpit voice recorder (CVR) from noise
python audiobook_cleaner.py cvr.mp3 --model denoise --no-denoise
```

## Processing Speed

Approximate processing time for 5-minute chunk (300 sec):

| Configuration | Time | Notes |
|--------------|-------|------------|
| CPU (i7-8700) | 2:30 | No GPU |
| GPU GTX 1660 Super (PyTorch) | 1:40 | ONNX → PyTorch conversion |
| GPU GTX 1660 Super (ONNX) | 0:30-0:40 | Full acceleration |

For 7-hour audiobook (85 chunks of 5 min):
- CPU: ~3.5 hours
- GPU (PyTorch): ~2.5 hours
- GPU (ONNX): ~45-60 minutes

## How It Works

1. **Step 1**: MDX-Net/Roformer model separates audio into "vocals" and "instrumental"
2. **Step 2** (optional): DeNoise model removes residual noise

Output files:
- `*_(Vocals).mp3` — clean narrator's voice
- `*_(Instrumental).mp3` — separated music/noise

## Requirements

- Python 3.9+
- FFmpeg
- 8+ GB RAM (16 GB recommended for large files)
- NVIDIA GPU with 4+ GB VRAM (optional, for acceleration)

## Troubleshooting

### "bad allocation" / Out of memory

```bash
# Use chunking + small segment size
python audiobook_cleaner.py book.mp3 --chunk-duration 300 --segment-size 16 --no-denoise
```

### "Model file not found"

Models are downloaded automatically. Check your internet connection.

### GPU not being used

1. Verify installation:
```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

2. If `CUDAExecutionProvider` is missing:
```bash
pip uninstall onnxruntime onnxruntime-gpu -y
pip install onnxruntime-gpu==1.18.1
pip install nvidia-cudnn-cu12==8.9.7.29
```

### "CUDAExecutionProvider not available" 

cuDNN is missing. Install:
```bash
pip install nvidia-cudnn-cu12==8.9.7.29
```

### "LoadLibrary failed with error 126"

ONNX Runtime can't find CUDA DLLs. Options:
1. Install cuDNN via pip (see above)
2. Or install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) system-wide

### Slow processing even with GPU

If you see in logs:
```
Model converted from onnx to pytorch due to segment size not matching dim_t
```

The model is being converted to PyTorch, which is slower. This is normal for some configurations.

## License

MIT

## Credits

- [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
- [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
- Authors of MDX-Net, Demucs, and Roformer models
