# GeneFacePlusPlus - Boshen's Enhanced Fork

> **Original Repository**: [yerfor/GeneFacePlusPlus](https://github.com/yerfor/GeneFacePlusPlus)

This fork contains several enhancements and utilities built on top of the original GeneFacePlusPlus repository, focusing on improved audio-to-landmark generation, enhanced visualization tools, and better debugging capabilities.

## 🚀 Key Enhancements

### 1. **Enhanced Audio-to-Landmark Pipeline** 
- **New Files**: `audio2lmk.py`, `lmk_gen.py`
- **Features**:
  - Standalone audio-driven facial landmark generation
  - Support for MediaPipe-style 478-point landmarks
  - Interactive CLI for batch processing
  - 3D landmark visualization with GIF export
  - Optimized for both GPU and CPU inference

### 2. **Advanced SECC Renderer**
- **Modified**: `deep_3drecon/secc_renderer.py`
- **Enhancements**:
  - Batch rendering capabilities for video generation
  - GIF and MP4 export options
  - Configurable frame rates and output formats
  - Memory-efficient processing for long sequences

### 3. **Debug and Visualization Tools**
- **Modified**: `inference/genefacepp_infer.py`
- **Features**:
  - Intermediate SECC image saving for debugging
  - Step-by-step visualization of the inference pipeline
  - Enhanced logging and progress tracking

### 4. **Environment Customization**
- **New File**: `docs/prepare_env/requirements_boshen.txt`
- **Features**:
  - Additional dependencies for enhanced functionality
  - Gradio support for web interfaces
  - Updated package versions for compatibility

### 5. **Dataset Configurations**
- **New Directories**: `egs/datasets/withheadmotion_*`
- **Features**:
  - Pre-configured dataset settings for different scenarios
  - Support for various head motion patterns
  - Emotional expression datasets

## 📋 Quick Start

### Audio-to-Landmark Generation

```bash
# Interactive mode - generates landmarks from audio
python lmk_gen.py --ckpt_dir checkpoints/audio2motion_vae --device cuda

# Standalone pipeline with MediaPipe landmarks
python audio2lmk.py \
  --audio data/raw/val_wavs/sample.wav \
  --ckpt_dir checkpoints/audio2motion_vae \
  --out output/demo
```

### SECC Rendering and Visualization

```bash
# Run the enhanced SECC renderer (see deep_3drecon/secc_renderer.py main section)
cd deep_3drecon && python secc_renderer.py
```

## 🔧 Technical Improvements

### Path Configuration
- Updated hardcoded paths to be environment-agnostic
- Support for custom HuBERT model cache locations
- Flexible dataset and checkpoint directories

### Memory Optimization
- Efficient batch processing for long audio sequences
- GPU memory management improvements
- Chunked processing for large datasets

### Output Formats
- Multiple landmark output formats (68, 131, 478 points)
- Video export with customizable frame rates
- 3D visualization and animation support

## 📁 New File Structure

```
├── audio2lmk.py                    # Standalone audio→landmark pipeline
├── lmk_gen.py                      # Interactive landmark generation tool
├── bench_audio2lmk2.py            # Benchmarking utilities
├── docs/prepare_env/
│   └── requirements_boshen.txt     # Enhanced dependencies
├── egs/datasets/
│   ├── withheadmotion_clipped/     # Head motion dataset configs
│   └── withheadmotion_emo/         # Emotional expression configs
└── output/                         # Generated outputs directory
```

## 🎯 Use Cases

1. **Research and Development**
   - Facial animation research
   - Audio-visual synchronization studies
   - Expression transfer experiments

2. **Content Creation**
   - Animated avatar generation
   - Lip-sync for digital characters
   - Video dubbing and translation

3. **Interactive Applications**
   - Real-time facial animation
   - Virtual meetings and avatars
   - Gaming and entertainment

## 🔍 Key Differences from Original

| Feature | Original | This Fork |
|---------|----------|-----------|
| Landmark Output | Limited formats | 68/131/478 point support |
| Visualization | Basic | 3D animation + GIF export |
| Debug Tools | Minimal | Comprehensive debugging |
| Batch Processing | Manual | Interactive CLI |
| Environment Setup | Generic | Customized for WSL/Linux |
| Documentation | Basic | Enhanced with examples |

## 🛠 Development Environment

This fork has been developed and tested on:
- **OS**: WSL2 (Windows Subsystem for Linux)
- **GPU**: CUDA-enabled (RTX/GTX series recommended)
- **Python**: 3.8+
- **PyTorch**: 1.10+

## 📖 Usage Examples

### Generate Landmarks from Audio
```python
from lmk_gen import extract_features, run_model, load_audio2secc

# Extract audio features
hubert, f0 = extract_features("audio.wav", device="cuda")

# Load model and generate landmarks
model = load_audio2secc("checkpoints/audio2motion_vae", "cuda")
batch = run_model(model, hubert, f0, "cuda")

# Access 68-point landmarks
landmarks_68 = batch["lm68"].cpu().numpy()
```

### Batch SECC Rendering
```python
from deep_3drecon.secc_renderer import SECC_Renderer

renderer = SECC_Renderer(rasterize_size=512)
# See secc_renderer.py main section for complete example
```

## 🤝 Contributing

This fork maintains compatibility with the original GeneFacePlusPlus while adding enhanced functionality. Contributions are welcome, especially:

- Performance optimizations
- Additional output formats
- New visualization tools
- Documentation improvements

## 📄 License

This fork maintains the same license as the original GeneFacePlusPlus repository. Please refer to the original license for terms and conditions.

## 🙏 Acknowledgments

- Original GeneFacePlusPlus team ([yerfor](https://github.com/yerfor/GeneFacePlusPlus))
- MediaPipe team for landmark detection frameworks
- HuggingFace for transformer models and utilities

---

## Original GeneFace++ Information

For the original setup instructions, training procedures, and core functionality, please refer to the [original repository](https://github.com/yerfor/GeneFacePlusPlus).

### Original Citation
```
@article{ye2023geneface,
  title={GeneFace: Generalized and High-Fidelity Audio-Driven 3D Talking Face Synthesis},
  author={Ye, Zhenhui and Jiang, Ziyue and Ren, Yi and Liu, Jinglin and He, Jinzheng and Zhao, Zhou},
  journal={arXiv preprint arXiv:2301.13430},
  year={2023}
}
@article{ye2023geneface++,
  title={GeneFace++: Generalized and Stable Real-Time Audio-Driven 3D Talking Face Generation},
  author={Ye, Zhenhui and He, Jinzheng and Jiang, Ziyue and Huang, Rongjie and Huang, Jiawei and Liu, Jinglin and Ren, Yi and Yin, Xiang and Ma, Zejun and Zhao, Zhou},
  journal={arXiv preprint arXiv:2305.00787},
  year={2023}
}
```