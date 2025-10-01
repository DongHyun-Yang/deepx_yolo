# DEEPX-YOLO Project

This project demonstrates various YOLO Model inference implementations, from standalone ONNX Runtime to DEEPX-accelerated execution. It provides multiple example scripts for different use cases: object detection, pose estimation, instance segmentation, and oriented bounding box detection.

## 🎯 Project Overview

This repository includes:
- **Standalone implementations**: Zero external dependencies (ONNX/DXNN inference without Ultralytics)
- **Ultralytics DEEPX implementations**: Using custom Ultralytics DEEPX library for enhanced debugging and DXNN support
- **Multiple task support**: Detection, Pose Estimation, Segmentation, OBB
- **Model conversion utilities**: PyTorch to ONNX export scripts

## 🗂️ Directory Structure

```plaintext
yolov11l_poc/
├── README.md
├── requirements.txt
├── assets/                     # Sample images
│   ├── boats.jpg
│   ├── bus.jpg
│   └── zidane.jpg
├── test_images/               # Test images folder
│   └── 1.jpg ~ 7.jpg
├── lib/
│   └── ultralytics/           # Custom Ultralytics DEEPX library (submodule)
├── yolo11l/                   # Object Detection examples
│   ├── export_onnx.py                            # PyTorch to ONNX conversion
│   ├── ultralytics_deepx_lib_setup.py            # Custom library setup script
│   ├── predict_onnx_standalone.py                # Standalone ONNX inference (zero dependencies)
│   ├── predict_onnx_ultralytics_postprocess.py   # ONNX inference + Ultralytics postprocessing (hybrid)
│   ├── predict_onnx_ultralytics_deepx.py         # ONNX inference with custom Ultralytics DEEPX library
│   ├── predict_dxnn_standalone.py                # Standalone DXNN inference (zero dependencies)
│   ├── predict_dxnn_ultralytics_postprocess.py   # DXNN inference + Ultralytics postprocessing (hybrid)
│   ├── predict_dxnn_ultralytics_deepx.py         # DXNN inference with custom Ultralytics DEEPX library
│   ├── models/
│   │   ├── metadata.yaml
│   │   ├── yolo11l.pt                        # PyTorch model
│   │   ├── yolo11l.onnx                      # ONNX model
│   │   └── yolo11l.dxnn                      # DEEPX model
│   └── runs/predict/                         # Output results
├── yolo11l-pose/              # Pose estimation example
│   ├── ultralytics_deepx_lib_setup.py
│   ├── export_onnx.py
│   ├── predict_onnx_standalone.py
│   ├── predict_onnx_ultralytics_postprocess.py
│   ├── predict_onnx_ultralytics_deepx.py
│   ├── predict_dxnn_standalone.py
│   ├── predict_dxnn_ultralytics_postprocess.py
│   ├── predict_dxnn_ultralytics_deepx.py
│   └── models/
├── yolo11l-seg/               # Instance segmentation example
│   ├── ultralytics_deepx_lib_setup.py
│   ├── export_onnx.py
│   ├── predict_onnx_standalone.py
│   ├── predict_onnx_ultralytics_postprocess.py
│   ├── predict_onnx_ultralytics_deepx.py
│   ├── predict_dxnn_standalone.py
│   ├── predict_dxnn_ultralytics_postprocess.py
│   ├── predict_dxnn_ultralytics_deepx.py
│   └── models/
└── yolo11l-obb/               # Oriented bounding box example
│   ├── ultralytics_deepx_lib_setup.py
│   ├── export_onnx.py
│   ├── predict_onnx_standalone.py
│   ├── predict_onnx_ultralytics_postprocess.py
│   ├── predict_onnx_ultralytics_deepx.py
│   ├── predict_dxnn_standalone.py
│   ├── predict_dxnn_ultralytics_postprocess.py
│   ├── predict_dxnn_ultralytics_deepx.py
    └── models/
```

## 🛠️ Prerequisites

### 1. Python Environment Requirements

- Python 3.12 or higher (tested with Python 3.12.3)

### 2. Required Package Installation

```bash
# Create and activate virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Linux
# venv\Scripts\activate   # On Windows

# Install required packages
pip install -r requirements.txt
```

### 3. Main Dependencies

**Core Libraries:**
- **torch**: PyTorch for tensor operations
- **ultralytics**: YOLOv11 model loading and conversion (for Ultralytics DEEPX examples)
- **opencv-python**: Image processing and visualization
- **numpy**: Numerical computation
- **onnxruntime**: ONNX model inference

**DEEPX Runtime Python Library (for DXNN inference):**
- **dx-engine**: DEEPX runtime for DXNN model inference

### 4. Custom Ultralytics DEEPX Library Setup

The custom Ultralytics DEEPX library is included as a Git submodule in `lib/ultralytics/`. It provides:
- Debug visualization of input tensors
- Debug saving of raw output tensors
- DXNN model inference support

To initialize the submodule:
```bash
git submodule update --init --recursive
```

The `ultralytics_deepx_lib_setup.py` script automatically configures the Python path to use this custom library.

## 📥 Model Download

### YOLOv11 Model Download

YOLOv11 models can be downloaded from [Ultralytics official documentation](https://docs.ultralytics.com/models/yolo11/#performance-metrics).

```bash
# Navigate to yolo11l folder
cd yolo11l/models

# Download YOLOv11l model (approximately 50MB)
# Method 1: Using wget
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt

# Method 2: Direct download
# Download from browser and save to yolo11l/models/ folder
```

**Available Models:**
- `yolo11l.pt`: Object Detection (PyTorch format)
- `yolo11l-pose.pt`: Pose Estimation
- `yolo11l-seg.pt`: Instance Segmentation
- `yolo11l-obb.pt`: Oriented Bounding Box Detection

## 🚀 Usage

### 1. Model Conversion (PyTorch → ONNX)

```bash
cd yolo11l
python export_onnx.py
```

**export_onnx.py features:**
- Converts `models/yolo11l.pt` → `models/yolo11l.onnx`
- Exports metadata.yaml with model configuration
- Uses ONNX opset 12 for maximum compatibility
- Supports SSL certificate bypass for corporate environments

### 2. Object Detection Inference

The project provides **six different inference implementations**:

#### 2.1. Standalone ONNX Inference (Recommended for learning)

```bash
cd yolo11l
python predict_onnx_standalone.py
```

**Features:**
- ✅ **Zero Ultralytics dependencies** - All functions ported into single file
- ✅ **Fully self-contained** - Complete Boxes and Results classes included
- ✅ **Educational** - Easy to understand preprocessing, inference, and postprocessing
- ✅ **Reusable** - Can be adapted for pose, segmentation, and OBB tasks

**Implementation highlights:**
- Custom preprocessing: letterbox, normalization
- Direct ONNX Runtime execution
- Ported NMS and coordinate scaling functions
- Complete Results object with all Boxes attributes

#### 2.2. ONNX Inference with Ultralytics Postprocessing (Hybrid)

```bash
cd yolo11l
python predict_onnx_ultralytics_postprocess.py
```

**Features:**
- ✅ **Hybrid Approach** - Custom preprocessing/inference + Ultralytics postprocessing
- ✅ **Minimal Dependencies** - Uses Ultralytics utilities only for postprocessing
- ✅ **Flexibility** - Custom preprocessing/inference with validated postprocessing
- ✅ **Learning-Friendly** - Clear separation of each step

**Implementation highlights:**
- Custom preprocessing: letterbox, normalization
- Direct ONNX Runtime execution
- Ultralytics postprocessing: non_max_suppression, ops.scale_boxes, Results class
- Uses Ultralytics utilities without YOLO class

#### 2.3. ONNX Inference with Ultralytics DEEPX Library

```bash
cd yolo11l
python predict_onnx_ultralytics_deepx.py
```

**Features:**
- Uses custom Ultralytics DEEPX library
- Complete YOLO class for end-to-end inference
- Debug features: input tensor visualization, raw output saving
- All preprocessing/inference/postprocessing handled by Ultralytics

#### 2.4. Standalone DXNN Inference

```bash
cd yolo11l
python predict_dxnn_standalone.py
```

**Features:**
- DEEPX runtime for accelerated inference
- Zero Ultralytics dependencies
- Similar structure to standalone ONNX version

#### 2.5. DXNN Inference with Ultralytics Postprocessing (Hybrid)

```bash
cd yolo11l
python predict_dxnn_ultralytics_postprocess.py
```

**Features:**
- ✅ **Hybrid Approach** - Custom preprocessing/DXNN inference + Ultralytics postprocessing
- ✅ **DEEPX Acceleration** - Fast inference via DXNN runtime
- ✅ **Validated Postprocessing** - Uses Ultralytics NMS and coordinate scaling
- ✅ **Production-Ready** - Balance of performance and accuracy

**Implementation highlights:**
- Custom preprocessing: letterbox, normalization
- DXNN Runtime execution (dx_engine)
- Ultralytics postprocessing: non_max_suppression, ops.scale_boxes, Results class
- Leverages DEEPX acceleration without full YOLO class

#### 2.6. DXNN Inference with Ultralytics DEEPX Library

```bash
cd yolo11l
python predict_dxnn_ultralytics_deepx.py
```

**Features:**
- Uses DEEPX runtime via custom Ultralytics library
- Complete YOLO class support
- Enhanced debugging capabilities

### 3. Execution Process

All inference scripts follow this pattern:

1. **Input**: Search for images in `../assets/` or specified directory
2. **Preprocessing**: Letterbox resize, normalization, channel conversion
3. **Inference**: ONNX Runtime or DEEPX execution
4. **Postprocessing**: NMS, coordinate scaling, Results object creation
5. **Visualization**: Draw bounding boxes and save results
6. **Output**: Save to `runs/predict/{backend}/{script_name}/` directory

### 4. Execution Result Example

```plaintext
Processing directory of images.
Results will be saved in 'runs/predict/onnx/standalone' folder.
--------------------------------------------------

[1/3] Processing: boats.jpg
Debug info:
  Original size: 1280x720
  Ratio: (0.5, 0.5)
  Padding (dw, dh): (80.0, 0.0)
  Input tensor shape: (1, 3, 640, 640)
  Input tensor range: [0.000, 1.000]

Loading ONNX model: models/yolo11l.onnx
Running ONNX inference...
Raw output shape: (1, 84, 8400)
Raw output range: [-4.234, 8.567]

[Postprocess] Total detections: 2
==================================================
Total object detections: 2
Boxes tensor shape: torch.Size([2, 6])
Confidence range: 0.878 ~ 0.914
Class distribution: {0: 1, 8: 1}

[boats] Total 2 objects detected.
  1. person: 0.91 - Position: (221, 402) ~ (344, 857)
  2. boat: 0.88 - Position: (90, 456) ~ (1259, 880)

Detection result saved to 'runs/predict/onnx/standalone/boats_detected_20251021_143052_123.jpg' file.
--------------------------------------------------

PROCESSING SUMMARY
======================================================================
Total images processed: 3
Output directory: runs/predict/onnx/standalone

Saved files:
  1. runs/predict/onnx/standalone/boats_detected_20251021_143052_123.jpg
  2. runs/predict/onnx/standalone/bus_detected_20251021_143053_456.jpg
  3. runs/predict/onnx/standalone/zidane_detected_20251021_143054_789.jpg
======================================================================
Processing completed successfully!
======================================================================
```



## ⚙️ Configuration Options

### Common Configuration (All Scripts)

```python
# Model paths
MODEL_PATH = 'models/yolo11l.onnx'  # or 'models/yolo11l.dxnn' for DXNN
SOURCE_PATH = '../assets'            # Input image path (file or directory)
OUTPUT_DIR = 'runs/predict/...'     # Result storage directory

# Detection parameters (Ultralytics defaults)
CONFIDENCE_THRESHOLD = 0.25   # Confidence threshold (0.0 ~ 1.0)
IOU_THRESHOLD = 0.45         # IoU threshold for NMS
INPUT_SIZE = 640             # Model input size
```



## 🔧 Troubleshooting

### SSL Certificate Error


### ONNX Runtime Error

```bash
# Reinstall CPU version
pip uninstall onnxruntime
pip install onnxruntime

# For GPU version (requires NVIDIA GPU)
pip install onnxruntime-gpu
```

### Custom Ultralytics Library Not Found

```bash
# Initialize Git submodule
git submodule update --init --recursive

# Verify lib/ultralytics/ exists
ls lib/ultralytics/

# The ultralytics_deepx_lib_setup.py script should handle path configuration
```

### DEEPX Runtime Error

```bash
# Install dx-engine for DXNN inference
pip install dx-engine

# Verify installation
python -c "from dx_engine import InferenceEngine; print('DEEPX OK')"
```

## 📈 Performance Information

### Model Specifications

| Model | Size | mAP50-95 | Speed (CPU) | Parameters | FLOPs |
|-------|------|----------|-------------|------------|-------|
| YOLOv11n | ~6MB | 39.5% | ~50ms | 2.6M | 6.5B |
| YOLOv11s | ~19MB | 47.0% | ~100ms | 9.4M | 21.5B |
| YOLOv11m | ~40MB | 51.5% | ~200ms | 20.1M | 68.0B |
| YOLOv11l | ~50MB | 53.4% | ~300ms | 25.3M | 86.9B |
| YOLOv11x | ~110MB | 54.7% | ~500ms | 56.9M | 194.9B |

### Inference Performance

**ONNX Runtime (CPU):**
- Image preprocessing: ~10-20ms
- Model inference: ~200-500ms (depends on model size)
- Postprocessing (NMS): ~10-30ms
- Total: ~250-550ms per image

**DEEPX Runtime (Accelerated):**
- Significant speedup on supported hardware
- Optimized memory usage
- Lower latency for batch processing

### Supported Resolutions

- Default input: 640x640 (automatic letterbox resize)
- Maximum tested: 1920x1080
- Minimum recommended: 320x320

## 🎓 Learning Resources

### Understanding the Code

1. **Start with standalone scripts**: 
   - `predict_onnx_standalone.py` is the best starting point
   - All preprocessing, inference, and postprocessing in one file
   - Well-commented with debug outputs

2. **Key concepts to understand**:
   - **Letterbox preprocessing**: Maintain aspect ratio while resizing
   - **NMS (Non-Maximum Suppression)**: Remove duplicate detections
   - **Coordinate scaling**: Convert from model space to image space
   - **Results object**: Container for all detection information

3. **Progression path**:
   ```
   predict_onnx_standalone.py         → Understand full pipeline
   predict_onnx_ultralytics_deepx.py  → See Ultralytics DEEPX integration library
   predict_dxnn_standalone.py         → Learn DEEPX runtime
   predict_dxnn_ultralytics_deepx.py  → See Ultralytics DEEPX integration library
   ```

### Code Reuse

standalone scripts include complete implementations that can be reused:

```python
# From predict_onnx_standalone.py

# Reusable Boxes class with 11 attributes
class Boxes:
    - xyxy, xywh, xyxyn, xywhn  # Various coordinate formats
    - conf, cls, id              # Detection metadata
    - shape, is_track            # Properties
    - cpu(), numpy(), cuda()     # Device management

# Reusable utility functions
def letterbox()              # Aspect-ratio preserving resize
def preprocess_image()       # Complete preprocessing pipeline
def non_max_suppression()    # NMS implementation
def scale_boxes()            # Coordinate transformation
```

These can be adapted for:
- Pose estimation (add keypoint processing)
- Instance segmentation (add mask processing)
- Oriented bounding boxes (add angle processing)

## File Descriptions

### Core Scripts (yolo11l/)

| File | Purpose | Dependencies | Use Case |
|------|---------|--------------|----------|
| `export_onnx.py` | Convert PyTorch to ONNX | ultralytics, torch | Model preparation |
| `ultralytics_deepx_lib_setup.py` | Configure custom library path | - | Library initialization |
| `predict_onnx_standalone.py` | ONNX inference (zero deps) | cv2, numpy, torch, onnxruntime | Learning, customization |
| `predict_onnx_ultralytics_deepx.py` | ONNX inference (full lib) | ultralytics (custom), cv2, numpy, torch | Quick prototyping, debugging |
| `predict_dxnn_standalone.py` | DXNN inference (zero deps) | cv2, numpy, torch, dx-engine | Production, minimal deps |
| `predict_dxnn_ultralytics_deepx.py` | DXNN inference (full lib) | ultralytics (custom), dx-engine | Development with acceleration |

### Output Structure

```plaintext
yolo11l/runs/predict/
├── onnx/
│   ├── standalone/                    # predict_onnx_standalone.py outputs
│   │   ├── [input_image_name]_detected_[timestamp].jpg
│   │   └── debug/
│   │       ├── input/                 # Preprocessed input visualizations
│   │       └── raw_output/            # Raw model outputs (.npy)
│   ├── ultralytics_postprocess/       # predict_onnx_ultralytics_postprocess.py outputs
│   │   ├── [input_image_name]_detected_[timestamp].jpg
│   │   └── debug/
│   │       ├── input/                 # Preprocessed input visualizations
│   │       └── raw_output/            # Raw model outputs (.npy)
│   └── ultralytics_deepx/             # predict_onnx_ultralytics_deepx.py outputs
│       ├── [input_image_name]_detected_[timestamp].jpg
│       └── debug/
│           ├── input/
│           ├── raw_output/
│           └── origin_output/         # Ultralytics native outputs
└── dxnn/
    ├── standalone/                    # predict_dxnn_standalone.py outputs
    │   ├── [input_image_name]_detected_[timestamp].jpg
    │   └── debug/
    │       ├── input/                 # Preprocessed input visualizations
    │       └── raw_output/            # Raw model outputs (.npy)
    ├── ultralytics_postprocess/       # predict_dxnn_ultralytics_postprocess.py outputs
    │   ├── [input_image_name]_detected_[timestamp].jpg
    │   └── debug/
    │       ├── input/                 # Preprocessed input visualizations
    │       └── raw_output/            # Raw model outputs (.npy)
    └── ultralytics_deepx/             # predict_dxnn_ultralytics_deepx.py outputs
        ├── [input_image_name]_detected_[timestamp].jpg
        └── debug/
            ├── input/
            ├── raw_output/
            └── origin_output/         # Ultralytics native outputs
    
```


## 🔍 Debugging and Comparison

### Comparing Outputs

The project includes utilities to compare outputs between different implementations:

```bash
# Compare raw model outputs (standalone vs Ultralytics DEEPX)
python util/compare_raw_outputs.py \
    runs/predict/onnx/standalone/debug/raw_output/raw_output_[timestamp].npy \
    runs/predict/onnx/ultralytics_deepx/debug/raw_output/raw_output_[timestamp].npy

# Compare raw model outputs (onnx vs dxnn)
python util/compare_raw_outputs.py \
    runs/predict/onnx/standalone/debug/raw_output/raw_output_[timestamp].npy \
    runs/predict/dxnn/standalone/debug/raw_output/raw_output_[timestamp].npy
```


### Debug Features

**standalone scripts:**
```python
# Enable debug mode in run_inference()
result_path = run_inference(MODEL_PATH, image_path, OUTPUT_DIR, debug=True)

# Outputs:
# - Preprocessed input tensor visualization
# - Raw model output (.npy file)
# - Detailed console logs
```

**Ultralytics DEEPX Scripts:**
```python
# Debug features automatically enabled via custom library
# Additional outputs:
# - Input tensor: debug/input/preprocessed_input_[timestamp].jpg
# - Raw output: debug/raw_output/raw_output_[timestamp].npy
# - Ultralytics output: debug/origin_output/
```


## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) - YOLOv11 model and framework
- [ONNX Runtime](https://onnxruntime.ai/) - Efficient inference engine
- [DEEPX](https://www.deepx.ai/) - Hardware acceleration runtime
- COCO Dataset - Training and evaluation dataset

**Built with ❤️ using YOLOv11, ONNX Runtime, and DEEPX**
