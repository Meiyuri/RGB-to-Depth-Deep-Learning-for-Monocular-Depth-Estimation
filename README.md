# RGB-to-Depth-Deep-Learning-for-Monocular-Depth-Estimation

## I. Preprocessing

## II. Build Deep Learning Model
### 1. Training Environment
- Language: Python 3.11.9
- IDE/Interface: Google Colab
- Environment: Google Colab connected to a remote runtime (Virtual Machine)
- Hardware Specifications:
  + GPU: 1x NVIDIA RTX 5060 Ti (16GB VRAM)
  + Setup: Local tunneling via Colab (to leverage high-end remote hardware)
- Core Libraries:
  + Deep Learning Framework: torch, torchvision, timm
  + Data Augmentation: albumentation
  + Visualization: matplotlib
  + Deployment: torch.onnx
### 2. Data Augmentation
- Input Dimensions: RGB images ($256 \times 320$)
- Training Augmentations: Horizontal Flip, Brightness & Contrast adjustment, Noise injection and slight Blurring
