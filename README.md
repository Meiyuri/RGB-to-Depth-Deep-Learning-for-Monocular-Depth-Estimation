# RGB-to-Depth-Deep-Learning-for-Monocular-Depth-Estimation
## I. Project Overview
This research focuses on monocular depth estimation using the DIODE Dataset, transitioning from CNN-based ResNet to *Swin-Transformers*. The architecture utilizes a **Dual-head** design, using the `is_indoor` flag to separate indoor/outdoor supervision for improved accuracy and reduced computational cost
>🔗[Pretrained Model (.onnx)](https://drive.google.com/drive/folders/1j0ZphIGvUC1EFjFhOrcYs6Nlg45xgDge?usp=sharing) 
## II. Environment
- Language: Python 3.11.9
- IDE/Interface: Google Colab
- Environment: Google Colab connected to a **remote runtime** (Virtual Machine)
- Hardware Specifications:
  + GPU: 1x NVIDIA RTX 5060 Ti (16GB VRAM)
  + Setup: Local tunneling via Colab (to leverage high-end remote hardware)
- Core Libraries:
  + Deep Learning Framework: `torch`, `torchvision`, `timm`
  + Preprocessing: `numpy`, `PIL`, `opencv`
  + Data Augmentation: `albumentation`
  + Visualization: `matplotlib`, `opencv`
  + Deployment: `torch.onnx`

## III. Preprocessing 
- Dataset: DIODE (selected scenes)
  + Train: scene_00000, 00001, 00002, 00010, 00011
  + Val: scene_00004 (indoor), scene_00008 (outdoor)
  + Test: scene_00003 (indoor), scene_00007, 00009 (outdoor)
- Input Dimensions: RGB images ($256 \times 320$) 
- Resize:
  + RGB: bilinear interpolation
  + Depth: nearest neighbor interpolation
- Depth Processing:
  + Clipping Depth: Max 80m 
  + Inverse Depth transformation ($d = \frac{1}{d}$)
  + Min-max scaling to [0, 1]

## IV. Model Architectures
Training *Augmentations*: Horizontal Flip, Brightness & Contrast adjustment, Noise injection and slight Blurring
### 1. Baseline Model (ResNet-101)
- Encoder: ResNet-101 (Pretrained).
- Decoder: Standard ConvBlocks + Bilinear Upsampling + Skip Connections
### 2. Combine Model
- Encoder: ResNet-101 (Pretrained)
- Bottleneck: TransformerBlock (self-attention for Global Context)
- Decoder: `FusionBlock` (Residual + RELU)
### 3. SOTA Compare
- Encoder: Swin transformer
- Bottleneck: Dilated Conv for receptive field expansion
- Decoder: `FeatureFusionBlock` with Projection Shortcuts
### 4. Lightweight Model
- Encoder: MobileNetV2 / EfficientNet-Lite0
- Decoder: Lightweight Fusion (Depthwise Separable Convolutions)
