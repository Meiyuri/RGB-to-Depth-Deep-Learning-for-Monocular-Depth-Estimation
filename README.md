# RGB-to-Depth-Deep-Learning-for-Monocular-Depth-Estimation
## I. Project Overview
This research focuses on monocular depth estimation using the DIODE Dataset, transitioning from CNN-based ResNet to *Swin-Transformers*. 

The architecture explores a **Dual-head design** (`is_indoor`) to handle indoor/outdoor domain gaps and evaluate its effectiveness.
>🔗[Pretrained Model (.onnx)](https://drive.google.com/drive/folders/1j0ZphIGvUC1EFjFhOrcYs6Nlg45xgDge?usp=sharing) 
## II. Environment
- Language: Python 3.11.9
- IDE/Interface: Google Colab
- Environment: Google Colab connected to a **remote runtime** (Virtual Machine)
- Hardware Specifications:
  + GPU: 1x NVIDIA RTX 3060 Ti 16GB VRAM
  + Setup: Local tunneling via Colab (to leverage high-end remote hardware)
- Core Libraries:
  + Deep Learning Framework: `torch`, `torchvision`, `timm`
  + Preprocessing: `numpy`, `PIL`, `opencv`
  + Data Augmentation: `albumentations`
  + Visualization: `matplotlib`, `opencv`
  + Deployment: `torch.onnx`

## III. Preprocessing
- Dataset: DIODE (Diverse Indoor and Outdoor Depth Dataset)
- Data Split:
  + Train: scene_00000, 00001, 00002, 00010, 00011
  + Val: scene_00004 (indoor), scene_00008 (outdoor)
  + Test: scene_00003 (indoor), scene_00007, 00009 (outdoor)
- Input Resolution: 
  + RGB images ($256 \times 320$) 
  + Depth: aligned with RGB
- Preprocessing Pipeline:
  + Resize:
    + RGB: Bilinear interpolation
    + Depth: Nearest neighbor interpolation
  + Depth Processing:
    + Depth clipping: 0.1m → 80m
    + Inverse Depth transformation ($d' = \frac{1}{d}$)
    + Min-max Normalization: [0, 1]
    + Validity mask to remove invalid regions (sky, reflections, sensor noise)
- Data Augmentation (on-the-fly)
  + Horizontal Flip
  + Brightness & Contrast adjustment
  + Gaussian Noise
  + Blur
    
→ These augmentations improve robustness and reduce overfitting.

## IV. Model Architectures
### 1. Baseline Model (ResNet-101)
- Encoder: ResNet-101 (Pretrained).
- Decoder: Standard ConvBlocks + Bilinear Upsampling + Skip Connections
### 2. Combine Model
- Encoder: ResNet-101 (Pretrained)
- Bottleneck: TransformerBlock (self-attention for Global Context)
- Decoder: `FusionBlock` (Residual + RELU)
### 3. SOTA Compare
- Encoder: Swin Transformer
- Bottleneck: Dilated Conv for receptive field expansion
- Decoder: `FeatureFusionBlock` with Projection Shortcuts
### 4. Lightweight Model
- Encoder: MobileNetV2 / EfficientNet-Lite0
- Decoder: Lightweight Fusion (Depthwise Separable Convolutions)
## V. Training Strategy 
- Loss Functions:
  + Baseline: L1 Loss + SSIM
  + Transformer (Swin): Huber Loss + Scale-Invariant Loss + SSIM + Edge-aware Loss
  + Lightweight:
    
      $L = 0.7 \cdot L_{GT} + 0.3 \cdot L_{Teacher}$

- Knowledge Distillation:
  + Teacher: Swin Transformer
  + Student: MobileNet / EfficientNet-Lite0
    
→ Helps lightweight models retain high-level depth understanding

- Optimization
  + Optimizer: AdamW
  + Learning Rate Scheduler: Cosine Annealing / Step Decay
  + Mixed Precision Training (FP16) for efficiency
