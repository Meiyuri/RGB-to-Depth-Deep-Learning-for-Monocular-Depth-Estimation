# RGB-to-Depth-Deep-Learning-for-Monocular-Depth-Estimation
## I. Objective
This research focuses on Monocular Depth Estimation (MDE) using a subset of the DIODE Dataset. The project systematically explores the transition from Convolutional Neural Network (CNN)-based architectures (ResNet) to Transformer-based models (Swin Transformer).

As part of the experimental process, the system investigates a Dual-head architecture, controlled by an is_indoor flag, to test if separately learning indoor and outdoor depth distributions improves generalization. Ultimately, the research proposes a single-head Swin Transformer as the State-of-the-Art (SOTA) for this pipeline.

The main objectives include:

- Learning depth information from monocular RGB input
- Handling the domain gap between indoor and outdoor environments
- Exploring the trade-off between accuracy and computational efficiency
- Enabling deployment on resource-constrained edge devices via lightweight architectures
## II. Key Contributions
- Investigate a Dual-head architecture (Combine Model) with a shared encoder and domain-specific routing, analyzing the effectiveness of explicit indoor/outdoor separation
- Propose a Swin Transformer-based model as the SOTA for this specific setup, leveraging global attention for better depth reasoning
- Conduct a comprehensive benchmark across CNN (ResNet), Hybrid (Dual-head), Transformer (Swin), and Lightweight models (MobileNetV2, EfficientNet-Lite0) on the challenging DIODE subset
- Provide an in-depth analysis of the domain gap and error metrics, explaining the mathematical behaviors of different architectures in structured vs. unstructured environments
- Enable real-time edge deployment by exporting models to ONNX (Open Neural Network Exchange) and ensuring TensorRT compatibility
## III. Dataset
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
## IV. Environment
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
## V. Model
The architectures are designed to progressively address key challenges in monocular depth estimation, including local feature limitation (CNN), global context modeling (Transformer), and deployment constraints (lightweight models).
### 1. Baseline Model (ResNet-101)
- Encoder: ResNet-101 (ImageNet pretrained)
- Decoder: Standard ConvBlocks + Bilinear Upsampling + Skip Connections
### 2. Combine Model (CNN + Transformer, Dual-head)
- Encoder: ResNet-101 (Pretrained)
- Bottleneck: TransformerBlock (self-attention for Global Context)
- Decoder: `FusionBlock` (Residual + RELU)
- Dual-head output:
  + Indoor head
  + Outdoor head
#### Characteristics:
- Improved global context understanding
- Designed to reduce domain gap
- Increased computational cost
### 3. Transformer-based Model (Swin Transformer) – Proposed SOTA Model
- Encoder: Swin Transformer
- Bottleneck: Dilated convolution
- Decoder: FeatureFusionBlock
#### Characteristics:
- Strong global reasoning capability
- Highest overall accuracy
- High memory and computation requirements
### 4. Lightweight Model
a. MobileNetV2 Variant
- Encoder: MobileNetV2
- Decoder: FeatureFusionBlock (reuse from SOTA)
- Channel alignment via 1×1 Conv
#### Characteristics:
- Good trade-off between speed and accuracy
b. EfficientNet-Lite0 Variant
- Encoder: EfficientNet-Lite0
- Decoder: Depthwise Separable Convolutions
- Training: Mixed Precision (FP16)
#### Characteristics:
- Very lightweight
- Reduced depth estimation performance
## VI. Training Strategy 
- Loss Functions:
  + Baseline: L1 Loss + SSIM
  + Transformer (Swin): Huber Loss + Scale-Invariant Loss + SSIM + Edge-aware Loss
  + Lightweight:
      $Loss = 0.7 \times GroundTruth + 0.3 \times Teacher (Distillation)$
- Knowledge Distillation:
  + Teacher: Swin Transformer
  + Student: MobileNet / EfficientNet-Lite0
→ Helps lightweight models retain high-level depth understanding
- Optimization
  + Optimizer: AdamW
  + Learning Rate Scheduler: Cosine Annealing / Step Decay
  + Mixed Precision Training (FP16) for efficiency
## VII. Evaluation Metrics 
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- δ Accuracy:
  + δ < 1.25
  + δ < 1.25²
  + δ < 1.25³
- Scale-Invariant Error

## VIII. Results 
- The following table summarizes the evaluation results of all models on the DIODE dataset.

| Model                           | Scene   | Abs Rel ↓  | RMSE (m) ↓ | δ < 1.25 ↑ | δ < 1.25² ↑ | δ < 1.25³ ↑ |
| ------------------------------- | ------- | ---------- | ---------- | ---------- | ----------- | ----------- |
| **ResNet-101 (Baseline)** | Overall | 0.4547     | 5.6582     | 0.3824     | 0.6338      | 0.7641      |
|                                 | Indoor  | **0.3233** | 0.7648     | 0.5108     | 0.8095      | 0.9191      |
|                                 | Outdoor | 0.5782     | 10.2556    | 0.2618     | 0.4688      | 0.6184      |
| **Combine (CNN + Transformer)** | Overall | 0.5305     | 5.6198     | 0.3760     | 0.6279      | 0.7668      |
|                                 | Indoor  | **0.3214** | 0.7935     | 0.5107     | 0.8079      | 0.9167      |
|                                 | Outdoor | 0.7270     | 10.1543    | 0.2494     | 0.4588      | 0.6258      |
| **Swin Transformer (Proposed SOTA Model)** | Overall | 0.5603 | **5.1497** | **0.4059** | **0.6728** | **0.8148** |
|                                 | Indoor  | 0.3349     | **0.7511** | **0.5191** | **0.8199** | **0.9379** |
|                                 | Outdoor | **0.7722** | **9.2823** | **0.2996** | **0.5345** | **0.6991** |
| **MobileNetV2 (Lightweight)** | Overall | 0.4971     | 5.7570     | 0.3701     | 0.6040      | 0.7394      |
|                                 | Indoor  | 0.3501     | 0.8783     | 0.4948     | 0.7775      | 0.8966      |
|                                 | Outdoor | 0.6352     | 10.3407    | 0.2530     | 0.4410      | 0.5918      |
| **EfficientNet-Lite0** | Overall | 0.5876     | 6.3472     | 0.2859     | 0.5077      | 0.6632      |
|                                 | Indoor  | 0.4535     | 1.2403     | 0.3717     | 0.6398      | 0.8040      |
|                                 | Outdoor | 0.7137     | 11.1451    | 0.2053     | 0.3836      | 0.5310      |

#### Key Findings

- The **Swin Transformer (Proposed SOTA Model)** achieves the best overall performance in terms of RMSE and δ accuracy metrics, indicating stronger global reasoning capability

- Specifically:
  + Best RMSE: 5.1497 (Overall), 0.7511 (Indoor), 9.2823 (Outdoor)
  + Best δ < 1.25: 0.4059 (Overall), 0.5191 (Indoor), 0.2996 (Outdoor)

- However, the ResNet-101 baseline achieves lower Abs Rel error (0.4547 overall) compared to the **Swin Transformer** (0.5603). This highlights an important mathematical paradox:
  + RMSE penalizes large outliers (due to squared errors). Swin's superior RMSE proves it is exceptionally good at avoiding catastrophic depth failures (e.g., predicting 80m instead of 10m) thanks to its global context understanding
  + Abs Rel measures average proportional error. The higher Abs Rel suggests that while Swin gets the "big picture" right, it struggles slightly more with local, fine-grained depth boundaries compared to the CNN baseline

- Indoor performance is consistently better across all models:
  + Lower RMSE (~0.7–1.2m)
  + Higher δ accuracy (~0.5+)
→ This is due to more stable depth ranges and structured environments

- Outdoor scenes remain significantly more challenging:
  + RMSE increases drastically (~9–11m)
  + δ accuracy drops across all models
→ Caused by large depth variation, lighting changes, and sparse depth signals

- The Dual-head (Combine) model does not outperform the baseline:
  + Outdoor Abs Rel worsens (0.7270 vs 0.5782)
→ Indicates that domain separation alone is insufficient without stronger supervision or balancing

- MobileNetV2 provides the best trade-off between efficiency and performance:
  + Comparable RMSE (5.7570) with significantly reduced computational cost
  + Suitable for real-time and edge deployment

- EfficientNet-Lite0 shows the weakest performance:
  + Highest RMSE (6.3472 overall)
  + Lowest δ accuracy
→ Suggests excessive model compression leads to a severe loss of depth representation capacity, causing the model to over-rely on pixel intensity rather than geometric structure
## IX. Deployment 
- Export format: ONNX
- Compatible with:
  + TensorRT
  + ONNX Runtime
- Supports:
  + Real-time inference
  + Edge deployment (Jetson, mobile devices)

## X. Pretrained Model
>🔗[Pretrained Model (.onnx)](https://drive.google.com/drive/folders/1j0ZphIGvUC1EFjFhOrcYs6Nlg45xgDge?usp=sharing) 

## XI. Limitations & Future Work
- Outdoor performance remains significantly lower than indoor across all architectures
- Dual-head architecture has not fully improved outdoor generalization due to limited data variance
- Knowledge Distillation Risks: Since the Teacher model (Swin) exhibited slightly weaker local structure learning (higher Abs Rel), performing Knowledge Distillation may have inherently passed these local inaccuracies down to the lightweight Student models
#### Future work:
- Improve outdoor data balance and apply targeted augmentations
- Explore better loss weighting for dual-head training
- Apply self-supervised or multi-task learning (e.g., incorporating Surface Normal Loss) to improve depth boundaries
