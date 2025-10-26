
# HSLU Generative AI Course

A comprehensive hands-on course covering Computer Vision, Deep Learning, and Generative AI from fundamentals to state-of-the-art models.

## Course Overview

This repository contains 16 practical Jupyter notebooks that progressively build knowledge from basic image processing to advanced generative models. The course covers:

- **Foundations**: Image processing, feature extraction, optical flow
- **Deep Learning**: ANNs, CNNs, ResNets, Vision Transformers
- **Object Detection**: YOLO, SSD, Mask R-CNN fine-tuning
- **Generative Models**: VAEs, GANs, Diffusion Models
- **Vision-Language Models**: CLIP zero-shot classification
- **Advanced Topics**: Attention visualization, conditional generation

## Table of Contents

- [Notebooks](#notebooks)
  - [Part 1: Image Processing Fundamentals](#part-1-image-processing-fundamentals-notebooks-1-3)
  - [Part 2: Convolutional Neural Networks](#part-2-convolutional-neural-networks-notebooks-4-7)
  - [Part 3: Generative Models](#part-3-generative-models-notebooks-8-10)
  - [Part 4: Transformers & Vision-Language Models](#part-4-transformers--vision-language-models-notebooks-11-14)
  - [Part 5: Diffusion Models](#part-5-diffusion-models-notebooks-15-16)
- [Learning Paths](#learning-paths)
- [Requirements](#requirements)
- [Quick Start](#quick-start)

---

## Notebooks

### Part 1: Image Processing Fundamentals (Notebooks 1-3)

#### 1. Image Processing Explained
**File**: `1-image_processing_explained.ipynb`

**Goals**:
- Understand mathematical foundations of image processing operations
- Learn filtering techniques (Gaussian blur, Sobel edge detection)
- Master morphological operations (erosion, dilation, opening, closing)
- Apply histogram equalization and CLAHE for image enhancement

**Key Concepts**:
- Convolution and correlation
- Gaussian blur: $G(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$
- Sobel operator for edge detection
- Morphological transformations

**Output**: Enhanced images with various filters and transformations

---

#### 2. Feature Matching and Optical Flow
**File**: `2-feature_matching_optical_flow.ipynb`

**Goals**:
- Detect and match keypoints across image pairs
- Compare SIFT (128-dim float) vs ORB (256-bit binary) descriptors
- Understand scale and rotation invariance
- Analyze motion through displacement vectors

**Key Techniques**:
- **SIFT**: DoG scale-space detection, Lowe's ratio test
- **ORB**: FAST corner detection + BRIEF descriptors
- Feature matching with BFMatcher (L2 norm and Hamming distance)
- Displacement heatmap visualization

**Dataset**: Video frames (640×360)

---

#### 3. Motion Analysis with Optical Flow
**File**: `3-Motion_analysis_optical_flow.ipynb`

**Goals**:
- Implement sparse optical flow (Lucas-Kanade)
- Apply dense optical flow (Farneback algorithm)
- Visualize motion direction and magnitude
- Create animated frame-by-frame flow analysis

**Key Concepts**:
- **Lucas-Kanade**: Window-based least-squares motion estimation
- **Farneback**: Polynomial approximation for dense flow
- Optical flow constraint: $I_x v_x + I_y v_y + I_t = 0$
- HSV visualization (hue = direction, value = magnitude)

**Output**: Motion vectors and animated optical flow visualizations

---

### Part 2: Convolutional Neural Networks (Notebooks 4-7)

#### 4. ANN vs CNN on MNIST
**File**: `4-ANN_vs_CNN_MNIST.ipynb`

**Goals**:
- Compare fully-connected vs convolutional architectures
- Understand why CNNs excel at image tasks
- Learn spatial hierarchy and translation invariance
- Evaluate performance with confusion matrices

**Architectures**:
- **ANN**: 784 → 128 → 64 → 10 (fully-connected)
- **CNN**: Conv(32,64,256) → MaxPool → FC(128,10)

**Dataset**: MNIST (60,000 training, 10,000 test)

**Results**:
- ANN: ~97.89% test accuracy
- CNN: ~99.24% test accuracy

---

#### 5. Pretrained CNNs from PyTorch
**File**: `5-Pretrained_CNNs_from_torch.ipynb`

**Goals**:
- Explore evolution of CNN architectures (2012-2019)
- Load and use pretrained models from torchvision
- Visualize feature maps using forward hooks
- Compare accuracy vs parameters vs speed

**Models Covered**:
1. **AlexNet (2012)**: 61.1M params - Deep learning breakthrough
2. **VGG16 (2014)**: 138.4M params - Homogeneous architecture
3. **ResNet50 (2015)**: 25.6M params - Skip connections
4. **DenseNet121 (2016)**: 8.0M params - Dense connectivity
5. **MobileNetV2 (2018)**: 3.5M params - Mobile optimization
6. **EfficientNet-B0 (2019)**: 5.3M params - Compound scaling

**Dataset**: ImageNet (1000 classes, 1.2M images)

---

#### 6. Pretrained Models from PyTorch Hub
**File**: `6_Pretrained_models_from_torch_hub.ipynb`

**Goals**:
- Perform object detection with multiple architectures
- Extract instance segmentation masks
- Understand speed vs accuracy tradeoffs
- Visualize detections with bounding boxes and polygons

**Models**:
- **SSD300**: Single-stage, VGG16 backbone, balanced
- **YOLOv5s**: Extremely fast real-time detector
- **Mask R-CNN**: Two-stage with pixel-level masks

**Dataset**: COCO (80 object categories, 330K images)

---

#### 7. Fine-Tuning Ultralytics Models
**File**: `7-FineTuning_ultralytics_models.ipynb`

**Goals**:
- Fine-tune YOLO models (v5, v8, v9, v10, v11, RT-DETR)
- Prepare custom datasets in YOLO format
- Optimize hyperparameters for different scenarios
- Deploy trained detection models

**Key Techniques**:
- Transfer learning with pretrained weights
- Layer freezing strategies (backbone vs head)
- Hyperparameter recipes for small/large datasets
- Automatic hyperparameter tuning with genetic algorithms

**Output**: Custom-trained object detectors ready for deployment

---

### Part 3: Generative Models (Notebooks 8-10)

#### 8. Variational Autoencoders (VAE)
**File**: `8-VAE.ipynb`

**Goals**:
- Understand probabilistic generative modeling
- Implement encoder-decoder architecture
- Apply reparameterization trick for backprop through sampling
- Explore and interpolate in latent space

**Key Concepts**:
- **ELBO**: $\mathcal{L} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))$
- Loss = Reconstruction (BCE) + KL divergence
- Architecture: Conv encoder → 128-dim latent → Conv decoder

**Dataset**: CIFAR-10 (60,000 images, 32×32 RGB)

**Output**: Generated images and smooth latent space interpolations

---

#### 9. Deep Convolutional GAN (DCGAN)
**File**: `9-DCGAN.ipynb`

**Goals**:
- Build and train GANs from scratch
- Understand adversarial min-max game
- Apply architectural guidelines for stable training
- Generate realistic images from random noise

**Architecture**:
- **Generator**: Noise → TransposeConv layers → Tanh
- **Discriminator**: Conv layers → LeakyReLU → Sigmoid

**Key Concepts**:
- Adversarial loss: $\min_G \max_D \{D(x) - D(G(z))\}$
- Architecture: No MaxPool, BatchNorm, LeakyReLU (0.2)
- Hyperparameters: β₁=0.5, LR=0.0002

**Dataset**: CIFAR-10 (resized to 64×64)

**Output**: Generated images showing training progression

---

#### 10. Conditional GAN (cGAN)
**File**: `10-ConditionalGAN_MNIST.ipynb`

**Goals**:
- Condition GANs on class labels for targeted generation
- Generate specific digits on demand
- Implement label embedding techniques
- Compare conditional vs unconditional generation

**Key Additions**:
- Label embedding: discrete classes → dense vectors
- Concatenation: merge labels with noise (G) and images (D)
- Controlled generation of specific digit classes

**Dataset**: MNIST (10 digit classes, 28×28)

**Output**: Generate any digit (0-9) with multiple variations

---

### Part 4: Transformers & Vision-Language Models (Notebooks 11-14)

#### 11. Transformers Introduction
**File**: `11-transformers_notebook.ipynb`

**Goals**:
- Understand attention mechanism (Queries, Keys, Values)
- Learn scaled dot-product attention formula
- Load and use Vision Transformers (ViT)
- Run inference on ImageNet classes

**Key Concepts**:
- **Attention**: $\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$
- Multi-head attention with parallel projections
- ViT: Patch embedding → Transformer blocks → Classification

**Models**: Vision Transformer (ViT-B/32), ResNet50 comparison

---

#### 12. Attention Map Visualization
**File**: `12-visualization_attention_map.ipynb`

**Goals**:
- Extract attention weights from ViT layers
- Visualize where the model focuses on images
- Use forward hooks for feature extraction
- Overlay attention maps on original images

**Key Techniques**:
- Forward hook wrapper to capture attention weights
- CLS token attention to input patches
- Bilinear interpolation to image resolution
- Jet colormap overlay for visualization

**Model**: Vision Transformer (vit_base_patch16_224)

**Output**: Heatmaps showing model attention regions

---

#### 13. Fine-Tuning Vision Transformer
**File**: `13-Finetuning_ViT.ipynb`

**Goals**:
- Transfer pretrained ViT to new tasks
- Fine-tune with appropriate learning rates
- Adapt architecture for different class counts
- Evaluate on downstream datasets

**Approach**:
- Pretrain on ImageNet → Fine-tune on CIFAR-10
- Low learning rate (1e-4) for fine-tuning
- Freeze backbone, train only new classification head

**Dataset**: CIFAR-10 (50,000 training, 10,000 test)

**Results**: ~95-99% accuracy on CIFAR-10

---

#### 14. CLIP Zero-Shot Classification
**File**: `14-CLIP_zeroshot_classif.ipynb`

**Goals**:
- Use CLIP for zero-shot classification
- Design effective text prompts
- Compare image-text similarity scoring
- Evaluate without task-specific training

**Key Concepts**:
- **Zero-shot**: Classify without seeing training examples
- Text prompts: "a photo of a {class}"
- Cosine similarity between image and text embeddings
- Batch processing for efficiency

**Models**:
- OpenAI CLIP (ViT-B/32)
- OpenCLIP (ViT-B-32, LAION2B pretrained)

**Output**: Classifications with confidence scores

---

### Part 5: Diffusion Models (Notebooks 15-16)

#### 15. Diffusion Models from Scratch
**File**: `15-Diffusion_From_Scratch_FashionMNIST.ipynb`

**Goals**:
- Build DDPM (Denoising Diffusion Probabilistic Model) from scratch
- Understand forward diffusion (noise addition)
- Implement reverse diffusion (denoising)
- Train U-Net for noise prediction

**Key Concepts**:
- **Forward**: $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$
- **Reverse**: Learn $p_\theta(x_{t-1}|x_t)$ to denoise
- Training: Predict noise $\epsilon$ at each timestep
- Noise schedule: Linear progression (1000 steps)

**Architecture**:
- U-Net: Encoder → Bottleneck → Decoder
- Time embedding: Sinusoidal positional encoding
- Skip connections for spatial information

**Dataset**: FashionMNIST (60,000 items, 28×28)

**Output**: Generated fashion items, training progression

---

#### 16. Conditional Diffusion Models
**File**: `16-Conditional_Diffusion_MNIST.ipynb`

**Goals**:
- Extend DDPM with class conditioning
- Generate specific digits on demand
- Understand conditioning mechanisms in diffusion
- Compare conditional vs unconditional outputs

**Additions to Base DDPM**:
- Class embedding: discrete labels → dense vectors
- Concatenate class + time embeddings
- Propagate class info through all U-Net layers
- Learn class-specific noise patterns

**Dataset**: MNIST (10 digit classes, 60,000 training)

**Capabilities**:
- Generate specific digits (0-9) on demand
- Create multiple variations of same digit
- Interpolate between digits in latent space

---

## Learning Paths

### Path 1: Fundamentals to Advanced CNNs
Perfect for beginners starting their deep learning journey

1. Image Processing (Notebook 1)
2. ANN vs CNN (Notebook 4)
3. Pretrained CNNs (Notebook 5)
4. Vision Transformers (Notebook 11)

### Path 2: Computer Vision Engineer Track
For those focused on practical vision applications

1. Feature Matching (Notebooks 2-3)
2. Pretrained CNNs (Notebooks 5-6)
3. Fine-Tuning YOLO (Notebook 7)
4. ViT Fine-Tuning (Notebook 13)

### Path 3: Generative AI Specialist
Complete journey through generative models

1. VAE (Notebook 8)
2. DCGAN (Notebook 9)
3. Conditional GAN (Notebook 10)
4. Diffusion Models (Notebook 15)
5. Conditional Diffusion (Notebook 16)

### Path 4: Advanced Vision Research
For cutting-edge techniques and interpretability

1. Attention Visualization (Notebook 12)
2. CLIP Zero-Shot (Notebook 14)
3. All Generative Models (Notebooks 8-10, 15-16)

---

## Notebook Difficulty & Category Overview

| # | Notebook | Category | Difficulty | Key Output |
|---|----------|----------|-----------|-----------|
| 1 | Image Processing | Foundations | Beginner | Enhanced images |
| 2 | Feature Matching | Video Analysis | Intermediate | Keypoint matches |
| 3 | Optical Flow | Video Analysis | Intermediate | Motion vectors |
| 4 | ANN vs CNN | Neural Networks | Intermediate | Architecture comparison |
| 5 | Pretrained CNNs | Transfer Learning | Intermediate | Model zoo exploration |
| 6 | PyTorch Hub Models | Object Detection | Intermediate | Detection results |
| 7 | YOLO Fine-Tuning | Object Detection | Advanced | Custom detectors |
| 8 | VAE | Generative Models | Advanced | Latent interpolations |
| 9 | DCGAN | Generative Models | Advanced | Generated images |
| 10 | Conditional GAN | Generative Models | Advanced | Controlled generation |
| 11 | Transformers Intro | Transformers | Intermediate | ViT inference |
| 12 | Attention Maps | Interpretability | Intermediate | Attention heatmaps |
| 13 | ViT Fine-Tuning | Transfer Learning | Intermediate | Fine-tuned classifier |
| 14 | CLIP Zero-Shot | Vision-Language | Advanced | Zero-shot results |
| 15 | Diffusion Basics | Generative Models | Advanced | Generated samples |
| 16 | Conditional Diffusion | Generative Models | Advanced | Class-conditional outputs |

---

## Requirements

The notebooks use the following main libraries:

- **Core**: PyTorch, torchvision, NumPy, Matplotlib
- **Computer Vision**: OpenCV, scikit-image
- **Object Detection**: Ultralytics (YOLO), detectron2
- **Transformers**: timm, transformers (Hugging Face)
- **Vision-Language**: open_clip, clip
- **Utilities**: tqdm, PIL, einops

To install dependencies:

```bash
pip install torch torchvision opencv-python scikit-image matplotlib
pip install ultralytics transformers timm open-clip-torch
pip install tqdm pillow einops
```

---

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/HSLU-GENAI.git
cd HSLU-GENAI
```

2. Install dependencies (see Requirements above)

3. Launch Jupyter:
```bash
jupyter notebook
```

4. Start with the recommended learning path for your level!

---

## Course Structure

This course progressively builds knowledge:

**Weeks 1-2**: Image processing fundamentals and feature extraction
**Weeks 3-4**: Deep learning architectures (CNNs, ResNets, ViTs)
**Weeks 5-6**: Object detection and fine-tuning
**Weeks 7-9**: Generative models (VAEs, GANs)
**Weeks 10-12**: Advanced topics (Transformers, CLIP, Diffusion)

---

## Key Concepts Covered

- **Image Processing**: Convolution, filtering, morphology
- **Feature Extraction**: SIFT, ORB, optical flow
- **Deep Learning**: CNNs, ResNets, attention mechanisms
- **Transfer Learning**: Fine-tuning, domain adaptation
- **Object Detection**: YOLO, SSD, Mask R-CNN
- **Generative Models**: VAEs, GANs, Diffusion Models
- **Vision-Language**: CLIP, zero-shot learning
- **Interpretability**: Attention visualization, feature maps

---

## Datasets Used

- **MNIST**: 70,000 handwritten digits (28×28)
- **CIFAR-10**: 60,000 natural images (32×32, 10 classes)
- **FashionMNIST**: 70,000 fashion items (28×28, 10 classes)
- **ImageNet**: 1.2M images (1000 classes) - pretrained models
- **COCO**: 330K images (80 object categories) - Pretrained detection
- **Video Frames**: Custom sequences (640×360) - optical flow

---

## Contributing

This is a course repository. For questions or suggestions, please open an issue.

---

## License

Educational use only. Check individual model licenses before commercial use.

---

## Acknowledgments

Content of this course was inpired from [Deepia](https://www.youtube.com/@Deepia-ls2fo) and [3Blue1Brown](https://www.youtube.com/@3blue1brown).

---

**Happy Learning!** Start with the fundamentals and work your way to state-of-the-art generative models.

---

Some content used in ipynb

Diffusion Scheduler parameters

<img width="618" height="276" alt="image" src="https://github.com/user-attachments/assets/982c49cb-0583-494f-ab37-2e33cc839cca" />

Video example

https://github.com/user-attachments/assets/ef9ddbf3-44e8-4120-b84c-a4423b1c247f

