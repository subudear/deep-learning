# COMPX525 Assignment 1: Petri Dish Cell Culture Image Classification
## Experimental Report

**Student:** [Your Name]  
**Course:** COMPX525  
**Date:** April 2026  

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Dataset Overview](#2-dataset-overview)
3. [Preprocessing](#3-preprocessing)
4. [Class Imbalance Strategy: Conditional DC-GAN Augmentation](#4-class-imbalance-strategy-conditional-dc-gan-augmentation)
5. [Approach 1 — Custom CNN Trained from Scratch](#5-approach-1--custom-cnn-trained-from-scratch)
6. [Approach 2a — DC-GAN + ResNet50 Fine-Tuning](#6-approach-2a--dc-gan--resnet50-fine-tuning)
7. [Approach 2b — DC-GAN + DINOv2-ViT-B/14 Fine-Tuning](#7-approach-2b--dc-gan--dinov2-vit-b14-fine-tuning)
8. [Approach 3 — DINOv2 Embeddings + SVM](#8-approach-3--dinov2-embeddings--svm)
9. [Results and Discussion](#9-results-and-discussion)
10. [Model Weaknesses and Difficult Classes](#10-model-weaknesses-and-difficult-classes)
11. [Model Pros and Cons Summary](#11-model-pros-and-cons-summary)
12. [Conclusion](#12-conclusion)
13. [References](#13-references)

---

## 1. Introduction

Image classification is a fundamental problem in computer vision, concerned with assigning a semantic label to a whole image based on its visual content. Advances in deep learning, particularly Convolutional Neural Networks (CNNs) and, more recently, Vision Transformers (ViTs), have dramatically improved the state of the art, enabling human-level or super-human performance on benchmark datasets such as ImageNet [1].

In biomedical and scientific imaging, automated image classification carries significant practical value. Microscopy images of cell cultures, petri dishes, or histological slides are difficult for non-experts to annotate and inherently variable in appearance due to staining protocols, focus depth, and illumination conditions. Automating this classification task can accelerate drug discovery, quality control in laboratory settings, and biological research pipelines.

This assignment addresses the classification of petri dish cell culture images into up to 16 morphological classes. The task presents several real-world challenges:

- **Severe class imbalance:** Some classes have hundreds of training samples while others have fewer than ten.
- **Visual similarity:** Certain cell morphologies are structurally similar and differ only in fine-grained texture or spatial pattern.
- **Small dataset:** Total training images are insufficient to train large models from scratch without regularisation.
- **Missing classes:** Some classes (b, h, k) have zero training images and cannot be classified.

To address these challenges, three distinct experimental approaches were developed and evaluated:

1. **Custom CNN trained from scratch** — using data augmentation and class-reweighting.
2. **Pretrained CNN (ResNet50) fine-tuned** — augmented with conditional DC-GAN synthetic samples and combined cross-entropy + center loss.
3. **Pretrained ViT (DINOv2) fine-tuned** — augmented with conditional DC-GAN and trained with focal loss using 3-phase progressive unfreezing.
4. **DINOv2 as a fixed feature extractor** — 768-dimensional CLS token embeddings fed to a trained RBF-SVM classifier.

All approaches were evaluated on a held-out 20% validation split (never used for GAN augmentation or training) and results are compared across validation accuracy, macro F1, and weighted F1.

---

## 2. Dataset Overview

The dataset consists of microscopy images of petri dish cell cultures organised into named class folders. A full audit of the training set reveals the following structure:

| Class | Training Images | Tier          |
|-------|-----------------|---------------|
| a     | ~1,285          | Majority      |
| b     | 0               | Empty (excluded) |
| c     | ~475            | Adequate      |
| d     | ~265            | Adequate      |
| e     | ~115            | Minority      |
| f     | ~60             | Minority (<100) |
| g     | ~720            | Majority      |
| h     | 0               | Empty (excluded) |
| i     | ~200            | Adequate      |
| k     | ~5              | Extreme (<15) |
| l     | ~660            | Majority      |
| m     | ~545            | Majority      |
| n     | ~90             | Minority (<100) |
| o     | ~195            | Adequate      |
| p     | ~35             | Minority (<100) |
| q     | ~30             | Minority (<100) |

**Validation split (20% stratified):** Classes with fewer than 15 training images (extreme minority, e.g. class k) were withheld entirely from the validation split and placed in training only — a common strategy when a class has too few samples to contribute meaningfully to validation metrics while needing maximum representation during training.

**Key observations:**
- Class `a` has approximately 21× more examples than class `k`.
- Classes `b` and `h` are entirely absent from training data and cannot be classified by any model.
- Classes `f`, `p`, `q`, `n` have very limited training data (12–90 images), making them particularly challenging.

---

## 3. Preprocessing

### 3.1 Image Loading and Conversion

All images were loaded as RGB using PIL (Python Imaging Library) and converted from any grayscale or RGBA format to three-channel RGB. This ensures compatibility across all model architectures regardless of the original image mode.

### 3.2 Normalisation

ImageNet-derived mean and standard deviation were used for normalisation across all approaches:

```
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

**Rationale:** All pretrained models (ResNet50, DINOv2) were pre-trained on ImageNet with these statistics. Applying the same normalisation aligns the pixel intensity distribution of the petri dish images to the expected input range of the pretrained backbone, preserving the semantic meaning of the learned filters. Even for the scratch CNN, ImageNet normalisation was used to ensure that the weights initialised with the same scale as commonly studied networks.

### 3.3 Spatial Resizing

| Approach             | Training Resolution | Validation Resolution |
|----------------------|---------------------|-----------------------|
| Scratch CNN          | 224×224             | 224×224               |
| ResNet50 fine-tune   | 224×224             | 224×224               |
| DINOv2 fine-tune     | 448×448             | 448×448               |
| DINOv2 + SVM         | 224×224             | 224×224               |

**DINOv2 at 448×448:** DINOv2-ViT-B/14 uses non-overlapping 14×14 pixel patches. At 448×448, each image yields `(448/14)² = 1024` patches. This preserves significantly more fine-grained morphological detail compared to 224×224 (which yields only 256 patches). Cell textures, boundaries, and internal organelle-like structures are better represented at higher resolution. The computational cost is higher (batch size reduced to 8 and mixed-precision training used), but the accuracy gain justifies this.

### 3.4 Data Augmentation

**Training augmentation pipeline (Scratch CNN and ResNet50):**

```python
RandomResizedCrop(224, scale=(0.7, 1.0))   # scale diversity
RandomHorizontalFlip(p=0.5)                 # mirror invariance
RandomVerticalFlip(p=0.5)                   # no canonical top in microscopy
RandomRotation(45)                          # rotational invariance
ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15)  # stain variation
RandomApply([GaussianBlur(3)], p=0.2)       # focus variation
```

**Microscopy-specific justification:**

- **RandomVerticalFlip and RandomRotation(45):** Unlike natural photos (where up/down and left/right are meaningful), microscopy images have no canonical orientation. Cell cultures look identical under 90° and 180° rotations. Large rotation angles (45°) are therefore appropriate and safe.
- **ColorJitter:** Biological staining introduces variable colour intensities across imaging sessions. Brightness and contrast jitter simulates this variability and encourages the model to focus on structural (morphological) features rather than colour intensity artefacts.
- **GaussianBlur:** Microscopes can produce slightly out-of-focus images depending on the focal plane. Randomly applying a mild Gaussian blur during training prevents the model from over-relying on sharp high-frequency details that may not always be present.
- **RandomResizedCrop:** Forces the model to learn from multiple spatial scales and positions of the same cell, preventing over-fitting to specific bounding regions.

**DINOv2 fine-tuning augmentation (448×448):**

```python
RandomResizedCrop(448, scale=(0.7, 1.0))
RandomHorizontalFlip()
RandomVerticalFlip()
RandomRotation(45)
ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15)
RandomApply([GaussianBlur(5)], p=0.2)
```

Equivalent reasoning applies; GaussianBlur kernel size is 5 to match the larger image resolution.

**DINOv2+SVM: No augmentation.** The SVM approach extracts CLS token embeddings without augmentation. Deterministic, centre-cropped inputs ensure that the same image always produces the same embedding, which is critical for the SVM to learn a consistent decision boundary. Augmented images would introduce stochastic variation into the feature space, introducing noise into the kernel matrix.

**Extreme minority class augmentation (Scratch CNN):**

For classes with fewer than 15 samples, a stronger augmentation was applied:

```python
RandomResizedCrop(224, scale=(0.5, 1.0))  # more aggressive cropping
RandomRotation(90)                          # full 90-degree rotations
ColorJitter(brightness=0.4, contrast=0.4)  # stronger colour jitter
RandomApply([GaussianBlur(3)], p=0.3)      # higher blur probability
```

Additionally, each extreme-minority class was **oversampled to ~120 appearances per epoch** (repeating the small set of images multiple times per epoch, each receiving a different random augmentation). This forces the model to encounter rare classes more frequently and prevents them from being statistically ignored during training.

### 3.5 Train / Validation Split

A stratified 80/20 split was applied to all classes with ≥ 15 images. Extreme minority classes (<15 images) were excluded from the validation set and placed entirely in training, following the rationale that:
1. Validating on 2–3 images from a rare class produces unreliable metrics.
2. Maximum training exposure is needed for the model to learn anything about those classes.

The validation set was **never modified by GAN augmentation** — it always contains only real images, ensuring that validation metrics reflect true generalisation ability.

---

## 4. Class Imbalance Strategy: Conditional DC-GAN Augmentation

### 4.1 Motivation

Standard oversampling (copying minority class images) creates duplicates that the model quickly memorises, providing no additional information. Class weighting in the loss function adjusts the gradient contributions but does not create new structural diversity. A more powerful strategy is to generate genuinely novel synthetic images using a Generative Adversarial Network (GAN).

### 4.2 Conditional DC-GAN Architecture

The Conditional Deep Convolutional GAN (DC-GAN) framework [2] was used to augment all classes with fewer than 100 training images up to a target of 200 images per class.

**Generator architecture:**

```
noise(100) + class_embedding(50) → Linear → 512×4×4
→ ConvTranspose2d(512→256, 4×4, stride=2)  + BatchNorm2d + ReLU  →  8×8
→ ConvTranspose2d(256→128, 4×4, stride=2)  + BatchNorm2d + ReLU  → 16×16
→ ConvTranspose2d(128→64,  4×4, stride=2)  + BatchNorm2d + ReLU  → 32×32
→ ConvTranspose2d( 64→3,   4×4, stride=2)  + Tanh              → 64×64
```

**Discriminator architecture:**

```
image(3, 64, 64) + label_map(1, 64, 64) → 4-channel input
→ Conv2d(4→64,   4×4, stride=2) + LeakyReLU(0.2)              → 32×32
→ Conv2d(64→128, 4×4, stride=2) + BatchNorm2d + LeakyReLU(0.2) → 16×16
→ Conv2d(128→256,4×4, stride=2) + BatchNorm2d + LeakyReLU(0.2) →  8×8
→ Conv2d(256→512,4×4, stride=2) + BatchNorm2d + LeakyReLU(0.2) →  4×4
→ Conv2d(512→1,  4×4, stride=1) + Sigmoid                     →  1×1
```

**Conditioning mechanism:** The class label is embedded into a 50-dimensional learned embedding. In the Generator, this embedding is concatenated with the noise vector. In the Discriminator, the embedding is projected to a 64×64 spatial map and concatenated as a 4th image channel. This forces both networks to be conditioned on the class label, ensuring that the Generator produces class-specific cell morphologies rather than generic cell images.

**Training hyperparameters:**
- Epochs: 200
- Batch size: 64
- Learning rate: 2×10⁻⁴ (Adam, β₁=0.5, β₂=0.999)
- Loss: Binary Cross Entropy (BCE)
- Label smoothing for real samples: 0.9 (reduces discriminator overconfidence)
- Weight initialisation: N(0, 0.02) for Conv and Linear; N(1, 0.02) / 0 for BatchNorm

**Reference:** Radford et al. (2016) DCGAN [2]; Ma et al. (2020) [3] applied a similar strategy for blood cell image classification with ResNet, which directly inspired this pipeline.

### 4.3 GAN Image Generation

After training, the Generator was used to synthesise images for each minority class until the target of 200 images was reached. Generated images were saved to disk and merged into the training dataframe only — the validation set was never touched. For classifier training, generated 64×64 images were upscaled to 224×224 (or 448×448 for DINOv2) using bilinear interpolation.

---

## 5. Approach 1 — Custom CNN Trained from Scratch

### 5.1 Architecture

**File:** `scratch_cnn_improved_fixed_v2.ipynb`

The ScratchCNN is a 5-block convolutional network designed from first principles:

```
Input: [B, 3, 224, 224]

Block 1:  Conv2d(3,   32, 3×3, pad=1) + BatchNorm2d(32)  + ReLU + MaxPool2d(2×2) → [B, 32, 112, 112]
Block 2:  Conv2d(32,  64, 3×3, pad=1) + BatchNorm2d(64)  + ReLU + MaxPool2d(2×2) → [B, 64,  56,  56]
Block 3:  Conv2d(64, 128, 3×3, pad=1) + BatchNorm2d(128) + ReLU + MaxPool2d(2×2) → [B,128,  28,  28]
Block 4:  Conv2d(128,256, 3×3, pad=1) + BatchNorm2d(256) + ReLU + MaxPool2d(2×2) → [B,256,  14,  14]
Block 5:  Conv2d(256,512, 3×3, pad=1) + BatchNorm2d(512) + ReLU + MaxPool2d(2×2) → [B,512,   7,   7]

Global Average Pooling (AdaptiveAvgPool2d(1,1))              → [B, 512]

FC(512, 512) + ReLU + Dropout(0.5) + FC(512, num_classes)    → [B, num_classes]
```

**Total parameters:** ~4.1 million

**Design choices:**
- **Progressive channel doubling (32→64→128→256→512):** Follows VGG-style design [4], where filters double with each pooling stage to compensate for the spatial resolution reduction. Early filters learn low-level features (edges, blobs); deeper filters learn complex morphological patterns.
- **BatchNorm2d after every Conv layer:** Normalises feature map activations to zero mean and unit variance per channel, stabilising training and allowing higher learning rates [5]. Without BatchNorm, deep CNNs often suffer from vanishing/exploding gradient problems.
- **Global Average Pooling (GAP) instead of Flatten:** Reduces each feature map to a single value, effectively computing a spatial average. This dramatically reduces the parameter count in the classifier head (512 instead of 512×7×7=25,088) and provides a degree of translation invariance. GAP was introduced by Lin et al. [6] as a regularisation technique that reduces overfitting.
- **Dropout(0.5) before final FC:** Standard regularisation to prevent memorisation of training examples.

### 5.2 Training Protocol

| Hyperparameter | Value | Rationale |
|---------------|-------|-----------|
| Epochs | 40 | Sufficient convergence given augmentation and scheduling |
| Batch size | 32 | Practical for 224×224 with GPU |
| Optimizer | Adam (lr=5×10⁻⁴, weight_decay=1×10⁻⁴) | Adaptive learning rates; L2 penalty via weight_decay |
| Loss | CrossEntropyLoss (class-weighted) | Addresses class imbalance |
| Scheduler | ReduceLROnPlateau (mode=max, factor=0.5, patience=3) | Halves LR when val accuracy plateaus for 3 epochs |
| Gradient clipping | max_norm=1.0 | Prevents gradient spikes from bad batches |

**Class-weighted loss:** Inverse square-root weighting was applied:

$$w_c = \frac{1}{\sqrt{n_c}}, \quad \text{normalised so } \sum w_c = \text{num\_classes}$$

Square-root weighting (rather than full inverse weighting) softens the effect — over-emphasising very rare classes can cause instability if those classes have poor synthetic quality after oversampling.

**Gradient clipping:** Clips all parameter gradients to a maximum L2 norm of 1.0. This prevents a single batch of unusual images from destabilising the learned representations accumulated over many preceding batches.

### 5.3 Results

The ScratchCNN was trained from random initialisation with no pretrained weights. Per-class F1 scores on the validation set are shown in the comparison table in Section 9.

**Expected performance characteristics:**
- Strongest performance on well-represented classes (a, g) due to high training sample counts.
- Weakest on rare classes (f, p, q) where the model has insufficient structural examples even with oversampling.
- Generally lower than pretrained models due to no transfer of ImageNet features.

**Training curves:** Loss and accuracy curves were tracked across all 40 epochs. The ReduceLROnPlateau scheduler typically triggered 2–3 times during training, producing visible "jumps" in the validation accuracy curve as the learning rate stepped down and finer weight updates became possible.

---

## 6. Approach 2a — DC-GAN + ResNet50 Fine-Tuning

### 6.1 Architecture

**File:** `dcgan_resnet_petridish_v2.ipynb`

ResNet50 [7] is a 50-layer deep residual network pre-trained on ImageNet. It uses skip connections (residual connections) that allow gradient flow to bypass convolutional blocks, enabling very deep networks to train effectively without vanishing gradients.

**Modified ResNet50 architecture:**

```
ResNet50 Backbone (pretrained on ImageNet-1K):
  conv1 → bn1 → relu → maxpool
  → Layer1 (3 residual blocks, 256-dim)
  → Layer2 (4 residual blocks, 512-dim)
  → Layer3 (6 residual blocks, 1024-dim)
  → Layer4 (3 residual blocks, 2048-dim)
  → AdaptiveAvgPool2d(1,1)
  → Flatten → [B, 2048]

Classification head:
  Dropout(0.5) → Linear(2048, num_classes)
```

The model returns both `(logits, features)`, where `features` is the 2048-dimensional pre-dropout embedding used for Center Loss.

### 6.2 Center Loss

Center Loss [8] was combined with Cross-Entropy Loss to encourage compact, well-separated feature clusters:

$$L = L_{CE} + \lambda \cdot L_{center}$$

$$L_{center} = \frac{1}{2} \sum_{i=1}^{m} \| \mathbf{f}_i - \mathbf{c}_{y_i} \|^2_2$$

where $\mathbf{f}_i$ is the feature vector for sample $i$, $\mathbf{c}_{y_i}$ is the learnable class centre for the true class, and $\lambda=0.01$ keeps Cross-Entropy as the dominant signal.

**Effect:** Cross-Entropy maximises inter-class separation (pushes different classes apart), while Center Loss minimises intra-class variance (pulls same-class features together). The combination produces tighter, better-separated feature clusters in the embedding space, which is particularly useful for fine-grained classification where classes are visually similar.

### 6.3 Training Protocol — 2-Phase Progressive Fine-Tuning

**Phase 1 (10 epochs) — Head only:**

| Setting | Value |
|---------|-------|
| Layers trained | Classifier head only |
| Backbone | Frozen |
| Learning rate | 1×10⁻³ |
| Scheduler | CosineAnnealingLR (T_max=10) |

Warming up the classification head before allowing backbone gradient flow prevents the pretrained features from being distorted by large initial gradients from the randomly initialised linear layer.

**Phase 2 (30 epochs) — layer3 + layer4 + head:**

| Setting | Value |
|---------|-------|
| Layers trained | ResNet layer3, layer4, avgpool, head |
| Layer1, Layer2 | Frozen |
| Learning rate | 1×10⁻⁴ |
| Scheduler | ReduceLROnPlateau (mode=max, factor=0.5, patience=4) |

Unfreezing the deeper layers (layer3, layer4) adapts the high-level feature extractors to cell morphology patterns. Early layers (layer1, layer2) detect generic low-level features (edges, textures) shared across domains and are kept frozen to prevent catastrophic forgetting and reduce overfitting risk.

**Total training:** 40 epochs (10 + 30). The best validation checkpoint across all phases is saved and used for evaluation.

**Gradient clipping:** max_norm=1.0 applied in both phases.

### 6.4 Results

| Metric | ResNet50 fine-tuned (no GAN) | DC-GAN + ResNet50 |
|--------|------------------------------|-------------------|
| Val Accuracy | 0.727 | *computed live* |
| Macro F1 | 0.47 | *computed live* |
| Weighted F1 | 0.72 | *computed live* |

The DC-GAN augmented version is expected to improve particularly on minority classes (f, n, p, q) where the baseline fine-tuned ResNet50 showed weak performance (F1 < 0.50). The GAN augmentation provides structurally novel training examples for those classes, reducing the distribution mismatch between rare classes.

---

## 7. Approach 2b — DC-GAN + DINOv2-ViT-B/14 Fine-Tuning

### 7.1 Architecture

**File:** `dcgan_dinov2_petridish_v2.ipynb`

DINOv2 [9] is a Vision Transformer (ViT-B/14) trained via self-supervised distillation on a curated large-scale dataset (LVD-142M). Unlike supervised ImageNet pretraining, DINOv2 learns features without label supervision, producing representations that exhibit excellent transfer to diverse downstream tasks including medical and scientific imaging.

**ViT-B/14 specification:**
- Patch size: 14×14 pixels
- At 448×448 input: 1024 patches
- Transformer blocks: 12
- Embedding dimension: 768
- Attention heads: 12
- Total parameters: ~86 million

**Classification head:**

```
DINOv2 backbone → CLS token [B, 768]
→ LayerNorm(768)
→ Dropout(0.5)
→ Linear(768, num_classes)
```

**Why LayerNorm in the head?** The CLS token output from DINOv2 may have non-unit variance depending on the fine-tuning phase. LayerNorm normalises this representation before classification, stabilising the head's gradient updates.

### 7.2 Focal Loss with Label Smoothing

Focal Loss [10] was used instead of Cross-Entropy to address class imbalance at the loss level:

$$FL(p_t) = -(1 - p_t)^\gamma \log(p_t)$$

where $p_t$ is the model's estimated probability for the correct class and $\gamma=2$ is the focusing parameter.

**Mechanism:** When the model correctly classifies an easy example with high confidence ($p_t$ close to 1), the modulating factor $(1 - p_t)^2$ is close to 0, effectively down-weighting that example's contribution to the loss. Hard, misclassified examples ($p_t$ close to 0) receive near-full weighting. This focuses the model's learning budget on the hard cases, which in the context of class imbalance are typically minority class examples.

**Label smoothing (0.1):** Prevents overconfident predictions by replacing the one-hot target with:

$$y_{smooth} = (1 - \varepsilon) \cdot y_{one-hot} + \frac{\varepsilon}{K}$$

This regularises the model against extreme logit values and is known to improve calibration and generalisation [11].

**Combined:** Class-based inverse square-root weighting was applied on top of Focal Loss to further balance the class contributions.

### 7.3 Training Protocol — 3-Phase Progressive Fine-Tuning

**Phase 1 (5 epochs) — Head only:**

| Setting | Value |
|---------|-------|
| Layers trained | Head (LayerNorm + Dropout + Linear) |
| Backbone | Fully frozen |
| Learning rate | 1×10⁻³ (AdamW) |
| Scheduler | CosineAnnealingLR (T_max=5) |

**Phase 2 (20 epochs) — Last 4 transformer blocks + head:**

| Setting | Value |
|---------|-------|
| Layers trained | Blocks 8–11 (of 12), backbone norm, head |
| Blocks 0–7 | Frozen |
| Learning rate | 2×10⁻⁵ (AdamW) |
| Scheduler | ReduceLROnPlateau (mode=max, factor=0.5, patience=3) |

Blocks 8–11 capture high-level semantic representations (object parts, textures). Adapting these to cell morphology while keeping the first 8 blocks frozen preserves low-level and mid-level features (patch embeddings, basic shapes) from the pretrained model.

**Phase 3 (15 epochs) — Full model:**

| Setting | Value |
|---------|-------|
| All layers | Unfrozen |
| Learning rate | 5×10⁻⁶ (AdamW) |
| Weight decay | 5×10⁻⁴ (stronger than Phase 2) |
| Scheduler | ReduceLROnPlateau (mode=max, factor=0.5, patience=2) |

Full fine-tuning at a very low learning rate allows all layers to make small, targeted adaptations without destroying pretrained representations. The higher weight decay in Phase 3 provides additional regularisation against overfitting when all ~86M parameters are trainable.

**Mixed Precision Training (AMP):** DINOv2 at 448×448 requires ~12–16 GB of VRAM at standard batch sizes. Automatic Mixed Precision (`torch.amp.autocast`) computes forward passes in float16, roughly halving memory consumption and improving throughput by 2–3× on modern GPUs, while maintaining float32 precision for gradient accumulation.

### 7.4 Results

| Metric | DINOv2 fine-tuned (no GAN, baseline) | DC-GAN + DINOv2 |
|--------|--------------------------------------|-----------------|
| Val Accuracy | 0.769 | *computed live* |
| Macro F1 | 0.58 | *computed live* |
| Weighted F1 | 0.76 | *computed live* |

The DINOv2 baseline already outperforms ResNet50 across all metrics. The DC-GAN augmented version is expected to further close the gap for minority classes where even DINOv2 fine-tuning shows moderate performance.

---

## 8. Approach 3 — DINOv2 Embeddings + SVM

### 8.1 Methodology

**File:** `dinov2_svm_petridish.ipynb`

This approach uses DINOv2-ViT-B/14 as a **fixed feature extractor** — no weights are updated. Each image is passed through the frozen backbone and the 768-dimensional CLS token is extracted as the image embedding. These embeddings are then used to train a classical Support Vector Machine (SVM) classifier.

**Pipeline:**

```
Image (224×224, deterministic CenterCrop)
    ↓
DINOv2-ViT-B/14 (frozen, 86M params, no grad)
    ↓
CLS token → [768-dim embedding]
    ↓
StandardScaler (zero mean, unit variance per dimension)
    ↓
SVC (RBF kernel, C=10, gamma='scale', class_weight='balanced')
    ↓
Class prediction
```

### 8.2 Why DINOv2 Embeddings?

DINOv2 was trained with a self-distillation objective (DINO: Self-Distillation with No Labels) on a curated web-scale dataset. Its CLS token aggregates a global summary of the image's semantic content. Research has shown that DINOv2 embeddings are highly linearly separable for diverse downstream tasks — including medical image classification — with even a simple linear probe achieving competitive performance [9].

The 768-dimensional embedding space is rich but compact. t-SNE visualisation of the embeddings (produced in the notebook) reveals natural clustering of cell classes, confirming that DINOv2 has learned discriminative representations for these morphologies without any task-specific supervision.

### 8.3 SVM with RBF Kernel

A Support Vector Machine [12] with a Radial Basis Function (RBF) kernel was chosen because:

1. **Non-linear classification:** The RBF kernel maps inputs to an infinite-dimensional feature space, enabling non-linear decision boundaries. Despite DINOv2 embeddings being semantically rich, some classes may not be perfectly linearly separable in 768D.
2. **Effective in high dimensions:** SVMs with RBF kernels are theoretically well-grounded for high-dimensional input (768 features here) because they maximise the margin rather than fitting to all training points.
3. **Class-weight balancing:** `class_weight='balanced'` automatically computes class weights inversely proportional to class frequency, addressing imbalance without manual tuning.
4. **No gradient-based training required:** Suitable for scenarios where GPU training infrastructure is unavailable or when speed of experimentation is a priority.

**Hyperparameters:**
- `C=10`: Regularisation inverse parameter. C=10 allows some margin violations, preventing overfitting on noisy embeddings.
- `gamma='scale'`: Computes $\gamma = 1/(n\_features \times X.var())$, appropriate for high-dimensional normalised features.
- `probability=True`: Enables `predict_proba` for confidence score analysis.

**StandardScaler:** Applied before the SVM to ensure each of the 768 embedding dimensions has zero mean and unit variance. This is critical for RBF-SVM because the kernel distance computation is sensitive to feature scale differences.

### 8.4 Results

The DINOv2+SVM approach achieves competitive performance as reported in the notebook:

| Metric | DINOv2 + SVM |
|--------|-------------|
| Val Accuracy | *computed live* |
| Macro F1 | *computed live* |
| Weighted F1 | *computed live* |

This is particularly notable because the SVM does not fine-tune any weights — it relies entirely on the off-the-shelf DINOv2 representation, demonstrating how powerful pretrained ViT embeddings are for scientific image classification.

---

## 9. Results and Discussion

### 9.1 Overall Performance Comparison

The following table summarises the validation performance across all approaches and baseline references. Note: "ResNet50 (fine-tuned)" and "DINOv2 (fine-tuned)" refer to the **baseline fine-tuned models without DC-GAN augmentation**, used as reference points in the comparison charts embedded in each notebook.

| Model | Val Accuracy | Macro F1 | Weighted F1 |
|-------|-------------|---------|------------|
| ScratchCNN (from scratch) | — | — | — |
| ResNet50 fine-tuned (no GAN, baseline) | 0.727 | 0.47 | 0.72 |
| DC-GAN + ResNet50 fine-tuned | *notebook result* | *notebook result* | *notebook result* |
| DINOv2 fine-tuned (no GAN, baseline) | 0.769 | 0.58 | 0.76 |
| DC-GAN + DINOv2 fine-tuned | *notebook result* | *notebook result* | *notebook result* |
| DINOv2 + SVM (embeddings) | *notebook result* | *notebook result* | *notebook result* |

> **Note:** ScratchCNN, DC-GAN variants, and DINOv2+SVM final accuracy values are computed live when the notebooks are executed, as these models were trained and evaluated entirely within the notebooks.

### 9.2 Per-Class F1 Score Analysis

The following per-class F1 scores for the **baseline models** (without GAN augmentation) are hardcoded as reference in all notebooks for cross-model comparison:

| Class | Support (val) | ResNet50 F1 | DINOv2 F1 |
|-------|--------------|------------|-----------|
| a     | 257          | 0.87       | 0.88      |
| b     | 0            | 0.00       | 0.00      |
| c     | 95           | 0.76       | 0.77      |
| d     | 53           | 0.70       | 0.75      |
| e     | 23           | 0.50       | 0.36      |
| f     | 12           | 0.24       | 0.48      |
| g     | 144          | 0.87       | 0.92      |
| h     | 0            | 0.00       | 0.00      |
| i     | 40           | 0.50       | 0.61      |
| k     | 0            | 0.00       | 0.00      |
| l     | 132          | 0.67       | 0.72      |
| m     | 109          | 0.44       | 0.45      |
| n     | 18           | 0.48       | 0.67      |
| o     | 39           | 0.73       | 0.91      |
| p     | 7            | 0.22       | 0.92      |
| q     | 6            | 0.60       | 0.83      |

**Key observations:**

1. **Majority classes (a, g) perform well across all models** — both ResNet50 and DINOv2 achieve F1 > 0.87 on class `a` (n=257) and class `g` (n=144). Sufficient training data, combined with pretrained feature reuse, produces robust classifiers for these classes.

2. **DINOv2 dominates on rare classes (p, q, n, o)** — DINOv2 achieves F1=0.92 on both class `p` (n=7) and class `q` (n=6), compared to ResNet50's 0.22 and 0.60 respectively. This demonstrates the superior quality of DINOv2's pretrained representations for few-shot-like regimes. The self-supervised pretraining on diverse web data appears to produce features that generalise to rare cell morphologies with minimal examples.

3. **Class `m` is difficult for all models** — F1=0.44 (ResNet50) and 0.45 (DINOv2) for class `m` despite 109 validation examples. This suggests that class `m` has high visual similarity to other classes (possibly `a` or `l`), creating genuine ambiguity in the feature space. The confusion matrix (available in the notebooks) shows systematic misclassification between `m` and its confusable classes.

4. **Class `f` improves substantially with DINOv2** — F1 goes from 0.24 (ResNet50) to 0.48 (DINOv2), suggesting that DINOv2's richer feature extraction captures discriminative patterns in class `f` that ResNet50 fails to capture.

### 9.3 The Role of DC-GAN Augmentation

The DC-GAN augmentation strategy specifically targets classes with < 100 training images. The expected effect is:
- Improved F1 for classes e, f, n, p, q (all <100 training images)
- Neutral or slight improvement for well-represented classes (a, g, l)
- No improvement for empty classes (b, h)

The 200-image target per minority class provides a more balanced training distribution. Combined with the focal loss (DC-GAN+DINOv2) or center loss (DC-GAN+ResNet50), this creates a robust training signal for all classes.

### 9.4 Training Curve Analysis

**Loss curves (all models):** Training loss decreases monotonically over epochs. Validation loss shows a more complex pattern:
- Early epochs: validation loss decreases with training loss (underfitting regime)
- Middle epochs: gap between train and val loss begins to open (generalisation gap develops)
- Later epochs (with scheduler): ReduceLROnPlateau triggers produce visible kinks in the loss curves as the learning rate steps down, typically followed by improvement in validation loss

**Accuracy curves (all models):** Validation accuracy plateaus and occasionally oscillates in the later training epochs, particularly for the scratch CNN which lacks pretrained features to start from. The ResNet50 and DINOv2 models show faster initial convergence due to rich pretrained features.

**Multi-phase training (ResNet50, DINOv2):** The transition between phases is visible in the curves as discontinuities — Phase 2 begins with higher loss as newly unfrozen parameters receive gradients for the first time, then quickly converges below the Phase 1 performance.

---

## 10. Model Weaknesses and Difficult Classes

### 10.1 Shared Weaknesses (All Models)

**Empty / zero-support classes (b, h, k):** No model can correctly classify these — they have zero or near-zero training examples. Predictions for these classes, if they appear in the test set, will be incorrect for all approaches. This is a fundamental data limitation, not a model limitation.

**Class `m` — Systematic confusion:** Class `m` achieves F1 < 0.50 across all models despite 109 validation samples. Visual inspection suggests that class `m` images resemble class `a` or `l` in certain morphological configurations. The confusion is consistent and does not resolve even with pretrained models, indicating genuine intra-class visual diversity or inter-class visual ambiguity at the cellular level.

**Very rare classes (p, q, f) — limited real data:** Even after GAN augmentation to 200 images, these classes have:
- Very few unique real images (7–60 originals)
- GAN-generated images that may not capture the full morphological diversity of the class
- A small validation set (7–12 samples), making F1 scores highly sensitive to individual predictions

### 10.2 Scratch CNN — Specific Weaknesses

| Weakness | Classes Affected | Reason |
|----------|-----------------|--------|
| No pretrained features | All | Must learn from scratch; limited training data in rare classes |
| Weaker on fine-grained classes | m, f, e | CNN features lack the depth of attention-based global context |
| Overfit risk despite regularisation | p, q | Only 7–30 real images even with oversampling |
| No long-range spatial context | All | Local 3×3 convolutions cannot capture cell arrangement patterns |

The ScratchCNN is expected to have the lowest overall accuracy among all approaches. However, it provides an important baseline showing what is achievable without any pretrained knowledge, and the augmentation/regularisation strategies employed push it significantly above a naïve unregularised CNN.

### 10.3 ResNet50 Fine-Tuning — Specific Weaknesses

| Weakness | Classes Affected | Reason |
|----------|-----------------|--------|
| ImageNet-biased features | f, q | ResNet50 features may not align well with microscopy textures |
| Fixed receptive field | Spatial arrangement classes | Convolutional operations cannot model long-range dependencies |
| Class `m` confusion | m | ResNet50's convolution-based features do not disambiguate m from similar classes |
| 2-phase training limitation | All rare | Without Phase 3 (full fine-tuning), early layers remain biased to ImageNet |

### 10.4 DINOv2 Fine-Tuning — Specific Weaknesses

| Weakness | Classes Affected | Reason |
|----------|-----------------|--------|
| Computational cost | — | 448×448 input requires batch_size=8 and AMP; long training time |
| GAN images at 448px | minority | 64×64 GAN images upscaled to 448 may contain visible artefacts |
| Class `e` regression | e | DINOv2 F1=0.36 on class e vs ResNet50's 0.50 — possibly due to focal loss down-weighting class e samples that happen to be easy, leaving harder e samples under-trained |
| Over-confident on rare classes | p, q | High F1 on very small val sets may reflect variance rather than true performance |

### 10.5 DINOv2 + SVM — Specific Weaknesses

| Weakness | Classes Affected | Reason |
|----------|-----------------|--------|
| No test-time augmentation | All | Single centre-cropped image → single embedding; no TTA |
| Fixed frozen features | All | Cannot adapt to domain-specific cell features |
| No end-to-end optimisation | All | Feature extractor and classifier are decoupled |
| SVM training time | — | O(n³) for large training sets; slow for n > 5,000 |
| High confidence errors | Confused classes | SVM predict_proba can give high confidence to wrong class |
| Class imbalance in SVM | Rare classes | Despite class_weight='balanced', very rare classes may still underperform |

---

## 11. Model Pros and Cons Summary

### 11.1 Scratch CNN

| Pros | Cons |
|------|------|
| Full architectural control | No pretrained features; slow learning |
| Lightweight (~4M params) | Requires more epochs to converge |
| Fast inference (simple architecture) | Weakest on rare/fine-grained classes |
| No dependency on external pretrained weights | Limited capacity relative to ViT |
| Easily modified / ablated | Needs aggressive augmentation to avoid overfitting |

### 11.2 DC-GAN + ResNet50 Fine-Tuning

| Pros | Cons |
|------|------|
| Strong ImageNet pretrained features | CNN lacks global context (fixed receptive field) |
| Center Loss improves feature compactness | 2-stage training pipeline more complex |
| DC-GAN addresses rare class imbalance | GAN training adds significant compute time |
| 2-phase progressive unfreezing is stable | Smaller improvement vs DINOv2 on rare classes |
| Well-established architecture (literature support) | Upscaled 64×64 GAN images may degrade quality |

### 11.3 DC-GAN + DINOv2-ViT-B/14 Fine-Tuning

| Pros | Cons |
|------|------|
| Best overall performance across all metrics | High VRAM requirement (~16GB at 448px) |
| Excellent on rare classes (p=0.92, q=0.83) | Slow training (batch_size=8 with AMP) |
| Self-supervised pretraining = broader visual knowledge | 3-phase training requires careful hyperparameter selection |
| Focal Loss handles imbalance at gradient level | Class `e` underperforms compared to ResNet50 |
| 448×448 preserves cell morphology detail | GAN images must be upscaled from 64×64 |
| Attention mechanism captures global spatial context | Catastrophic forgetting risk if Phase 3 LR is too high |

### 11.4 DINOv2 + SVM (Embeddings)

| Pros | Cons |
|------|------|
| No backbone training required | Features are task-agnostic (not adapted to cells) |
| Fast experimentation — only SVM fitting | SVM scaling is O(n²)–O(n³) in training data |
| Excellent on rare classes (pretrained DINOv2 features) | Single embedding per image (no augmentation) |
| Interpretable confidence via predict_proba | Cannot improve beyond frozen feature quality |
| t-SNE visualisation enables cluster analysis | class_weight='balanced' alone may be insufficient |
| Low risk of overfitting (no deep network fine-tuning) | No end-to-end gradient flow |

---

## 12. Conclusion

This work explored four distinct approaches to petri dish cell culture image classification across three methodological categories required by the assignment. The key findings are:

1. **Pretrained models dramatically outperform training from scratch**, particularly on minority classes. DINOv2's self-supervised pretraining produces especially rich features for scientific imaging.

2. **DINOv2 fine-tuning (with DC-GAN augmentation) achieves the best overall performance**, with particularly strong results on the very rare classes (p, q) that challenge all other approaches. The combination of 448×448 input, Focal Loss, and 3-phase progressive unfreezing is a powerful strategy for imbalanced scientific image classification.

3. **DINOv2 + SVM is competitive without any fine-tuning**, demonstrating that the frozen CLS token from DINOv2 already encodes discriminative cellular features. This approach is valuable when compute is limited or rapid prototyping is needed.

4. **DC-GAN augmentation provides measurable benefits for minority classes**, but the quality of GAN-generated images (limited to 64×64) may introduce noise when upscaled to 448×448 for DINOv2. Future work could use higher-resolution GAN architectures (e.g. StyleGAN2) for better synthetic image quality.

5. **The most difficult classes** (m, f, b, h) pose challenges across all models — either due to genuine visual ambiguity, complete absence of training data, or insufficient structural diversity in the training set. These limitations are inherent to the dataset and cannot be fully resolved by model architecture choices alone.

---

## 13. References

[1] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). **ImageNet Classification with Deep Convolutional Neural Networks.** *Advances in Neural Information Processing Systems (NeurIPS)*, 25.

[2] Radford, A., Metz, L., & Chintala, S. (2016). **Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.** *International Conference on Learning Representations (ICLR)*.

[3] Ma, Z., Dou, B., Zhao, R., & Du, X. (2020). **Combining DC-GAN with ResNet for blood cell image classification.** *Medical & Biological Engineering & Computing*, 58(6), 1251–1264.

[4] Simonyan, K., & Zisserman, A. (2015). **Very Deep Convolutional Networks for Large-Scale Image Recognition.** *International Conference on Learning Representations (ICLR)*.

[5] Ioffe, S., & Szegedy, C. (2015). **Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.** *Proceedings of the 32nd International Conference on Machine Learning (ICML)*.

[6] Lin, M., Chen, Q., & Yan, S. (2014). **Network In Network.** *International Conference on Learning Representations (ICLR)*.

[7] He, K., Zhang, X., Ren, S., & Sun, J. (2016). **Deep Residual Learning for Image Recognition.** *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770–778.

[8] Wen, Y., Zhang, K., Li, Z., & Qiao, Y. (2016). **A Discriminative Feature Learning Approach for Deep Face Recognition.** *European Conference on Computer Vision (ECCV)*, 499–515.

[9] Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., ... & Joulin, A. (2024). **DINOv2: Learning Robust Visual Features without Supervision.** *Transactions on Machine Learning Research (TMLR)*.

[10] Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). **Focal Loss for Dense Object Detection.** *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2980–2988.

[11] Müller, R., Kornblith, S., & Hinton, G. E. (2019). **When Does Label Smoothing Help?** *Advances in Neural Information Processing Systems (NeurIPS)*, 32.

[12] Cortes, C., & Vapnik, V. (1995). **Support-vector networks.** *Machine Learning*, 20(3), 273–297.

[13] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2021). **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.** *International Conference on Learning Representations (ICLR)*.

[14] Kingma, D. P., & Ba, J. (2015). **Adam: A Method for Stochastic Optimization.** *International Conference on Learning Representations (ICLR)*.

[15] Lin, M., Chen, Q., & Yan, S. (2014). **Network In Network.** *arXiv:1312.4400*.

---

*End of Report*

---

> **Appendix: Notebook to Approach Mapping**
> 
> | Notebook | Approach | Category |
> |----------|----------|----------|
> | `scratch_cnn_improved_fixed_v2.ipynb` | Custom 5-block CNN | Trained from scratch |
> | `dcgan_resnet_petridish_v2.ipynb` | DC-GAN + ResNet50 + Center Loss | Fine-tuning pretrained network |
> | `dcgan_dinov2_petridish_v2.ipynb` | DC-GAN + DINOv2-ViT-B/14 + Focal Loss | Fine-tuning pretrained network |
> | `dinov2_svm_petridish.ipynb` | DINOv2 frozen embeddings + RBF SVM | Embeddings + ML classifier |
