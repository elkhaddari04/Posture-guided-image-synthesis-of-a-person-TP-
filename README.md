# Posture-guided image synthesis of a person (TP)

## Project Description

This project implements a pose transfer system based on the "Everybody Dance Now" paper (Chan et al., ICCV 2019). The system transfers dance movements from a source video to a target person by learning to generate realistic images of the target person performing the source video's movements. This implementation provides three different approaches of increasing complexity, from simple nearest neighbor matching to advanced GAN-based generation.

## Objectives

- Implementation of three distinct approaches for pose transfer:
    1. Baseline nearest neighbor search for direct frame matching
    2. Direct neural network generation (VanillaNN) with two variants
    3. GAN-based generation with adversarial training
- Comparative analysis of different methods for pose-guided person image synthesis
- Progressive understanding from basic to advanced approaches
- Practical implementation of concepts from the original paper

## Technical Requirements

### Software Dependencies
- Python 3.8+
- PyTorch 1.9+
- OpenCV (cv2) 4.5+
- MediaPipe 0.8+
- NumPy 1.19+
- PIL (Python Imaging Library)
- scikit-learn (for nearest neighbors)
- CUDA Toolkit 11.8+ (for GPU acceleration)

### Hardware 
- Tested Configuration:
  - GPU: NVIDIA GTX 1050 (4GB VRAM)
  - CUDA Toolkit 11.8
  - 16GB System RAM

## Implementation Details

### 1. Data Preprocessing Pipeline

```bash
python VideoSkeleton.py data/taichi1.mp4 [force_compute] [frame_mod]
```

Parameters:
- `force_compute`: Boolean to force recomputation of skeletons
- `frame_mod`: Integer for frame sampling rate (default: 10)

Pipeline Steps:
1. Video frame extraction
2. MediaPipe pose detection
3. Skeleton normalization
4. Data caching for training
5. Optional frame cropping based on skeleton bounds

### 2. Nearest Neighbor Approach (GenNearest)

A baseline implementation that:
- Maintains a database of skeleton-image pairs
- Uses Euclidean distance for skeleton matching
- Provides real-time frame retrieval
- Serves as a performance baseline

Limitations:
- Limited generalization capability
- Memory-intensive for large datasets
- Can produce jerky animations

### 3. VanillaNN Approaches

#### Direct Skeleton to Image (Variant 1)
Architecture:
```
Input (26) → ConvTranspose2d → BatchNorm → ReLU → ... → Output (3×64×64)
```

Key Features:
- Input: 26D vector (13 2D joint coordinates)
- Progressive upsampling using ConvTranspose2d
- Batch normalization for training stability
- ReLU activations
- Output: 64×64 RGB image

#### Skeleton Drawing to Image (Variant 2)
Architecture:
```
Input (3×64×64) → Conv2d → BatchNorm → ReLU → ... → Output (3×64×64)
```

Improvements:
- Structured input representation
- Better spatial information preservation
- More stable training behavior

Training Parameters:
- Learning Rate: 0.0002
- Batch Size: 32
- Optimizer: Adam (β1=0.5, β2=0.999)
- Loss Function: L1
- Training Episodes: 200

Results:
- Good convergence in terms of L1 loss
- Limited realism in generated videos
- Pre-trained models available at: `data/Dance/DanceVanillaFromSke.pth` and `DanceVanillaFromSkeim.pth`
  
![alt text](https://github.com/elkhaddari04/Posture-guided-image-synthesis-of-a-person-TP-/blob/main/Image%202024-11-16%2021-26-38.gif)

### 4. GAN Implementation

#### Generator Architecture ( Direct Skeleton to Image )
- Based on DCGAN with modifications
- Progressive feature map expansion
- Skip connections for detail preservation
- Batch normalization and ReLU activations

#### Discriminator Design
Architecture:
```python
Conv2d(3, 64, 4, 2, 1) → LeakyReLU →
Conv2d(64, 128, 4, 2, 1) → BatchNorm → LeakyReLU →
Conv2d(128, 256, 4, 2, 1) → BatchNorm → LeakyReLU →
Conv2d(256, 1, 4, 2, 1) → Sigmoid
```

Stability Features:
- Dropout layers (0.25 rate)
- Soft label smoothing
- Dynamic L1 weight scheduling
- Learning rate scheduling
  
![alt text](https://github.com/elkhaddari04/Posture-guided-image-synthesis-of-a-person-TP-/blob/main/gan_training)

Training Configuration:
- Generator LR: 0.0002
- Discriminator LR: 0.0001
- Batch Size: 16
- Training Episodes: 200
- L1 Weight: Dynamic scaling (100 → 30)
- Label Smoothing: Real [0.85-1.0], Fake [0.0-0.15]
  
![alt text](https://github.com/elkhaddari04/Posture-guided-image-synthesis-of-a-person-TP-/blob/main/Image%202024-11-16%2002-36-13.gif)

## Usage Instructions

### Model Training

```bash
# Basic nearest neighbor
python GenNearest.py data/taichi1.mp4

# VanillaNN (Direct Skeleton)
python GenVanillaNN.py data/taichi1.mp4 false 1

# VanillaNN (Skeleton Drawing)
python GenVanillaNN.py data/taichi1.mp4 false 2

# GAN
python GenGAN.py data/taichi1.mp4
```

### Running Inference

```python
from DanceDemo import DanceDemo

# Choose model type (1=Nearest, 2=VanillaNN-Skeleton, 3=VanillaNN-Drawing, 4=GAN)
demo = DanceDemo("source_video.mp4", model_type=4)
demo.draw()
```


## Current Problems
- Videos can look choppy sometimes
- Generated images are often blurry, especially faces
- Training takes a long time .

## What We Can Improve
- Make movements smoother and more natural looking in the generated videos
- Improve image quality, especially faces and complex poses
- Speed up training time using better hardware and optimized code



## References

1. Chan, C., Ginosar, S., Zhou, T., & Efros, A. A. (2019). Everybody dance now. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 5933-5942).

2. Radford, A., Metz, L., & Chintala, S. (2015). Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434.

3. MediaPipe Pose Detection

## Acknowledgments
http://alexandre.meyer.pages.univ-lyon1.fr/m2-apprentissage-profond-image/am/tp_dance/
Base implementation derived from **Vision, image and machine learning** course materials, with significant modifications and enhancements for the GAN implementation and training pipeline.
