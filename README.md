# DCGAN for MNIST Digit Generation

A PyTorch implementation of Deep Convolutional Generative Adversarial Networks (DCGAN) for generating handwritten digits using the MNIST dataset.

## 📋 Overview

This project implements a DCGAN architecture that learns to generate realistic handwritten digits. The model consists of two neural networks - a Generator and a Discriminator - that compete against each other in a game-theoretic framework to produce high-quality synthetic images.

## 🎯 Key Features

- **DCGAN Architecture**: Implements the standard DCGAN paper architecture with convolutional layers
- **Label Smoothing**: Uses one-sided label smoothing for training stability
- **Epoch-wise Visualization**: Displays real vs fake image comparisons after each epoch
- **Training Metrics Tracking**: Monitors Generator/Discriminator losses and discriminator outputs
- **Training Health Indicators**: Automatic feedback on training dynamics

## 🏗️ Architecture

### Generator Network
```
Input: 100-dim latent vector (noise)
↓
ConvTranspose2d (100 → 512) → BatchNorm → ReLU
ConvTranspose2d (512 → 256) → BatchNorm → ReLU
ConvTranspose2d (256 → 128) → BatchNorm → ReLU
ConvTranspose2d (128 → 64) → BatchNorm → ReLU
ConvTranspose2d (64 → 1) → Tanh
↓
Output: 64x64x1 grayscale image
```

### Discriminator Network
```
Input: 64x64x1 grayscale image
↓
Conv2d (1 → 64) → LeakyReLU
Conv2d (64 → 128) → BatchNorm → LeakyReLU
Conv2d (128 → 256) → BatchNorm → LeakyReLU
Conv2d (256 → 512) → BatchNorm → LeakyReLU
Conv2d (512 → 1) → Sigmoid
↓
Output: Probability (real vs fake)
```

## ⚙️ Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 128 | Training batch size |
| `image_size` | 64 | Resized image dimensions |
| `nz` | 100 | Latent vector dimension |
| `ngf` | 64 | Generator feature map size |
| `ndf` | 64 | Discriminator feature map size |
| `num_epochs` | 30 | Number of training epochs |
| `lr_G` | 0.0002 | Generator learning rate |
| `lr_D` | 0.0001 | Discriminator learning rate |
| `beta1` | 0.5 | Adam optimizer beta1 |
| `real_label_smooth` | 0.85 | Real label smoothing value |
| `fake_label_smooth` | 0.2 | Fake label smoothing value |

## 🔧 Key Implementation Details

### 1. **Label Smoothing**
Instead of using hard labels (1 for real, 0 for fake), the implementation uses:
- Real images: labeled as 0.85
- Fake images: labeled as 0.2

This prevents the discriminator from becoming overconfident and stabilizes training.

### 2. **Weight Initialization**
All convolutional and batch normalization layers are initialized following DCGAN paper recommendations:
- Conv weights: Normal(0, 0.02)
- BatchNorm weights: Normal(1.0, 0.02)
- BatchNorm bias: Constant(0)

### 3. **Training Strategy**
Each iteration performs:
1. **Discriminator Update**: Train on real images, then fake images
2. **Generator Update**: Train to fool the discriminator with fake images

### 4. **Monitoring Metrics**
The code tracks:
- `D(x)`: Discriminator's output on real images (should be ~0.85)
- `D(G(z))`: Discriminator's output on fake images (should be ~0.2)
- Generator and Discriminator losses
- Per-epoch averaged metrics

## 📊 Training Visualization

After each epoch, the code displays:
- **Real vs Fake Comparison**: Side-by-side grid showing 32 real images vs 32 generated images
- **Training Statistics**: Average losses and discriminator outputs
- **Health Indicators**: Color-coded feedback on training dynamics
  - 🟢 Green: Training is healthy
  - 🟡 Yellow: Training is reasonable
  - 🔴 Red: Potential training issues detected

## 🚀 Usage

### Requirements
```python
torch
torchvision
numpy
matplotlib
```

### Running the Code
```python
# The code automatically:
# 1. Downloads MNIST dataset
# 2. Initializes networks
# 3. Trains for specified epochs
# 4. Displays visualizations
# 5. Saves generated images

# Simply run all cells in sequence
```

### Dataset
- **Source**: MNIST handwritten digits
- **Size**: 60,000 training images
- **Original Format**: 28x28 grayscale
- **Processed Format**: 64x64 grayscale (resized and normalized to [-1, 1])

## 📈 Expected Results

### Training Progress
- **Early Epochs (1-5)**: Generator produces noisy, unclear digits
- **Mid Epochs (6-15)**: Recognizable digit shapes emerge
- **Later Epochs (16-30)**: Sharp, realistic-looking digits

### Healthy Training Signs
- D(x) stays around 0.7-0.9
- D(G(z)) gradually decreases but stays above 0.1
- Generator loss remains relatively stable
- Discriminator loss doesn't approach zero

## 🔍 Understanding the Output

### Loss Values
- **Loss_D**: Discriminator's total loss (lower means it's classifying well)
- **Loss_G**: Generator's loss (lower means it's fooling the discriminator)

### Discriminator Outputs
- **D(x)**: Average prediction on real images (higher = correctly identifying reals)
- **D(G(z))**: Average prediction on fake images (lower = correctly identifying fakes)

## 🛠️ Troubleshooting

### Common Issues

**Mode Collapse**: Generator produces limited variety
- Solution: Adjust label smoothing values, reduce learning rates

**Discriminator Too Strong**: D(G(z)) stays near 0, Loss_G explodes
- Solution: Lower discriminator learning rate, increase generator updates

**Discriminator Too Weak**: D(x) drops below 0.5
- Solution: Increase discriminator learning rate, check architecture

## 📚 References

- [DCGAN Paper](https://arxiv.org/abs/1511.06434) - Radford et al., 2015
- [PyTorch DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

## 💡 Future Improvements

- [ ] Implement Wasserstein GAN loss for better stability
- [ ] Add FID (Fréchet Inception Distance) metric for quality evaluation
- [ ] Conditional GAN for controlled digit generation
- [ ] Progressive growing for higher resolution outputs
- [ ] Integration with wandb for experiment tracking

## 📝 Notes

- Training on GPU is highly recommended (30 epochs ~15-30 minutes on GPU vs hours on CPU)
- The model uses fixed random seeds for reproducibility
- Generated images are saved at regular intervals for creating training animations
- Compatible with Kaggle notebook environment

---
