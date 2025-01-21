# 🎨 Deep Convolutional GAN (DCGAN) from Scratch

This repository contains the implementation of a **Deep Convolutional Generative Adversarial Network (DCGAN)** built from scratch using PyTorch. DCGANs use convolutional layers for generating high-quality synthetic images.

---

## 🧠 Model Overview

DCGAN consists of two key components:

1. **Generator**: Transforms random noise into realistic images through a series of transposed convolutional layers.
2. **Discriminator**: Distinguishes between real and fake images using convolutional layers.

---

## ✨ Features

- PyTorch implementation of DCGAN.
- Customizable hyperparameters (e.g., image channels, latent vector size).
- Trains on image datasets to generate 64x64 images.
- Demonstrates the use of optimizers, loss functions, and CUDA for GPU acceleration.

---

## 🛠 Implementation

### Generator Architecture

The generator creates images by progressively upsampling random noise through transposed convolutional layers.  
Key hyperparameters:
- `nc`: Number of channels in the output image (e.g., 3 for RGB).
- `nz`: Size of the latent vector (random noise input).
- `ngf`: Number of feature maps in the generator.

```python
import torch.nn as nn

class DC_Gen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.model(input)

```

### Discriminator Architecture

The discriminator is a binary classifier that distinguishes between real and fake images. It processes input images through convolutional layers, using LeakyReLU activation for non-linearity and Batch Normalization for stability.

Key hyperparameters:
- `nc`: Number of channels in the input images (e.g., 3 for RGB).
- `ndf`: Number of feature maps in the first convolutional layer.

```python
import torch.nn as nn

class DC_Dis(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # Input: (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # State: (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # State: (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # State: (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # State: (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)
```
