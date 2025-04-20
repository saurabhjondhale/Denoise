# Denoise
Noise removal from a RGB image.


# 🔍 CleanVision: Open-Source Image Denoising

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org)

Remove noise from images using deep learning! Perfect for:
- 📸 Restoring old photos  
- 🏥 Medical imaging (MRI/CT scans)  
- 🔭 Astronomy (telescope images)  

## Quick Start  
```python
pip install cleanvision
from cleanvision import DnCNN

model = DnCNN(pretrained=True)
denoised_image = model.predict("noisy_image.jpg")
