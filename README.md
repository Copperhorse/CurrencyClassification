# Currency Classification

A deep learning project for classifying Pakistani currency denominations using image recognition. This project uses an EfficientNet B0 model to identify different notes and their denominations.

## Overview

This project implements an image classification system trained to recognize and classify Pakistani currency notes into 15 different categories:
- **10 Rs** (front and back)
- **20 Rs** (front and back)
- **50 Rs** (front and back)
- **100 Rs** (front and back)
- **500 Rs** (front and back)
- **1000 Rs** (front and back)
- **5000 Rs** (front and back)
- **Others** (unrecognized or foreign currency)

## Model Architecture

The project uses **EfficientNet B0**, a lightweight yet powerful convolutional neural network that provides an optimal balance between model efficiency and accuracy. The model is pre-configured with:
- Input size: 224×224 pixels
- Output classes: 15 currency denominations
- Framework: PyTorch with timm library

## Project Structure

```
CurrencyClassification/
├── README.md
└── Testing_currencyClassification.ipynb
```

### Files

- **Testing_currencyClassification.ipynb** - Main Jupyter Notebook containing:
  - Model loading and initialization
  - Custom image preprocessing pipeline
  - Image classification inference
  - Prediction visualization
  - Currency value extraction and summation

## Dependencies

The project requires the following Python libraries:

- **PyTorch** - Deep learning framework
- **TorchVision** - Computer vision utilities
- **timm** - PyTorch Image Models
- **NumPy** - Numerical computing
- **Pillow (PIL)** - Image processing
- **OpenCV (cv2)** - Image processing and enhancement
- **Matplotlib** - Visualization
- **Seaborn** - Statistical visualization

### Installation

```bash
pip install torch torchvision
pip install timm
pip install numpy pillow opencv-python matplotlib seaborn
```

## Usage

### Running the Notebook

1. Open `Testing_currencyClassification.ipynb` in Google Colab or Jupyter Notebook
2. Install required dependencies (automated in the notebook)
3. Load the pre-trained model: `currency_classifier_model.pth`
4. Prepare your dataset with training/validation splits
5. Run cells sequentially to process images and make predictions

### Image Preprocessing

The project includes a custom `CustomPreprocessing` class that applies:
- **Histogram Equalization** (optional) - Improves contrast
- **Denoising** - Reduces noise using median blur
- **Edge Detection** (optional) - Applies Canny edge detection

All images are normalized to:
- Size: 224×224 pixels
- Mean: [0.5, 0.5, 0.5]
- Standard Deviation: [0.5, 0.5, 0.5]

### Making Predictions

```python
# Load and transform an image
image = Image.open("path/to/currency_image.jpg").convert('RGB')
image = transform(image)  # Apply preprocessing
image = image.unsqueeze(0)  # Add batch dimension

# Make prediction
model.eval()
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)
    predicted_class = class_names[predicted.item()]
```

### Extracting Currency Value

The notebook includes functionality to:
- Predict the currency denomination from an image
- Extract the numeric value using regex
- Sum multiple predictions to get total currency value

```python
sum_list = []  # Track predicted values
# After predictions are made:
sum_of_currency = sum(sum_list)
```

## Dataset Structure

Expected directory structure for training data:

```
Pakistan/
├── Training/
│   ├── 10Rs/
│   ├── 10Rsback/
│   ├── 20Rs/
│   ├── 20Rsback/
│   ├── 50Rs/
│   ├── 50Rsback/
│   ├── 100Rs/
│   ├── 100Rsback/
│   ├── 500Rs/
│   ├── 500Rsback/
│   ├── 1000Rs/
│   ├── 1000Rsback/
│   ├── 5000Rs/
│   ├── 5000Rsback/
│   └── others/
└── Valid/
    └── (same subdirectories)
```

## Results

The model achieves classification across 15 different currency categories (both front and back of each denomination), with the ability to:
- Identify specific currency denominations
- Distinguish between front and back of notes
- Handle various lighting and image quality conditions

## Hardware Requirements

- **GPU**: CUDA-compatible GPU recommended (CUDA 12.4+)
- **CPU**: CPU-only mode supported but slower
- **Memory**: Minimum 4GB RAM, 8GB+ recommended

## Future Improvements

- [ ] Implement data augmentation for better generalization
- [ ] Add confidence scores to predictions
- [ ] Support for other currencies
- [ ] Real-time video stream processing
- [ ] Mobile deployment optimization
- [ ] Improved handling of partially visible or damaged notes

## License

This project is open source and available for educational and research purposes.

## Author

Copperhorse

## Acknowledgments

- EfficientNet architecture by Google Brain
- PyTorch and timm communities
- Dataset contributors

---

For questions or contributions, please open an issue or pull request on the repository.
