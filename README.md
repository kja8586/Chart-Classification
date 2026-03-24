# Chart Classification Using Deep Learning

A comparative study of transfer learning approaches for automatic chart type classification on the **ICPR 2022 CHART-Infographics** dataset with **15 chart categories**.

## Results

### Individual CNN Models (Transfer Learning)

| Model | Accuracy | Loss | Precision | Recall |
|-------|----------|------|-----------|--------|
| VGG-16 | 0.8049 | 2.3079 | 0.8113 | 0.7997 |
| VGG-19 | 0.8239 | 1.5574 | 0.8306 | 0.8211 |
| MobileNet | 0.7768 | 1.5793 | 0.7865 | 0.7721 |
| Xception | 0.5949 | 9.0046 | 0.5960 | 0.5921 |
| ResNet-50 | 0.8285 | 2.6278 | 0.8298 | 0.8276 |
| ResNet-152 | 0.8450 | 1.4084 | 0.8466 | 0.8445 |
| Inception-V3 | 0.6481 | 6.2019 | 0.6512 | 0.6461 |
| DenseNet-121 | 0.6758 | 2.8417 | 0.7002 | 0.6585 |
| **EfficientNet-B0** | **0.8624** | **0.6356** | **0.8694** | **0.8561** |

### Ensemble Models

| Ensemble | Accuracy | Loss | Precision | Recall |
|----------|----------|------|-----------|--------|
| ResNet-152 + EfficientNet-B0 | **0.8765** | **0.5555** | **0.8871** | **0.8657** |
| VGG-19 + MobileNet + Xception | 0.8685 | 0.9814 | 0.8723 | 0.8664 |
| ResNet-152 + EfficientNet-B0 + VGG-19 | 0.8718 | 0.5674 | 0.8840 | 0.8620 |

> **Best CNN result: ResNet-152 + EfficientNet-B0 ensemble achieves 87.65% accuracy.**

## Chart Categories (15 classes)

`area` · `heatmap` · `horizontal_bar` · `horizontal_interval` · `line` · `manhattan` · `map` · `pie` · `scatter` · `scatter-line` · `surface` · `venn` · `vertical_bar` · `vertical_box` · `vertical_interval`

## Project Structure

```
Chart-Classification/
├── config.py               # Hyperparameters & constants
├── data_loader.py           # Keras ImageDataGenerator utilities
├── models/
│   ├── cnn_models.py        # 9 CNN backbones (Keras Applications)
│   └── ensemble.py          # Multi-backbone ensemble model
├── train_cnn.py             # Train any single CNN model
├── train_ensemble.py        # Train ensemble models
├── utils.py                 # Evaluation & visualization helpers
├── requirements.txt
└── README.md
```

## Setup

```bash
# Clone the repo
git clone https://github.com/<your-username>/Chart-Classification.git
cd Chart-Classification

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Train a single CNN model

```bash
python train_cnn.py --model resnet152 --train_path /path/to/train --test_path /path/to/test
```

Available models: `vgg16`, `vgg19`, `mobilenet`, `xception`, `resnet50`, `resnet152`, `inceptionv3`, `densenet121`, `efficientnetb0`

### Train an ensemble model

```bash
python train_ensemble.py --models resnet152 efficientnetb0 --train_path /path/to/train --test_path /path/to/test
```

### CLI help

```bash
python train_cnn.py --help
python train_ensemble.py --help
```

## Dataset

This project uses the **ICPR 2022 CHART-Infographics UB PMC** dataset. Images should be organized in class subfolders:

```
data/
├── train/
│   ├── area/
│   ├── heatmap/
│   └── .../
└── test/
    ├── area/
    ├── heatmap/
    └── .../
```

## Requirements

- Python 3.8+
- TensorFlow 2.10+
- scikit-learn
- matplotlib, seaborn
