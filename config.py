"""
Configuration and hyperparameters for Chart Classification.
"""

# Dataset class labels (15 chart types from ICPR 2022 CHART-Infographics)
CLASS_LABELS = [
    'area', 'heatmap', 'horizontal_bar', 'horizontal_interval', 'line',
    'manhattan', 'map', 'pie', 'scatter', 'scatter-line', 'surface',
    'venn', 'vertical_bar', 'vertical_box', 'vertical_interval'
]
NUM_CLASSES = len(CLASS_LABELS)

# Image settings
IMAGE_SIZE = (480, 670)

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 500
EARLY_STOP_PATIENCE = 20

# Default data paths (override via CLI args)
DEFAULT_TRAIN_PATH = "./data/train"
DEFAULT_TEST_PATH = "./data/test"

# Checkpoint directory
CHECKPOINT_DIR = "./checkpoints"
