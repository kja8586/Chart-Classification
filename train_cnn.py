"""
Train a single CNN model for chart classification.

Usage:
    python train_cnn.py --model vgg19 --train_path ./data/train --test_path ./data/test
    python train_cnn.py --model resnet152 --epochs 100
    python train_cnn.py --help
"""

import argparse
import os

import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint

from config import (
    CLASS_LABELS, NUM_CLASSES, IMAGE_SIZE, BATCH_SIZE,
    EPOCHS, EARLY_STOP_PATIENCE,
    DEFAULT_TRAIN_PATH, DEFAULT_TEST_PATH, CHECKPOINT_DIR,
)
from data_loader import get_dataloaders
from models.cnn_models import get_base_model, build_model, AVAILABLE_MODELS
from utils import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Train a CNN model for chart classification.")
    parser.add_argument("--model", type=str, required=True, choices=list(AVAILABLE_MODELS.keys()),
                        help=f"Model name. Options: {list(AVAILABLE_MODELS.keys())}")
    parser.add_argument("--train_path", type=str, default=DEFAULT_TRAIN_PATH, help="Path to training data.")
    parser.add_argument("--test_path", type=str, default=DEFAULT_TEST_PATH, help="Path to test data.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Max training epochs.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE, help="Early stopping patience.")
    parser.add_argument("--checkpoint_dir", type=str, default=CHECKPOINT_DIR, help="Directory to save weights.")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    display_name = AVAILABLE_MODELS[args.model][1]
    print(f"\nTraining {display_name} ({args.model})")

    # Data
    train_gen, test_gen = get_dataloaders(
        args.train_path, args.test_path, IMAGE_SIZE, args.batch_size
    )

    # Build model
    base = get_base_model(args.model)
    model = build_model(base, NUM_CLASSES)

    # Compile
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ],
    )

    # Callbacks
    weight_path = os.path.join(args.checkpoint_dir, f"{args.model}.weights.h5")
    callbacks = [
        EarlyStopping(
            monitor='loss', patience=args.patience, mode='min',
            verbose=1, restore_best_weights=True,
        ),
        ModelCheckpoint(
            filepath=weight_path, save_weights_only=True,
            monitor='accuracy', mode='max', save_best_only=True,
        ),
    ]

    # Train
    model.fit(train_gen, epochs=args.epochs, callbacks=callbacks)

    # Evaluate
    print(f"\n{'='*50}")
    print(f"Test Results for {display_name}")
    evaluate_model(model, test_gen, CLASS_LABELS, save_dir=args.checkpoint_dir)


if __name__ == "__main__":
    main()
