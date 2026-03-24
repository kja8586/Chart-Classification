"""
Data loading utilities using Keras ImageDataGenerator.
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import CLASS_LABELS, IMAGE_SIZE, BATCH_SIZE


def get_dataloaders(train_path, test_path, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE):
    """
    Create Keras data generators from directory structure.

    Args:
        train_path: Path to training images (organized in class subfolders).
        test_path: Path to test images (organized in class subfolders).
        image_size: Tuple (H, W) to resize images to.
        batch_size: Batch size.

    Returns:
        train_generator, test_generator
    """
    datagen = ImageDataGenerator()

    train_generator = datagen.flow_from_directory(
        train_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASS_LABELS,
        shuffle=True,
    )

    test_generator = datagen.flow_from_directory(
        test_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASS_LABELS,
        shuffle=False,
    )

    return train_generator, test_generator
