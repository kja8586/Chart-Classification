"""
CNN transfer-learning models for chart classification (TensorFlow/Keras).

Supports 9 pretrained ImageNet backbones. All use frozen feature extractors
with GlobalAveragePooling2D + Dense classification head.
"""

from keras.applications import (
    VGG16, VGG19, MobileNet, Xception, ResNet50,
    ResNet152, InceptionV3, DenseNet121, EfficientNetB0,
)
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense

from config import NUM_CLASSES, IMAGE_SIZE


# Registry of supported model names -> (Keras application class, display name)
AVAILABLE_MODELS = {
    'vgg16':          (VGG16,          'VGG-16'),
    'vgg19':          (VGG19,          'VGG-19'),
    'mobilenet':      (MobileNet,      'MobileNet'),
    'xception':       (Xception,       'Xception'),
    'resnet50':       (ResNet50,       'ResNet-50'),
    'resnet152':      (ResNet152,      'ResNet-152'),
    'inceptionv3':    (InceptionV3,    'Inception-V3'),
    'densenet121':    (DenseNet121,    'DenseNet-121'),
    'efficientnetb0': (EfficientNetB0, 'EfficientNet-B0'),
}


def get_base_model(name, input_shape=None):
    """
    Load a pretrained ImageNet backbone.

    Args:
        name: One of the keys in AVAILABLE_MODELS.
        input_shape: (H, W, C) input shape. Defaults to IMAGE_SIZE + (3,).

    Returns:
        Keras base model with include_top=False.
    """
    if input_shape is None:
        input_shape = IMAGE_SIZE + (3,)

    if name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: {list(AVAILABLE_MODELS.keys())}"
        )

    model_class, _ = AVAILABLE_MODELS[name]
    return model_class(weights='imagenet', include_top=False, input_shape=input_shape)


def build_model(base_model, num_classes=NUM_CLASSES):
    """
    Build classification model: freeze backbone → GAP → Dense(softmax).

    Args:
        base_model: Pretrained Keras base model.
        num_classes: Number of output classes.

    Returns:
        Compiled-ready Keras Model.
    """
    # Freeze all backbone layers
    for layer in base_model.layers:
        layer.trainable = False

    gap = GlobalAveragePooling2D()(base_model.output)
    output = Dense(num_classes, activation='softmax')(gap)
    model = Model(inputs=base_model.input, outputs=output)
    return model
