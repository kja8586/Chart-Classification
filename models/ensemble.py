"""
Ensemble model combining multiple frozen CNN backbones (TensorFlow/Keras).

Concatenates GlobalAveragePooling2D outputs from multiple backbones
and feeds them through a shared Dense classification head.
"""

from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Concatenate, Input

from config import NUM_CLASSES, IMAGE_SIZE
from models.cnn_models import get_base_model


def build_ensemble(model_names, num_classes=NUM_CLASSES, input_shape=None):
    """
    Build an ensemble model from multiple pretrained backbones.

    Args:
        model_names: List of backbone names (e.g., ['resnet152', 'efficientnetb0']).
        num_classes: Number of output classes.
        input_shape: (H, W, C) input shape.

    Returns:
        Keras Model with all backbones frozen and a shared classification head.
    """
    if input_shape is None:
        input_shape = IMAGE_SIZE + (3,)

    shared_input = Input(shape=input_shape)
    pooled_outputs = []

    for name in model_names:
        base = get_base_model(name, input_shape)

        # Freeze backbone
        for layer in base.layers:
            layer.trainable = False

        features = base(shared_input)
        pooled = GlobalAveragePooling2D()(features)
        pooled_outputs.append(pooled)

    # Concatenate all backbone features
    if len(pooled_outputs) > 1:
        merged = Concatenate()(pooled_outputs)
    else:
        merged = pooled_outputs[0]

    output = Dense(num_classes, activation='softmax')(merged)

    ensemble_name = "_".join(model_names)
    model = Model(inputs=shared_input, outputs=output, name=f"ensemble_{ensemble_name}")
    return model
