# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 13:56:03 2025

This module defines neural network architectures for various datasets.

Each dataset uses a different architecture suited to its characteristics:
- Simple feedforward for Iris
- LeNet-style CNN for MNIST/EMNIST/Fashion MNIST
- ResNet-based model for CIFAR-10
- CNN+Embedding for Reuters
- EfficientNet-based model for Tiny ImageNet
- CNN for SVHN

@author: byzkl
"""

import tensorflow as tf
from tensorflow.keras import Sequential, Model, layers
from tensorflow.keras.layers import (
    Dense, Conv2D, Conv1D, AveragePooling2D, MaxPooling2D,
    Flatten, Dropout, Embedding, GlobalAveragePooling1D,
    GlobalAveragePooling2D, ReLU
)
from tensorflow.keras.applications import ResNet50

def load_model(dataset):
    """
    Returns a compiled neural network model based on the specified dataset.

    Parameters
    ----------
    dataset : str
        Name of the dataset. Supported values:
        ["iris", "mnist", "fashion_mnist", "svhn", "cifar10", "reuters", "emnist", "tiny-imagenet"]

    Returns
    -------
    model : tf.keras.Model
        A TensorFlow Keras model ready to be compiled and trained.
    """

    if dataset == "iris":
        # Simple feedforward network for tabular data (Iris dataset)
        model = Sequential([
            Dense(64, input_dim=4),                   # Input: 4 features
            ReLU(name='feature_layer'),               # Nonlinear feature extraction
            Dense(3, activation='softmax')            # Output: 3 classes
        ])

    elif dataset == "mnist":
        # LeNet-style CNN for MNIST digit classification
        model = Sequential([
            Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
            AveragePooling2D(pool_size=(2, 2)),

            Conv2D(16, kernel_size=(5, 5), activation='relu'),
            AveragePooling2D(pool_size=(2, 2)),

            Flatten(),                                # Flatten for dense layers
            Dense(120, activation='relu'),
            Dense(84, activation='relu', name='feature_layer'),
            Dense(10, activation='softmax')           # 10 output classes
        ])

    elif dataset == "fashion_mnist":
        # CNN for Fashion MNIST (slightly deeper than MNIST version)
        model = Sequential([
            Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=(28, 28, 1)),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),

            Flatten(),
            Dense(128, activation='relu', name='feature_layer'),
            Dense(10, activation='softmax')           # 10 output classes
        ])

    elif dataset == "svhn":
        # LeNet-like architecture for Street View House Numbers dataset
        model = Sequential([
            Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(32, 32, 3)),
            AveragePooling2D(pool_size=(2, 2)),

            Conv2D(16, kernel_size=(5, 5), activation='relu'),
            AveragePooling2D(pool_size=(2, 2)),

            Flatten(),
            Dense(120, activation='relu'),
            Dense(84, activation='relu', name='feature_layer'),
            Dense(10, activation='softmax')           # 10 classes (digits 0–9)
        ])

    elif dataset == "cifar10":
        # Transfer learning with ResNet50 for CIFAR-10 classification
        base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(32, 32, 3),
            pooling='avg'
        )
        base_model.trainable = False                  # Freeze base model weights

        model = Sequential([
            base_model,
            Flatten(name='feature_layer'),
            Dense(10, activation='softmax')           # 10 CIFAR-10 classes
        ])

    elif dataset == "reuters":
        # Text classification using Embedding + Conv1D for Reuters dataset
        max_words = 10000                             # Vocabulary size
        maxlen = 100                                  # Max sequence length
        model = Sequential([
            Embedding(input_dim=max_words, output_dim=50, input_length=maxlen),
            Conv1D(64, 5, activation='relu'),
            GlobalAveragePooling1D(),
            Dense(128, activation='relu', name='feature_layer'),
            Dropout(0.5),
            Dense(46, activation='softmax')           # 46 Reuters topics
        ])

    elif dataset == "emnist":
        # LeNet for EMNIST letters (26 classes: a–z)
        model = Sequential([
            Conv2D(6, (5, 5), activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2)),

            Conv2D(16, (5, 5), activation='relu'),
            MaxPooling2D((2, 2)),

            Flatten(),
            Dense(120, activation='relu'),
            Dense(84, activation='relu', name='feature_layer'),
            Dense(26, activation='softmax')
        ])

    elif dataset == "tiny-imagenet":
        # Transfer learning using EfficientNetV2B0 for Tiny ImageNet
        vit_b16 = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )

        # Freeze most layers, fine-tune the last 10
        for layer in vit_b16.layers[:-10]:
            layer.trainable = False

        x = vit_b16.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        predictions = layers.Dense(200, activation='softmax')(x)  # 200 Tiny-ImageNet classes

        model = Model(inputs=vit_b16.input, outputs=predictions)

    return model
