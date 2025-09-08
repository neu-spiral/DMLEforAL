# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 13:57:38 2025

@author: byzkl
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, reuters
from sklearn.datasets import load_iris
from tensorflow.keras.preprocessing.sequence import pad_sequences
import scipy.io
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow_datasets as tfds

def convert_to_tensor(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.squeeze(image, axis=-1)
    return image, label

def flatten_dataset(dataset):
    """
    Flatten a tf.data.Dataset into NumPy arrays.

    Args:
        dataset (tf.data.Dataset): Dataset with (image, label) pairs.

    Returns:
        images: NumPy array of images.
        labels: NumPy array of labels.
    """
    images = []
    labels = []
    for img, lbl in dataset:
        images.append(img.numpy())
        labels.append(lbl.numpy())
    return np.concatenate(images), np.concatenate(labels)

def load_data(dataset, seed):
    if dataset=="iris":
        # Load Iris dataset
        iris = load_iris()
        X = iris.data
        y = iris.target
        X_val = []
        y_val = []
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 
    elif dataset=="mnist":
        # Load MNIST dataset
        (X, y), (X_test, y_test) = mnist.load_data()
        # X, _, y, _ = train_test_split(X, y, test_size=0.98, random_state=28371, stratify=y)
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
        height = 28
        width = 28  
        X_train = X_train.reshape((X_train.shape[0], height, width, 1))
        # Normalize the pixel values to the range [0, 1]
        X_train = X_train.astype('float32') / 255.0
        
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.96, random_state=42, stratify=y_test)
        X_val = X_val.reshape((X_val.shape[0], height, width, 1))
        # Normalize the pixel values to the range [0, 1]
        X_val = X_val.astype('float32') / 255.0
        
        X_test, _, y_test, _ = train_test_split(X_test, y_test, test_size=0.96, random_state=42, stratify=y_test)
        X_val = X_val.reshape((X_val.shape[0], height, width, 1))
        # Normalize the pixel values to the range [0, 1]
        X_val = X_val.astype('float32') / 255.0
        print(y_train.shape)
        print(y_test.shape)
        print(y_val.shape)
    elif dataset=="fashion_mnist":
        # Load Fashion MNIST dataset
        (X, y), (X_test, y_test) = fashion_mnist.load_data()
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
        height = 28
        width = 28  
        X_train = X_train.reshape((X_train.shape[0], height, width, 1))
        X_train = X_train.astype('float32') / 255.0
        
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.96, random_state=42, stratify=y_test)
        X_val = X_val.reshape((X_val.shape[0], height, width, 1))
        # Normalize the pixel values to the range [0, 1]
        X_val = X_val.astype('float32') / 255.0
        
        X_test, _, y_test, _ = train_test_split(X_test, y_test, test_size=0.96, random_state=42, stratify=y_test)
        X_test = X_test.reshape((X_test.shape[0], height, width, 1))
        # Normalize the pixel values to the range [0, 1]
        X_test = X_test.astype('float32') / 255.0
    elif dataset=="svhn":
        # Load SVHN dataset in .mat format
        svhn_train_path = "train_32x32.mat"  # Adjust the path to the location of your dataset file
        svhn_train_data = scipy.io.loadmat(svhn_train_path)
        
        # Extract features and labels
        X = svhn_train_data['X']
        X = np.transpose(X, (3, 0, 1, 2))
        y = svhn_train_data['y'].flatten()-1  # SVHN labels are 1-indexed, so subtract 1
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.8, random_state=42, stratify=y)
        X_train = X_train.astype('float32') / 255.0
        
        svhn_test_path = "test_32x32.mat"  # Adjust the path to the location of your dataset file
        svhn_test_data = scipy.io.loadmat(svhn_test_path)
        
        # Extract features and labels
        X = svhn_test_data['X']
        X = np.transpose(X, (3, 0, 1, 2))
        y = svhn_test_data['y'].flatten()-1  # SVHN labels are 1-indexed, so subtract 1
        X_val, X_test, y_val, y_test = train_test_split(X, y, test_size=0.98, random_state=42, stratify=y)
        X_val = X_val.astype('float32') / 255.0
        
        X_test, _, y_test, _ = train_test_split(X_test, y_test, test_size=0.98, random_state=42, stratify=y_test)
        X_test = X_test.astype('float32') / 255.0
    elif dataset=="reuters":
        # Load the Reuters Newswire dataset
        max_words = 10000  # Only consider the top 10,000 words in the dataset
        maxlen = 100  # Limit each document to the first 100 words
        (X, y), (X_test, y_test) = reuters.load_data(num_words=max_words)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
        # Pad sequences to ensure consistent length
        X_train = pad_sequences(X_train, maxlen=maxlen)
        
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)#, stratify=y_test)
        X_val = pad_sequences(X_val, maxlen=maxlen)
        X_test = pad_sequences(X_test, maxlen=maxlen)
    elif dataset=="emnist":
        # # Download and preprocess the data once
        # (emnist_train, emnist_test), emnist_info = tfds.load(
        #     'emnist/letters',
        #     split=['train', 'test'],
        #     shuffle_files=True,
        #     as_supervised=True,
        #     with_info=True
        # )
        # # Map the conversion function to the datasets
        # emnist_train = emnist_train.map(convert_to_tensor)
        # emnist_test = emnist_test.map(convert_to_tensor)

        # # Reduce the datasets to a single tensor
        # image_shape = emnist_info.features['image'].shape
        # emnist_train_images, emnist_train_labels = flatten_dataset(emnist_train)
        # emnist_train_images = tf.reshape(emnist_train_images, (-1, *image_shape))
        # emnist_train_images = tf.squeeze(emnist_train_images)
        # emnist_train_images = emnist_train_images.numpy()
        # emnist_train_labels = emnist_train_labels.numpy()
        # emnist_train_labels = emnist_train_labels - 1
        # X_train, X_test, y_train, y_test = train_test_split(emnist_train_images, emnist_train_labels, test_size=0.95, random_state=42, stratify=emnist_train_labels)
        # X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.996, random_state=42, stratify=y_test)
        # X_test, _, y_test, _ = train_test_split(X_test, y_test, test_size=0.996, random_state=42, stratify=y_test)
        # np.save('emnist_X_train.npy', X_train)
        # np.save('emnist_y_train.npy', y_train)
        # np.save('emnist_X_test.npy', X_test)
        # np.save('emnist_y_test.npy', y_test)
        # np.save('emnist_X_val.npy', X_val)
        # np.save('emnist_y_val.npy', y_val)
        
        # Load the prepared data
        X_train = np.load('emnist_X_train.npy')
        y_train = np.load('emnist_y_train.npy')
        X_test = np.load('emnist_X_test.npy')
        y_test = np.load('emnist_y_test.npy')
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 
        X_val = np.load('emnist_X_val.npy')
        y_val = np.load('emnist_y_val.npy')
    elif dataset=="cifar10":    
        # Load CIFAR-10 dataset
        (X, y), (X_test, y_test) = cifar10.load_data()
        X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=0.96, random_state=23478, stratify=y)
        # Normalize the pixel values to the range [0, 1]
        X_train = X_train.astype('float32') / 255.0
        y_train = np.squeeze(y_train)
        
        X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.995, random_state=42, stratify=y_rem)
        X_val = X_val.astype('float32') / 255.0
        y_val = np.squeeze(y_val)
        
        X_test, _, y_test, _ = train_test_split(X_test, y_test, test_size=0.995, random_state=42, stratify=y_test)
        X_test = X_test.astype('float32') / 255.0
        y_test = np.squeeze(y_test)
        print(y_train.shape)
        print(y_test.shape)
        print(y_val.shape)
    elif dataset=="tiny-imagenet":
        # # Define paths
        # dataset_dir = '/work/assist/experiments/tiny-imagenet-200'
        # train_dir = f'{dataset_dir}/train'

        # # Parameters
        # batch_size = 32
        # img_size = (224, 224)  # ViT models typically use 224x224 images
        
        # # Load datasets
        # train_dataset = image_dataset_from_directory(
        #     train_dir,
        #     image_size=img_size,
        #     batch_size=batch_size,
        #     label_mode='int'
        # )

        # train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        # # Extract images and labels from the datasets
        # X_train, y_train = flatten_dataset(train_dataset)
        # X_train, X_rem, y_train, y_rem = train_test_split(
        #     X_train, y_train, test_size=0.99, stratify=y_train, random_state=123
        # )
        # # X_val, y_val = extract_images_labels(validation_dataset)
        # X_val, X_rem, y_val, y_rem = train_test_split(
        #     X_rem, y_rem, test_size=0.997, stratify=y_rem, random_state=123
        # )
        # # X_test, y_test = extract_images_labels(test_dataset)
        # X_test, X_temp, y_test, y_temp = train_test_split(
        #     X_rem, y_rem, test_size=0.997, stratify=y_rem, random_state=123
        # )

        # # Save the arrays for later use
        # np.save('tiny_imagenet_X_train.npy', X_train)
        # np.save('tiny_imagenet_y_train.npy', y_train)
        # np.save('tiny_imagenet_X_val.npy', X_val)
        # np.save('tiny_imagenet_y_val.npy', y_val)
        # np.save('tiny_imagenet_X_test.npy', X_test)
        # np.save('tiny_imagenet_y_test.npy', y_test)

        X_train = np.load('tiny_imagenet_X_train.npy')
        y_train = np.load('tiny_imagenet_y_train.npy')
        X_val = np.load('tiny_imagenet_X_val.npy')
        y_val = np.load('tiny_imagenet_y_val.npy')
        X_test = np.load('tiny_imagenet_X_test.npy')
        y_test = np.load('tiny_imagenet_y_test.npy')
    return X_train, y_train, X_test, y_test, X_val, y_val
