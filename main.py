# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 13:54:23 2025

@author: byzkl
"""

# ====== IMPORTS ======
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import pairwise_distances
from tensorflow.keras import backend as K
import argparse
import tensorflow as tf
import os
import random
import time 

# Custom modules (model definition, data loaders, and utilities)
from model import load_model
from data import load_data
from utils import CustomLoss, calculate_uncertainty, mc_dropout_predictions, boltzmann_distribution, sample_unlabeled_instances, statistical_bias
from tensorflow.keras import Model

# Set the image data format for Keras (channels_last is default for TF)
K.set_image_data_format('channels_last')


# ====== ACTIVE LEARNING CYCLE ======
def active_learning_cycle(model, criterion, optimizer, X_labeled, y_labeled, X_unlabeled, y_unlabeled, 
                          X_test, y_test, q, num_queries=10, sampling_strategy='entropy', 
                          selection="topk", obj="imle"):
    """
    Perform a single active learning cycle:
      - Predict on unlabeled data
      - Calculate uncertainty based on the chosen strategy
      - Select samples to label
      - Retrain the model with updated labeled set
    """

    # Inference on unlabeled pool
    unlabeled_outputs = model.predict(X_unlabeled)
    predicted = np.argmax(unlabeled_outputs, axis=1)
    unlabeled_accuracy = np.sum(predicted == y_unlabeled) / len(y_unlabeled)

    # Calculate uncertainty for the unlabeled pool
    if strategy == 'bald':
        # Monte Carlo dropout for BALD strategy
        num_mc_samples = 100
        unlabeled_mc_outputs = mc_dropout_predictions(model, X_unlabeled, num_mc_samples)
        uncertainty = calculate_uncertainty(unlabeled_mc_outputs, strategy=strategy)
    elif strategy == 'coreset':
        # Extract features for coreset strategy
        unlabeled_features = feature_extractor.predict(X_unlabeled)
        labeled_features = feature_extractor.predict(X_labeled)
        features = [unlabeled_features, labeled_features]
        uncertainty = calculate_uncertainty(features, strategy=strategy)
    elif strategy == 'badge':
        # Use labeled and unlabeled sets for BADGE strategy
        features = [X_labeled, X_unlabeled, y_labeled]
        uncertainty = calculate_uncertainty(features, strategy=strategy)
    else:   
        # Default uncertainty strategy (entropy, margin, etc.)
        uncertainty = calculate_uncertainty(unlabeled_outputs, strategy=strategy)

    # Boltzmann distribution for probabilistic selection
    prob_distribution = boltzmann_distribution(uncertainty, temperature)

    # Query selection based on strategy and selection method
    query_indices = sample_unlabeled_instances(uncertainty, prob_distribution, beta=temperature, 
                                               num_samples=num_queries, selection=selection)

    # Track weights for statistical bias correction
    if obj == "statistical_bias":
        q.append(prob_distribution[query_indices[0]])

    # Add selected samples to labeled set
    X_query = X_unlabeled[query_indices].copy()
    y_query = y_unlabeled[query_indices].copy()
    X_labeled = np.concatenate([X_labeled, X_query])
    y_labeled = np.concatenate([y_labeled, y_query])

    # Remove queried samples from unlabeled set
    X_unlabeled = np.delete(X_unlabeled, query_indices, axis=0)
    y_unlabeled = np.delete(y_unlabeled, query_indices, axis=0)

    # Train on updated labeled dataset
    for epoch in range(num_epochs):
        # Shuffle labeled data each epoch
        shuffle_indices = np.random.permutation(len(X_labeled))
        X_labeled = X_labeled[shuffle_indices]
        y_labeled = y_labeled[shuffle_indices]

        dep_time = time.time()
        with tf.GradientTape() as tape:
            outputs = model(X_labeled, training=True)

            # Compute uncertainty on labeled data (if needed for loss)
            if strategy == 'bald':
                mc_outputs = mc_dropout_predictions(model, X_labeled, num_mc_samples)
                labeled_uncertainty = calculate_uncertainty(mc_outputs, strategy=strategy)
            elif strategy in ['coreset', 'badge']:
                features = feature_extractor(X_labeled, training=True)
                dist_matrix = pairwise_distances(features, metric='euclidean')
                labeled_uncertainty = tf.convert_to_tensor(np.mean(dist_matrix, axis=1), dtype=tf.float32)
            else:   
                labeled_uncertainty = calculate_uncertainty(outputs, strategy=strategy)

            # Objective-dependent loss calculation
            if obj == 'imle':
                loss = criterion(y_labeled, outputs)
            elif obj == 'dmle' and selection in ["stochastic_softmax", "randomized"]:
                loss = criterion(y_labeled, outputs) - temperature * labeled_uncertainty
            elif obj == 'dmle' and selection == "stochastic_power":
                loss = criterion(y_labeled, outputs) - temperature * tf.math.log(labeled_uncertainty)
            elif obj == 'dmle' and selection in ["stochastic_softrank", "topk"]:
                descending_order = tf.argsort(labeled_uncertainty, direction='DESCENDING')
                ranked_arr = tf.cast(tf.argsort(descending_order) + 1, tf.float32)
                loss = criterion(y_labeled, outputs) + temperature * tf.math.log(ranked_arr)
            elif obj == "statistical_bias":
                loss = statistical_bias(model, X_unlabeled, X_labeled, y_labeled, q)

        # Gradient computation and optimization
        dep_elapsed_time = time.time() - dep_time
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    # Evaluate the updated model
    test_outputs = model.predict(X_test)
    predicted = np.argmax(test_outputs, axis=1)
    accuracy = np.sum(predicted == y_test) / len(y_test)

    return model, X_labeled, y_labeled, X_unlabeled, y_unlabeled, accuracy, \
           labeled_uncertainty.numpy(), uncertainty.numpy(), unlabeled_accuracy, dep_elapsed_time


# ====== ACTIVE LEARNING EXPERIMENT ======
def active_learning_experiment(seed, X_train, y_train, X_test, y_test, X_val, y_val, 
                               sampling_strategy='entropy', selection="topk", obj="imle", 
                               mode="test"):
    """
    Run a full active learning experiment with multiple cycles.
    """

    # If in validation mode, replace test set
    if mode == "val":
        X_test, y_test = X_val, y_val

    # Set random seeds for reproducibility
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Shuffle initial training data
    shuffle_indices = np.random.permutation(len(X_train))
    X_train, y_train = X_train[shuffle_indices], y_train[shuffle_indices]

    # Preprocess for iris dataset
    if dataset == "iris":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Initialize model, loss, optimizer
    model = load_model(dataset)
    criterion = CustomLoss(uncertainty_weight=temperature, obj=obj)
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss=criterion, metrics=['accuracy'])

    # Initialize labeled set
    initial_labeled_indices = np.random.choice(len(X_train), size=init_size, replace=False)
    X_labeled, y_labeled = X_train[initial_labeled_indices], y_train[initial_labeled_indices]

    # Remaining as unlabeled set
    X_unlabeled = np.delete(X_train, initial_labeled_indices, axis=0)
    y_unlabeled = np.delete(y_train, initial_labeled_indices, axis=0)

    # Storage for metrics
    accuracies, labeled_uncertainties, uncertainties = [], [], []
    unlabeled_accuracies, dependency_time = [], []
    q = [1.0] * len(X_labeled)

    # Active learning loop
    for cycle in range(num_cycles):
        model, X_labeled, y_labeled, X_unlabeled, y_unlabeled, accuracy, labeled_uncertainty, \
        uncertainty, unlabeled_accuracy, dep_elapsed_time = active_learning_cycle(
            model, criterion, optimizer, X_labeled, y_labeled, X_unlabeled, y_unlabeled,
            X_test, y_test, q, num_queries=num_queries, sampling_strategy=sampling_strategy, 
            selection=selection, obj=obj
        )

        # Store metrics per cycle
        accuracies.append(accuracy)
        uncertainties.append(uncertainty)
        labeled_uncertainties.append(labeled_uncertainty)
        unlabeled_accuracies.append(unlabeled_accuracy)
        dependency_time.append(dep_elapsed_time)

        # Save intermediate results after each cycle
        filename = f"{dataset}/data_server_smaller_no_update_active_learning_results_dataset_{dataset}_" \
                   f"init_size_{init_size}_num_queries_{num_queries}_num_cycles_{num_cycles}_" \
                   f"temperature_{temperature}_seed_{seed}_selection_{selection}_strategy_{strategy}_" \
                   f"obj_{obj}_num_epochs_{num_epochs}.npy"
        np.save(filename, accuracies)
        # Similar saving for other arrays (uncertainties, labeled_uncertainties, etc.)

    print('Active Learning Cycles Finished')

    return accuracies, np.array(labeled_uncertainties, dtype=object), \
           np.array(uncertainties, dtype=object), unlabeled_accuracies, \
           X_labeled, y_labeled, dependency_time


# ====== MAIN ENTRY POINT ======
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='NN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('init_size', default=1, type=int)
    parser.add_argument('num_queries', default=1, type=int)
    parser.add_argument('num_cycles', default=100, type=int)
    parser.add_argument('temperature', default=1.0, type=float)
    parser.add_argument('seed', default=1, type=int)
    parser.add_argument('selection', type=str, default='boltzmann',
                        choices=['randomized', 'topk', 'stochastic_softmax', 'stochastic_power', 'stochastic_softrank'])
    parser.add_argument('strategy', type=str, default='entropy',
                        choices=['entropy', 'margin', 'least_confident', 'margin_energy',
                                 'least_confident_energy','random','bald','coreset','badge'])
    parser.add_argument('dataset', type=str, default='iris',
                        choices=['iris', 'mnist', 'fashion_mnist', 'reuters', 'svhn', 'emnist', 'cifar10', 'tiny-imagenet'])
    parser.add_argument('obj', type=str, default='dmle', choices=['imle', 'dmle', 'statistical_bias'])
    
    args = parser.parse_args()

    # Extract arguments
    init_size = args.init_size
    num_queries = args.num_queries
    num_cycles = args.num_cycles
    temperature = args.temperature
    seed = args.seed
    selection = args.selection
    mode = "test"
    strategy = args.strategy
    dataset = args.dataset
    obj = args.obj
    num_epochs = 10

    # Load dataset
    start_time = time.time()
    X_train, y_train, X_test, y_test, X_val, y_val = load_data(dataset, seed)

    # Ensure labels are integers
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    y_val = y_val.astype(int) if dataset != 'iris' else []

    # Save full dataset snapshot
    # (so experiments can be reproduced)
    filename = f"{dataset}/data_server_smaller_no_update_X_all_active_learning_results_dataset_{dataset}_init_size_{init_size}_" \
               f"num_queries_{num_queries}_num_cycles_{num_cycles}_temperature_{temperature}_seed_{seed}_" \
               f"selection_{selection}_strategy_{strategy}_obj_{obj}_num_epochs_{num_epochs}.npy"
    np.save(filename, X_train)

    # Initialize model and feature extractor if needed
    model = load_model(dataset)
    if strategy == "coreset":
        feature_extractor = Model(inputs=model.input, outputs=model.get_layer('feature_layer').output)

    print(f"#########################################")
    print(f"Seed {seed}, Sampling Strategy: {strategy}, Objective: {obj}, Selection: {selection}")

    # Run active learning experiment
    accuracies, labeled_uncertainties, uncertainties, unlabeled_accuracies, \
    X_labeled, y_labeled, dependency_time = active_learning_experiment(
        seed, X_train, y_train, X_test, y_test, X_val, y_val, 
        sampling_strategy=strategy, selection=selection, obj=obj, mode=mode
    )

    # Measure total execution time
    elapsed_time = time.time() - start_time

    # Ensure dataset folder exists
    folder_path = f"{dataset}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    print(accuracies)
    # Save final results (accuracies, uncertainties, labeled set, etc.)
    # Multiple np.save() calls follow here for each metric and data snapshot
    filename = f"{dataset}/active_learning_results_dataset_{dataset}_init_size_{init_size}_num_queries_{num_queries}_num_cycles_{num_cycles}_temperature_{temperature}_seed_{seed}_selection_{selection}_strategy_{strategy}_obj_{obj}_num_epochs_{num_epochs}.npy"
    np.save(filename, accuracies)
    filename = f"{dataset}/labeled_uncertainty_active_learning_results_dataset_{dataset}_init_size_{init_size}_num_queries_{num_queries}_num_cycles_{num_cycles}_temperature_{temperature}_seed_{seed}_selection_{selection}_strategy_{strategy}_obj_{obj}_num_epochs_{num_epochs}.npy"
    np.save(filename, labeled_uncertainties)
    filename = f"{dataset}/uncertainty_active_learning_results_dataset_{dataset}_init_size_{init_size}_num_queries_{num_queries}_num_cycles_{num_cycles}_temperature_{temperature}_seed_{seed}_selection_{selection}_strategy_{strategy}_obj_{obj}_num_epochs_{num_epochs}.npy"
    np.save(filename, uncertainties)
    filename = f"{dataset}/unlabeled_active_learning_results_dataset_{dataset}_init_size_{init_size}_num_queries_{num_queries}_num_cycles_{num_cycles}_temperature_{temperature}_seed_{seed}_selection_{selection}_strategy_{strategy}_obj_{obj}_num_epochs_{num_epochs}.npy"
    np.save(filename, unlabeled_accuracies)
    filename = f"{dataset}/X_labeled_active_learning_results_dataset_{dataset}_init_size_{init_size}_num_queries_{num_queries}_num_cycles_{num_cycles}_temperature_{temperature}_seed_{seed}_selection_{selection}_strategy_{strategy}_obj_{obj}_num_epochs_{num_epochs}.npy"
    np.save(filename, X_labeled)
    filename = f"{dataset}/y_labeled_active_learning_results_dataset_{dataset}_init_size_{init_size}_num_queries_{num_queries}_num_cycles_{num_cycles}_temperature_{temperature}_seed_{seed}_selection_{selection}_strategy_{strategy}_obj_{obj}_num_epochs_{num_epochs}.npy"
    np.save(filename, y_labeled)
    filename = f"{dataset}/time_active_learning_results_dataset_{dataset}_init_size_{init_size}_num_queries_{num_queries}_num_cycles_{num_cycles}_temperature_{temperature}_seed_{seed}_selection_{selection}_strategy_{strategy}_obj_{obj}_num_epochs_{num_epochs}.npy"
    np.save(filename, elapsed_time)
    filename = f"{dataset}/dependency_time_active_learning_results_dataset_{dataset}_init_size_{init_size}_num_queries_{num_queries}_num_cycles_{num_cycles}_temperature_{temperature}_seed_{seed}_selection_{selection}_strategy_{strategy}_obj_{obj}_num_epochs_{num_epochs}.npy"
    np.save(filename, dependency_time)