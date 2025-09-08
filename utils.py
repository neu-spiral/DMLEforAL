# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 14:03:12 2025

@author: byzkl

This module provides various functions and utilities for active learning,
including uncertainty-based sampling strategies, acquisition functions, and
a custom loss for model training. It includes:

- CustomLoss: A sparse categorical crossentropy loss.
- Monte Carlo dropout predictions for uncertainty estimation.
- BALD acquisition function for Bayesian active learning.
- Pairwise distance-based coreset selection.
- Multiple uncertainty strategies: entropy, BALD, margin, least-confident, random, coreset.
- Boltzmann distribution for probabilistic selection.
- Sampling strategies: random, softmax, power, softrank.
- Statistical bias computation for weighted loss adjustment.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import tensorflow as tf
import numpy as np
from scipy.stats import gumbel_r

# Custom loss class (can be extended to include uncertainty-weighted loss)
class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, uncertainty_weight=1.0, obj="imle"):
        super(CustomLoss, self).__init__()
        self.uncertainty_weight = uncertainty_weight  # weight for uncertainty contribution
        self.obj = obj  # objective type (currently not used in loss calculation)

    def call(self, y_true, y_pred):
        # Standard sparse categorical crossentropy loss
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        return loss

# Perform multiple stochastic forward passes with dropout enabled
def mc_dropout_predictions(model, x, num_samples):
    all_predictions = []
    for _ in range(num_samples):
        # Forward pass with dropout active (training=True)
        predictions = model(x, training=True)
        all_predictions.append(predictions)
    # Stack predictions from all runs along a new axis (num_samples × batch × classes)
    return tf.stack(all_predictions, axis=0)

# Compute BALD (Bayesian Active Learning by Disagreement) acquisition scores
def bald_acquisition(y_pred):
    # Entropy of the mean prediction (predictive entropy)
    entropy = tf.keras.losses.categorical_crossentropy(np.mean(y_pred, axis=0), np.mean(y_pred, axis=0))
    # Expected entropy: average of entropy for each Monte Carlo sample
    expected_entropy = np.mean(tf.keras.losses.categorical_crossentropy(y_pred, y_pred), axis=0)

    # BALD = predictive entropy - expected entropy
    bald_acq = entropy - expected_entropy
    return bald_acq

# Update or compute pairwise distances for coreset-based selection
def update_dist(labeled_features, unlabeled_features, reset_dist=False, min_distances=None):
    if reset_dist:
        min_distances = None
    if unlabeled_features is not None and labeled_features is not None:
        # Compute pairwise Euclidean distances between unlabeled and labeled sets
        dist = pairwise_distances(unlabeled_features, labeled_features, metric='euclidean')
        if min_distances is None:
            # Take minimum distance to any labeled point for each unlabeled sample
            min_distances = np.min(dist, axis=1).reshape(-1, 1)
        else:
            # Update minimum distances with new computed distances
            min_distances = np.minimum(min_distances, dist)
    return min_distances

# Wrapper to get coreset acquisition scores based on minimum distances
def get_acquisition_scores_coreset(unlabeled_features, labeled_features):
    # Reset and calculate minimum distances
    min_distances = update_dist(labeled_features, unlabeled_features, reset_dist=True, min_distances=None)
    # Return as a flat array (1D)
    return min_distances.flatten()

# Compute uncertainty based on the specified strategy
def calculate_uncertainty(outputs, strategy='entropy'):
    if strategy == 'entropy':
        # Entropy-based uncertainty (self-cross entropy)
        uncertainty = tf.keras.losses.categorical_crossentropy(outputs, outputs)
    elif strategy == 'bald':
        # BALD uncertainty
        uncertainty = bald_acquisition(outputs)
    elif strategy == 'margin':
        # Margin-based uncertainty (difference between top two probabilities)
        top_k_values, top_k_indices = tf.nn.top_k(outputs, k=2)
        uncertainty = -(top_k_values[:, 0] - top_k_values[:, 1])
    elif strategy == 'least_confident':
        # Least confident uncertainty = 1 - max probability
        top_k_values = tf.reduce_max(outputs, axis=1)
        uncertainty = 1 - top_k_values
    elif strategy == 'random':
        # Random uncertainty (for baseline random sampling)
        uncertainty = tf.random.uniform(shape=(tf.shape(outputs)[0],))
    elif strategy == 'coreset':
        # Coreset-based uncertainty using pairwise distances
        unlabeled_features = outputs[0]
        labeled_features = outputs[1]
        uncertainty = get_acquisition_scores_coreset(unlabeled_features, labeled_features)
        uncertainty = tf.convert_to_tensor(uncertainty, dtype=tf.float32)
    return uncertainty

# Convert uncertainty values into probabilities using Boltzmann distribution
def boltzmann_distribution(uncertainty_values, temperature=1.0):
    """
    Calculate the Boltzmann distribution for a set of uncertainty values.

    Parameters:
    - uncertainty_values (tf.Tensor): Uncertainty values.
    - temperature (float): Temperature parameter for controlling the distribution sharpness.

    Returns:
    - tf.Tensor: Probability distribution.
    """
    with tf.name_scope("boltzmann_distribution"):
        # Exponentially scale negative uncertainty (lower uncertainty → higher weight)
        exp_values = tf.exp(-uncertainty_values * temperature)
        # Normalize to sum to 1 (probability distribution)
        prob_distribution = exp_values / tf.reduce_sum(exp_values)
    return prob_distribution

# Random sampling strategy (baseline)
def get_random_samples(scores_N, aquisition_batch_size):
    N = len(scores_N)
    aquisition_batch_size = min(aquisition_batch_size, N)  # Ensure batch size ≤ total
    # Uniform random selection without replacement
    indices = np.random.choice(N, size=aquisition_batch_size, replace=False)
    return indices.tolist()

# Softmax-based stochastic sampling using Gumbel noise
def get_softmax_samples(scores_N, beta, aquisition_batch_size):
    if beta == 0.0:
        # Beta = 0 means uniform random selection
        return get_random_samples(scores_N, aquisition_batch_size=aquisition_batch_size)
    N = len(scores_N)
    # Add Gumbel noise to scores for stochastic ranking
    noised_scores_N = scores_N + gumbel_r.rvs(loc=0, scale=1 / beta, size=N)
    # Select top-k indices based on noisy scores
    indices = np.argsort(noised_scores_N)[-aquisition_batch_size:]
    return indices[::-1]  # Reverse to get descending order

# Power sampling: apply log before softmax sampling
def get_power_samples(scores_N, beta, aquisition_batch_size):
    return get_softmax_samples(np.log(scores_N), beta=beta, aquisition_batch_size=aquisition_batch_size)

# Softrank-based sampling: uses rank information with power sampling
def get_softrank_samples(scores_N, beta, aquisition_batch_size):
    N = len(scores_N)
    # Sort scores in descending order and convert to ranks (1 = highest)
    sorted_indices_N = np.argsort(scores_N)[::-1]
    ranks_N = np.argsort(sorted_indices_N) + 1
    # Sample based on inverse rank weights
    return get_power_samples(1 / ranks_N, beta=beta, aquisition_batch_size=aquisition_batch_size)

# Unified function for selecting unlabeled instances using a specified strategy
def sample_unlabeled_instances(uncertainty, prob_distribution, beta=1.0, num_samples=5, selection='topk'):
    if selection == "topk":
        # Deterministic: select top-k by highest uncertainty
        query_indices = np.argsort(uncertainty)[::-1][:num_samples]
    elif selection == "randomized":
        # Randomized selection based on probability distribution
        query_indices = np.random.choice(len(prob_distribution), size=num_samples, replace=False, p=prob_distribution.numpy())
    elif selection == "stochastic_power":
        query_indices = get_power_samples(uncertainty, beta, num_samples)
    elif selection == "stochastic_softmax":
        query_indices = get_softmax_samples(uncertainty, beta, num_samples)
    elif selection == "stochastic_softrank":
        query_indices = get_softrank_samples(uncertainty, beta, num_samples)
    return query_indices

# Compute statistical bias correction for active learning weighting
def statistical_bias(model, X_unlabeled, X_labeled, y_labeled, q):
    # Total number of samples (labeled + unlabeled)
    N = len(X_unlabeled) + len(X_labeled)
    # Number of labeled samples
    M = len(X_labeled)
    # Indexing from 1 to M for labeled data
    m = np.arange(1, M + 1)

    # Get model predictions for labeled data (training=True to include dropout if used)
    prediction_N = model(X_labeled, training=True)
    # Compute per-sample negative log-likelihood
    raw_nll_N = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction_N, labels=y_labeled)
    nll = tf.reduce_sum(raw_nll_N)  # total negative log-likelihood

    # Compute sample-specific weights based on q and sample index
    weight = (1 / np.array(q)) + M - m
    weight = weight / N  # Normalize weights

    # Apply weights to NLL for bias correction
    weighted_nll = tf.reduce_sum(weight * raw_nll_N)

    return weighted_nll
