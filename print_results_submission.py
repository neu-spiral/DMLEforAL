import argparse
import numpy as np
import matplotlib.pyplot as plt

markers = ["o", "^", "*", "s", '.']
colors = ['#ff7f00', '#377eb8',  '#4daf4a', '#e41a1c', '#dede00']

def print_accuracies(init_size, num_queries, num_cycles, temperature, seeds, selection, strategy, dataset, obj, 
                      num_epochs, nl, until):
    # Generate file paths based on parameters
    file_paths = [
        f"experiments_stochastic/{dataset}/data_server_withZ_active_learning_results_dataset_{dataset}_init_size_{init_size}_num_queries_{num_queries}_num_cycles_{num_cycles}_temperature_{temperature}_seed_{seed}_selection_{selection}_strategy_{strategy}_obj_{obj}_num_epochs_{num_epochs}.npy"
        for seed in seeds
    ]
    # Load accuracies from numpy files
    accuracies = [np.load(file_path)[:until] for file_path in file_paths]
    # Calculate the mean accuracy for each file
    mean_accuracies = np.mean(accuracies, axis=0)
    std_accuracies = np.std(accuracies, axis=0)
    mean = mean_accuracies[nl]
    std = std_accuracies[nl]
    print(f"Objective_{obj}_Selection_{selection}_$N_l$_{nl}: {mean:.2f} +- {std:.2f}")
    
def print_time(init_size, num_queries, num_cycles, temperature, seeds, selection, strategy, dataset, obj, 
                      num_epochs, nl):
    # Generate file paths based on parameters
    file_paths = [
        f"{dataset}/Z_time_active_learning_results_dataset_{dataset}_init_size_{init_size}_num_queries_{num_queries}_num_cycles_{num_cycles}_temperature_{temperature}_selection_{selection}_strategy_{strategy}_obj_{obj}_num_epochs_{num_epochs}.npy"
        
    ]
    # print(file_paths)
    # Load accuracies from numpy files
    accuracies = [np.load(file_path) for file_path in file_paths]
    
    # Calculate the mean accuracy for each file
    mean_accuracies = np.mean(accuracies, axis=0)
    std_accuracies = np.std(accuracies, axis=0)
    # print(mean_accuracies)
    mean = mean_accuracies
    std = std_accuracies
    print(f"Time Objective_{obj}_Selection_{selection}_$N_l$_{nl}: {mean:.2f} +- {std:.2f}")

# Define values for selection, strategy, and obj
selection_values = ['stochastic_softmax', 'stochastic_power', 'stochastic_softrank', 'topk']
strategy_values = ['entropy'] # 'bald', 'least_confident']
obj_values = ['dmle', 'imle']
init_size = 1
num_queries = 1
num_cycles = 110
until = 110
temperatures = [2.5]
seeds = [204957, 291487, 730292, 982673, 843975, 638291, 495724, 213578]
dataset = 'iris'
num_epochs = 10

nls = [100]

# Use nested loops to iterate over parameter combinations
for selection in selection_values:
    for strategy in strategy_values[:1]:
        for obj in obj_values:
            for temperature in temperatures:
                for nl in nls:
                    print_accuracies(init_size, num_queries, num_cycles, temperature,
                                    seeds, selection, strategy, dataset, obj, num_epochs, nl, until)
