import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 
# Specify the folder path
folder_path = 'final_paper_plots'

# Check if the folder exists, and create it if not
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

i = 0
markers = ["o", "^", "s", "*", '.', "d"]
colors = ['#e6550d', '#0072B2', '#E69F00', '#E69F00', '#016c59', '#756bb1', '#fecc5c']

def plot_accuracies(init_size, num_queries, num_cycles, temperature, seeds, selection, strategy, dataset, obj, num_epochs, imagenet=""):
    # Generate file paths based on parameters
    file_paths = [
        imagenet+f"experiments_stochastic/{dataset}/data_server_active_learning_results_dataset_{dataset}_init_size_{init_size}_num_queries_{num_queries}_num_cycles_{num_cycles}_temperature_{temperature}_seed_{seed}_selection_{selection}_strategy_{strategy}_obj_{obj}_num_epochs_{num_epochs}.npy"
        for seed in seeds
    ]
    print(file_paths)
    # Load accuracies from numpy files
    accuracies = [np.load(file_path)[:until] for file_path in file_paths]
    
    # Calculate the mean accuracy for each file
    mean_accuracies = np.mean(accuracies, axis=0)
    std_accuracies = np.std(accuracies, axis=0)
    print(mean_accuracies.shape)
    # Plotting
    if selection == "topk":
        selection = "Top-k Sampling"
    elif selection == "stochastic_power":
        selection = "SPS"
    elif selection == "stochastic_softmax":
        selection = "SSMS"
    elif selection == "stochastic_softrank":
        selection = "SSRS"
        
    obj = str.upper(obj)
    plt.plot(range(1, until + 1), mean_accuracies, marker=markers[i], color=colors[i], label=f'{obj}')#'/temperature={temperature}')
    plt.fill_between(range(1, until + 1), mean_accuracies-std_accuracies, mean_accuracies+std_accuracies, alpha=0.5, color=colors[i])


def plot_Z(init_size, num_queries, num_cycles, temperature, seeds, selection, strategy, dataset, obj, num_epochs):
    if obj == "dmle":
        # Generate file paths based on parameters
        file_paths = [
            f"experiments_stochastic/{dataset}/data_server_active_learning_results_dataset_{dataset}_init_size_{init_size}_num_queries_{num_queries}_num_cycles_{num_cycles}_temperature_{temperature}_seed_{seed}_selection_{selection}_strategy_{strategy}_obj_{obj}_num_epochs_{num_epochs}.npy"
            for seed in seeds
        ]
        print(file_paths)
        # Load accuracies from numpy files
        unlabeled_accuracies = [np.log(np.load(file_path))[:110] for file_path in file_paths]
        # Calculate the mean accuracy for each file
        mean_unlabeled_accuracies = np.mean(unlabeled_accuracies, axis=0) * (-1)
        std_unlabeled_accuracies = np.std(unlabeled_accuracies, axis=0) * (-1)
        print(unlabeled_accuracies[-1])
        print(mean_unlabeled_accuracies[-1])
        # Plotting
        
        selection = str.capitalize(selection)
        obj = str.upper(obj)
        plt.plot(range(1, until + 1), mean_unlabeled_accuracies, marker=markers[i], color=colors[i], label=f'{obj}/{selection}')#'/temperature={temperature}')
        plt.fill_between(range(1, until + 1), mean_unlabeled_accuracies-std_unlabeled_accuracies, mean_unlabeled_accuracies+std_unlabeled_accuracies, alpha=0.5, color=colors[i])


# Define values for stochastic_power, strategy, and obj
selection_values = ['stochastic_softmax', 'stochastic_power', 'stochastic_softrank', 'topk']#, 'randomized']#, 'topk']#, 'stochastic_power']#, 'random']
strategy_values = ['entropy']#, 'bald', 'least_confident']

obj_values = ['dmle', 'imle']
init_size = 1
num_queries = 10
num_cycles = 110
until = 11
temperatures = [1.0]
seeds = [204957, 291487, 730292, 982673, 843975, 638291, 495724, 213578]#, 734853, 108344, 214346, 468763]
dataset = 'iris'
num_epochs = 10
y_lim = (0.0,1.1)
imagenet = ""

dataset_name = str.capitalize(dataset)
dataset_name.replace("_", "")
fig = plt.figure(figsize=(10, 6))   
# Use nested loops to iterate over parameter combinations
for selection in selection_values:
    fig = plt.figure(figsize=(10, 6))   
    for strategy in strategy_values:
        for obj in obj_values:
            for temperature in temperatures:
                plot_accuracies(init_size, num_queries, num_cycles, temperature,
                                seeds, selection, strategy, dataset, obj, num_epochs, imagenet=imagenet)
                i += 1
    
    plt.xlabel('Active Learning Cycles', fontsize=20)
    plt.ylabel('Mean Accuracy', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(y_lim)
    plt.locator_params(axis='x', integer=True)
    plt.legend(fontsize="20", loc ="lower right")
    i = 0
    fig.savefig(folder_path+f'/{dataset}_cycle_{until}_query_{num_queries}_selection_{selection}_acc.pdf', bbox_inches='tight', pad_inches=0, format='pdf')
