#!/bin/bash
for init_size in "1"
do 
    for num_queries in "1"
    do
        for num_cycles in "502"
        do
            for temperature in "1.0" #"0.1" "1.0" "10.0" "100.0" "1000.0"
            do
                # for seed in "1" "2" "3" "4" "5" "6" "7" "8" "9" "10"
                for seed in "204957" "291487" "730292" "982673" "843975" "638291" "495724" "213578"  # "734853" "108344" "214346" "468763" "204957" "291487" "730292" "982673" "843975" "638291" "495724" "213578" #"956327" # "24234234" "53452" "2353465" "9587945" "98348457" "756372" "6774385" "1385475"
                  do
                       for selection in "stochastic_softmax" #"stochastic_softrank" "topk" #"stochastic_softmax" #"stochastic_power" "stochastic_softrank" "topk" # "stochastic_softmax" #"stochastic_power" "stochastic_softrank" "topk" #"stochastic_softmax" #"stochastic_power" "stochastic_softrank" "topk" #"stochastic_power" #"stochastic_softrank" "stochastic_softmax" "topk" "stochastic_softmax"
                        do
                            for strategy in "entropy" #"least_confident" "bald" #"coreset" # "least_confident" "margin" "least_confident_energy" "margin_energy"
                            do
                                for dataset in "mnist" #"tiny-imagenet" "iris" "svhn" "reuters" "emnist" "fashion_mnist" "cifar10" #"isic" #"iris" #"svhn" "reuters" "emnist" "fashion_mnist" "mnist" "iris" # "caltech" "cifar10" "emnist" # "mnist" "fashion_mnist" #"mnist" "fashion_mnist" "breast_cancer" "diabetes" "wine" "imdb" "reuters" "iris"
                                do
                                    for obj in "imle" #"imle" #"statistical_bias"
                                    do
                                        # for mode in "test" # "val"
                                        # do
                                        sbatch execute_combined_cpu.bash $init_size $num_queries $num_cycles $temperature $seed $selection $strategy $dataset $obj
                                        # done
                                    done
                                done
                            done
                        done
                  done
            done
        done
    done
done
