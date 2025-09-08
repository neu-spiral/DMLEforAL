#!/bin/bash
#set a job name  
#SBATCH --job-name=dmle_exp
#################  
#a file for job output, you can check job progress
#SBATCH --output=im_filt.%j.out
#################
# a file for errors from the job
#SBATCH --error=im_filt.%j.err
#################
#time you think you need; default is one day
#in minutes in this case, hh:mm:ss
#SBATCH --time=23:00:00
#################
#number of tasks you are requesting
#SBATCH -N 1
##SBATCH --exclusive
#################
#partition to use
##SBATCH --partition=cpu
## SBATCH --gres=gpu:1
## SBATCH --mem=500Gb
#################
#number of nodes to distribute n tasks across
#################

python main.py $1 $2 $3 $4 $5 $6 $7 $8 $9
