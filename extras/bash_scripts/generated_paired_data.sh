#!/bin/bash 

#SBATCH --job-name=paired_data
#SBATCH --mem-per-cpu=1024
#SBATCH --partition long
#SBATCH --account cvit_bhaasha
#SBATCH --gres=gpu:1
#SBATCH --mincpus=10
#SBATCH --nodes=1
#SBATCH --time 4-00:00:00
#SBATCH --signal=B:HUP@600
#SBATCH -w gnode90

source /home2/aditya1/miniconda3/bin/activate stargan-v2
cd /ssd_scratch/cvit/aditya1/pix2pix_torch/extras
python pix2pix_data.py