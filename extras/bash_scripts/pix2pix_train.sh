#!/bin/bash 

#SBATCH --job-name=pix2pix
#SBATCH --mem-per-cpu=1024
#SBATCH --partition long
#SBATCH --account cvit_bhaasha
#SBATCH --gres=gpu:1
#SBATCH --mincpus=10
#SBATCH --nodes=1
#SBATCH --time 4-00:00:00
#SBATCH --signal=B:HUP@600
#SBATCH -w gnode88

source /home2/aditya1/miniconda3/bin/activate stargan-v2
cd /ssd_scratch/cvit/aditya1/pix2pix_pytorch
python train.py --dataroot ./datasets/celeba --name celeba_pix2pix --model pix2pix --direction AtoB