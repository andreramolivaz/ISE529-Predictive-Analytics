#!/bin/bash
#SBATCH --job-name=cnn_crack
#SBATCH --account=tma26892_1425
#SBATCH --partition=main     # CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=out/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ramoliva@usc.edu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate Py310

cd $SLURM_SUBMIT_DIR
python src/train_cnn.py \
       --data_dir /scratch1/$USER/datasets/concrete \
       --out_dir  out/$SLURM_JOB_ID \
       --epochs   10
