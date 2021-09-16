#!/bin/bash
#$ -q short.qc 
#$ -j y
#$ -r no 
#$ -o /well/saxe/users/qbe080/logs 
#$ -wd /well/saxe/users/qbe080/Biomed-Research-Project

echo "------------------------------------------------"
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "------------------------------------------------"

source ~/.bashrc
conda activate pytorch_paco
python train_model.py

