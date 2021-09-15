#$ -q short.qc -j y -o /well/saxe/users/qbe080/logs -wd /well/saxe/users/qbe080/Biomed-Research-Project
source ~/.bashrc
conda activate pytorch_paco
python train_model.py