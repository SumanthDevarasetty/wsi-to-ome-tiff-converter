#!/bin/sh
#SBATCH --account=sarder-hubmap
#SBATCH --qos=sarder-hubmap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256gb
#SBATCH --partition=
#SBATCH --time=100:00:00
#SBATCH --output=kpmp-ome-conversion-batch2.out
#SBATCH --job-name="kpmp-ome-conversion-batch2"
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST
echo "SLURM_NNODES="$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

echo "working directory = "$SLURM_SUBMIT_DIR
ulimit -s unlimited
module list
which python

echo "Launch job"
echo $SLURM_PROCID

module load conda
conda activate /blue/pinaki.sarder/sdevarasetty/conda/envs/kpmpenv
python /blue/pinaki.sarder/sdevarasetty/KPMP/wsi-to-ome-tiff-converter/convert.py

echo "All Done!"