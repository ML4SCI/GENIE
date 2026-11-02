#!/bin/bash

#SBATCH -p regular1                         # Use GPU nodes
#SBATCH -t 11:50:00                 # Wall clock time (HH:MM:SS)

#SBATCH --cpus-per-task=8           # CPU cores per GPU task (typical ratio)
#SBATCH --mem=56000                 # Memory for this job
#SBATCH -J gen_hdf5   # Job name
#SBATCH -o logs/output.out          # Stdout/stderr output (creates logs/my_gpu_job-<jobid>.out)
#SBATCH -e logs/error.err           # Standard error file

# ==== Info part (say things) ===== #
echo '------------------------------------------------------'
echo 'This job is allocated on '$SLURM_JOB_CPUS_PER_NODE' cpu(s)'
echo 'Job is running on node(s): '
echo  $SLURM_JOB_NODELIST
echo '------------------------------------------------------'
echo 'WORKINFO:'
echo 'SLURM: sbatch is running on      '$SLURM_SUBMIT_HOST
echo 'SLURM: executing on cluster      '$SLURM_CLUSTER_NAME
echo 'SLURM: executing on partition    '$SLURM_JOB_PARTITION
echo 'SLURM: working directory is      '$SLURM_SUBMIT_DIR
# echo 'SLURM: current home directory is '$(getent passwd $SLURM_JOB_ACCOUNT | cut -d: -f6)
echo ""
echo 'JOBINFO:'
echo 'SLURM: job identifier is         '$SLURM_JOBID
echo 'SLURM: job name is               '$SLURM_JOB_NAME
echo ""
echo 'NODEINFO:'
echo 'SLURM: number of nodes is        '$SLURM_JOB_NUM_NODES
echo 'SLURM: number of cpus/node is    '$SLURM_JOB_CPUS_PER_NODE
echo 'SLURM: number of gpus/node is    '$SLURM_GPUS_PER_NODE
echo '------------------------------------------------------'
# ==== End of Info part (say things) ===== 



# Optional: confirm GPU visibility
echo "CUDA devices visible:"
nvidia-smi

#################

# Activate environment
source ~/.bashrc
conda activate ML


#################

# Run script
bash run_calochallenge_eval.sh  ae_recon_ds2_full.hdf5  dataset_2_2.hdf5

