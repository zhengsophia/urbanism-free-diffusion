#!/bin/bash
#
#SBATCH --job-name=neurocontrast_base
#SBATCH --account=vision-torralba
#SBATCH --partition=vision-torralba
#SBATCH --qos=vision-torralba-main
#SBATCH --time=99:00:00
#SBATCH --output=/data/scratch/apprisco/logs/job_output_%j.log
#SBATCH --error=/data/scratch/apprisco/logs/job_output_%j.log
#SBATCH --gres=gpu:rtx_3090:7
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=192G
#SBATCH --requeue
# Trap timeout (SIGTERM)
trap 'scontrol requeue $SLURM_JOB_ID' SIGTERM

source /data/vision/torralba/scratch/apprisco/.bashrc
conda activate AWQ
cd /data/vision/torralba/selfmanaged/torralba/scratch/apprisco
echo $(pwd)
echo "My Job ID is $SLURM_JOB_ID"
echo "The time is $(date)"
echo "This job is running on $(hostname)"
echo $(pwd)
#torchrun --nproc_per_node=7 train_model_DDP.py   --topic filtered   --batch-size 8   --num-workers 8   --epochs 20   --lr 5e-5   --resolution 256
#torchrun --nproc_per_node=7 peft_lora_DDP.py --topic road --resolution 256 --batch-size 32 --num-workers 8 --epochs 5 --ids filtered
torchrun --nproc_per_node=7 peft_lora_DDP.py --topic vehicle --resolution 256 --batch-size 32 --num-workers 8 --epochs 5 --load road  --ids filtered,road
torchrun --nproc_per_node=7 peft_lora_DDP.py --topic building --resolution 256 --batch-size 32 --num-workers 8 --epochs 5 --load vehicle --ids filtered,road,vehicle
torchrun --nproc_per_node=7 peft_lora_DDP.py --topic city --resolution 256 --batch-size 32 --num-workers 8 --epochs 5 --load building --ids filtered,road,vehicle,building
torchrun --nproc_per_node=7 peft_lora_DDP.py --topic urban --resolution 256 --batch-size 32 --num-workers 8 --epochs 5 --load city --ids filtered,road,vehicle,building,city
