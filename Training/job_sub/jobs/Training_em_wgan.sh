#! /bin/bash
######## Part 1 #########
# Script parameters     #
#########################
  
# Specify the partition name from which resources will be allocated, mandatory option
#SBATCH --partition=gpu
  
# Specify the QOS, mandatory option
#SBATCH --qos=normal
  
# Specify which group you belong to, mandatory option
# This is for the accounting, so if you belong to many group,
#SBATCH --account=junogpu
  
# Specify your job name, optional option, but strongly recommand to specify some name
#SBATCH --job-name=train_1
  
# Specify how many cores you will need, default is one if not specified
#SBATCH --ntasks=1
  
# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
#SBATCH --output=/hpcfs/juno/junogpu/fangwx/FastSim/BES/job_sub/jobs/training_em_wgan.out
#SBATCH --error=/hpcfs/juno/junogpu/fangwx/FastSim/BES/job_sub/jobs/training_em_wgan.err
  
# Specify memory to use, or slurm will allocate all available memory in MB
#SBATCH --mem-per-cpu=30720
#SBATCH --cpus-per-task=2  
#
# Specify how many GPU cards to use
#SBATCH --gres=gpu:v100:1 
######## Part 2 ######
# Script workload    #
######################
  
# Replace the following lines with your real workload
  
# list the allocated hosts
echo CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES
hostname
df -h
cd /hpcfs/juno/junogpu/fangwx
source /hpcfs/juno/junogpu/fangwx/setup.sh

/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/BES/models/train_wgan.py --dataset /hpcfs/juno/junogpu/fangwx/FastSim/BES/data/Z_merged/single_em/mc_Hit_Barrel_em_zcut.h5 --nb-epochs 200 --batch-size 512 --latent-size 512 --reg-model-in '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/reco/mc_sig_em_1119.h5' --gen-out '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/gen_model_em_1203w.h5' --comb-out '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/comb_model_em_1203w.h5' --dis-out '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/dis_model_em_1203w.h5' --loss_out '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/wgan_out_1203.txt'  --LoadTrainedWeight False 
