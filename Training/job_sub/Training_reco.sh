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
#SBATCH --output=/hpcfs/juno/junogpu/fangwx/FastSim/BES/job_sub/training_reco.out
#SBATCH --error=/hpcfs/juno/junogpu/fangwx/FastSim/BES/job_sub/training_reco.err
  
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
######/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/train.py --datafile /hpcfs/juno/junogpu/fangwx/FastSim/BES/data/Hit_Barrel_em.h5 --nb-epochs 200 --batch-size 128  --model-out '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/reco_model.h5'
##########/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/train.py --datafile /hpcfs/juno/junogpu/fangwx/FastSim/BES/data/Hit_Barrel_em.h5 --nb-epochs 200 --batch-size 128  --model-out '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/reco_model_v1.h5'
############/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/train.py --datafile /hpcfs/juno/junogpu/fangwx/FastSim/BES/data/Hit_Barrel_em_100k.h5 --nb-epochs 200 --batch-size 128  --model-out '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/reco_model_v0_100k.h5'
#################/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/train_v1.py --datafile /hpcfs/juno/junogpu/fangwx/FastSim/BES/data/dataset.txt --nb-epochs 200 --batch-size 128  --model-out '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/reco_model_em_0922_16t12.h5'

#############/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/train_v2.py --datafile /hpcfs/juno/junogpu/fangwx/FastSim/BES/data/dataset.txt --nb-epochs 200 --batch-size 128  --model-out '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/mc_reco_model_em_1005_zcut_zHigh.h5'
/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/train_v2.py --removeZ True --datafile /hpcfs/juno/junogpu/fangwx/FastSim/BES/data/dataset.txt --nb-epochs 200 --batch-size 128  --model-out '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/mc_reco_model_em_1005_zcut_zLow_test.h5'
