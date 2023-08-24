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
#SBATCH --output=/hpcfs/juno/junogpu/fangwx/FastSim/BES/job_sub/jobs/testing_reco_em_Low.out
#SBATCH --error=/hpcfs/juno/junogpu/fangwx/FastSim/BES/job_sub/jobs/testing_reco_em_Low.err
  
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

#########/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/reco_test_v1.py --removeZ True --datafile /hpcfs/juno/junogpu/fangwx/FastSim/BES/data/test_data/mc_Hit_Barrel_em_test_zcut_Low.h5 --batch-size 5000 --model-in '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/reco/mc_em_Low_1008.h5' --output '/hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/mc_reco_result_em_Low_1008.h5'
#########/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/reco_test_v2.py --removeZ True --datafile /hpcfs/juno/junogpu/fangwx/FastSim/BES/data/test_data/mc_Hit_Barrel_em_test_zcut_Low.h5 --batch-size 5000 --model-in '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/reco/mc_em_Low_1025.h5' --output '/hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/reco_result_em_Low_1025.h5'
##########/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/reco_test_v2.py --removeZ True --datafile /hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen1027_em_Low.h5 --batch-size 5000 --model-in '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/reco/mc_em_Low_1025.h5' --output '/hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/fale_reco_result_em_Low_1027.h5'
#############/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/reco_test_v3.py --datafile /hpcfs/juno/junogpu/fangwx/FastSim/BES/data/test_data/mc_Hit_Barrel_em_test_zcut_Low.h5 --batch-size 5000 --model-in '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/reco/mc_em_Low_1028.h5' --output '/hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/reco_result_em_Low_1028.h5'
/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/reco_test_v4.py --datafile /hpcfs/juno/junogpu/fangwx/FastSim/BES/data/test_data/mc_Hit_Barrel_em_test_zcut_Low.h5 --batch-size 5000 --model-in '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/reco/mc_em_Low_1106.h5' --output '/hpcfs/juno/junogpu/fangwx/FastSim/BES/reco/reco_result_em_Low_1107.h5'
