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
#SBATCH --output=/hpcfs/juno/junogpu/fangwx/FastSim/BES/job_sub/jobs/training_em_Low.out
#SBATCH --error=/hpcfs/juno/junogpu/fangwx/FastSim/BES/job_sub/jobs/training_em_Low.err
  
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

##################/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/BES/models/train_v3.py --removeZ True --dataset /hpcfs/juno/junogpu/fangwx/FastSim/BES/data/mc_Hit_Barrel_em_zcut_Low.h5 --nb-epochs 100 --batch-size 128 --latent-size 512 --reg-model-in '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/reco/mc_em_Low_1008.h5' --gen-out '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/gen_model_em_Low_1017.h5' --comb-out '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/comb_model_em_Low_1017.h5' --dis-out '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/dis_model_em_Low_1017.h5'
#######/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/BES/models/train_v4.py --removeZ True --dataset /hpcfs/juno/junogpu/fangwx/FastSim/BES/data/mc_Hit_Barrel_em_zcut_Low.h5 --nb-epochs 100 --batch-size 128 --latent-size 512 --reg-model-in '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/reco/mc_em_Low_1008.h5' --gen-out '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/gen_model_em_Low_1025v1.h5' --comb-out '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/comb_model_em_Low_1025v1.h5' --dis-out '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/dis_model_em_Low_1025v1.h5'
#####/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/BES/models/train_v5.py --removeZ True --dataset /hpcfs/juno/junogpu/fangwx/FastSim/BES/data/mc_Hit_Barrel_em_zcut_Low.h5 --nb-epochs 500 --batch-size 128 --latent-size 512 --reg-model-in '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/reco/mc_em_Low_1025.h5' --gen-out '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/gen_model_em_Low_1027ep500.h5' --comb-out '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/comb_model_em_Low_1027ep500.h5' --dis-out '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/dis_model_em_Low_1027ep500.h5'
##########/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/BES/models/train_v7.py --dataset /hpcfs/juno/junogpu/fangwx/FastSim/BES/data/mc_Hit_Barrel_em_zcut_Low.h5 --nb-epochs 500 --batch-size 128 --latent-size 512 --reg-model-in '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/reco/mc_em_Low_1028.h5' --gen-out '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/gen_model_em_Low_1029add.h5' --comb-out '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/comb_model_em_Low_1029add.h5' --dis-out '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/dis_model_em_Low_1029add.h5'
#################/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/BES/models/train_v8.py --dataset /hpcfs/juno/junogpu/fangwx/FastSim/BES/data/mc_Hit_Barrel_em_zcut_Low.h5 --nb-epochs 500 --batch-size 128 --latent-size 512 --reg-model-in '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/reco/mc_em_Low_1028.h5' --gen-out '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/gen_model_em_Low_1030Add.h5' --comb-out '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/comb_model_em_Low_1030Add.h5' --dis-out '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/dis_model_em_Low_1030Add.h5'
/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/BES/models/train_v12.py --dataset /hpcfs/juno/junogpu/fangwx/FastSim/BES/data/mc_Hit_Barrel_em_zcut_Low.h5 --nb-epochs 200 --batch-size 128 --latent-size 512 --reg-model-in '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/reco/mc_em_Low_1106.h5' --gen-out '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/gen_model_em_Low_1107.h5' --comb-out '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/comb_model_em_Low_1107.h5' --dis-out '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/dis_model_em_Low_1107.h5'
