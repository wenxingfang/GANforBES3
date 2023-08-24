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
#SBATCH --output=/hpcfs/juno/junogpu/fangwx/FastSim/BES/job_sub/jobs/testing_em_wgan.out
#SBATCH --error=/hpcfs/juno/junogpu/fangwx/FastSim/BES/job_sub/jobs/testing_em_wgan.err
  
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

/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python /hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/generator_wgan.py --latent-size 512 --nb-events 5000 --gen-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/gen_model_em_1105w.h5" --comb-model-in "/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/comb_model_em_1105w.h5"   --exact-model True --exact-list '/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/exact_input_em.txt'  --output "/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen1107_em_wgan.h5" --check-dis-real False --dis-model-in '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/dis_model_em_1105w.h5' --real-data '/hpcfs/juno/junogpu/fangwx/FastSim/BES/data/Z_merged/mc_Hit_Barrel_em_zcut.h5' --convert2pb False --path_pb "/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/" --name_pb "model_em_1029.pb" --SavedModel False 
