#python root2hdf5_v1.py --for-test True  --for-em True  --using-Z-cut True --using-Low-z True  --using-High-z False
#python root2hdf5_v1.py --for-test True  --for-em True  --using-Z-cut True --using-Low-z False --using-High-z True
#python root2hdf5_v1.py --for-test True  --for-em False --using-Z-cut True --using-Low-z True  --using-High-z False
#python root2hdf5_v1.py --for-test True  --for-em False --using-Z-cut True --using-Low-z False --using-High-z True

#python root2hdf5_v_single_e.py --for-test False  --for-em True --using-Z-cut True --using-Low-z True --using-High-z False

python root2hdf5_v3.py --for-test True  --for-em True  --using-Z-cut True 
python root2hdf5_v3.py --for-test False --for-em True  --using-Z-cut True 
#python root2hdf5_v3.py --for-test False --for-em False --using-Z-cut True 
#python root2hdf5_v3.py --for-test True  --for-em False --using-Z-cut True 


#python event_compare_args.py  --real_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/data/Z_merged/mc_Hit_Barrel_em_zcut.h5 --fake_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen1104_em_epoch19.h5 --event 5000 --tag epoch19
#python event_compare_args.py  --real_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/data/Z_merged/mc_Hit_Barrel_em_zcut.h5 --fake_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen1104_em_epoch29.h5 --event 5000 --tag epoch29
#python event_compare_args.py  --real_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/data/Z_merged/mc_Hit_Barrel_em_zcut.h5 --fake_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen1104_em_epoch39.h5 --event 5000 --tag epoch39
#python event_compare_args.py  --real_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/data/Z_merged/mc_Hit_Barrel_em_zcut.h5 --fake_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen1104_em_epoch49.h5 --event 5000 --tag epoch49

#python event_compare_args.py  --real_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/data/Z_merged/test_data_0/mc_Hit_Barrel_em_test_zcut.h5 --fake_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen1104_em_epoch115_ch.h5 --event 1000 --tag check

#python event_compare_args.py  --real_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/data/Z_merged/old_1/mc_Hit_Barrel_ep_zcut.h5 --fake_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen1104_ep_epoch129.h5 --event 5000 --tag ep_epoch129
#python event_compare_args.py  --real_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/data/Z_merged/old_1/mc_Hit_Barrel_em_zcut.h5 --fake_file /hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/Gen1104_em_epoch115.h5 --event 5000 --tag em_epoch115
