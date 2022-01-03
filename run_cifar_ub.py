import os

#ROOT='./options/train/miniImageNet/bases60_noBuffer/train_res18_miniImageNet_CFRPModel_AG_SGD'
# ROOT= './options/train/cifar/bases60_noBuffer/train_res18_cifar_CFRPModel_upperBound_SGD'
ROOT_1= './options/train/cifar/FSIL/FSIL_train_res18_cifar_CFRPModel_upperBound_SGD'
ROOT_2= './options/train/cifar/FSIL/FSIL_train_res18_cifar_CFRPModel_upperBound2_SGD'
#ID = ['01', '02']
# ID = ['05', '06']
ID = ['01', '02', '03', '04', '05', '06', '07', '08']
# ID = ['05', '06', '07', '08', '09']
# ID = ['02', '03', '04', '05']
# ID = ['06', '07', '08']
GPU = '1'

for id in ID:
    NAME = f'{ROOT_1}_{id}.yml'
    run = f'CUDA_VISIBLE_DEVICES={GPU} python train_upper_bound.py -opt {NAME}'
    print(f'Now run: {run} \n')
    os.system(run)

    NAME = f'{ROOT_2}_{id}.yml'
    run = f'CUDA_VISIBLE_DEVICES={GPU} python train_upper_bound.py -opt {NAME}'
    print(f'Now run: {run} \n')
    os.system(run)
    print('finished! \n')
