import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from data_loader.GetDataset_ISIC2018 import ISIC2018_dataset
# from data_loader.GetDataset_Retouch import MyDataset
from data_loader.GetDataset_CHASE import MyDataset_CHASE
from data_loader.GetDataset_AQUA import MyDataset
from model.SDCTrans import SDCTrans
import glob
import argparse
from torchvision import datasets, transforms
from solver import Solver
import torch.nn.functional as F
import torch.nn as nn
# from GetDataset import MyDataset
import cv2
from skimage.io import imread, imsave
import os

torch.cuda.set_device(0) ## GPU id

def parse_args():
    parser = argparse.ArgumentParser(description='SDCTrans Training With Pytorch')

    # dataset info
    parser.add_argument('--dataset', type=str, default='aqua',  
                        help='retouch, isic, chase')

    parser.add_argument('--data_root', type=str, default='full_data/train_data',  
                        help='dataset directory')
    parser.add_argument('--resize', type=int, default=[352,512], nargs='+',
                        help='image size: [height, width]')

    # network option & hyper-parameters
    parser.add_argument('--num-class', type=int, default=7, metavar='N',
                        help='number of classes for your data')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=80, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.00025, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--lr-update', type=str, default='poly',  
                        help='the lr update strategy: poly, step, warm-up-epoch, CosineAnnealingWarmRestarts')
    parser.add_argument('--lr-step', type=int, default=12,  
                        help='define only when you select step lr optimization: what is the step size for reducing your lr')
    parser.add_argument('--gamma', type=float, default=0.5,  
                        help='define only when you select step lr optimization: what is the annealing rate for reducing your lr (lr = lr*gamma)')

    parser.add_argument('--use_SDL', action='store_true', default=False,
                        help='set as True if use SDL loss; only for Retouch dataset. If you use it with other dataset please define your own path of label distribution in solver.py')
    parser.add_argument('--folds', type=int, default=4,
                        help='define folds number K for K-fold validation')

    # checkpoint and log
    parser.add_argument('--pretrained', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--weights', type=str, default=None,
                        help='path of SDL weights')
    parser.add_argument('--save', default='save',
                        help='Directory for saving checkpoint models')

    parser.add_argument('--save-per-epochs', type=int, default=100,
                        help='per epochs to save')

                        
    # evaluation only
    parser.add_argument('--test_only', action='store_true', default=True,
                        help='test without training')
    args = parser.parse_args()

    if not os.path.isdir(args.save):
        os.makedirs(args.save)

    return args



def main(args):
    
    modality = 'W'
    # root_p = '/home/ziyun/Desktop/project/MK project/segmentation_code/full_data/train_data'
    modality_dict={'B':args.data_root+'/Blue_light/', 'W':args.data_root+'/White_light/', 'ScS':args.data_root+'/Scs/'}
    root = modality_dict[modality]

    patient_id = []
    pat_ls = glob.glob(root+'*')
    for pat in pat_ls:
        pat_id = pat.split('/')[-1]
        patient_id.append(pat_id)
    # print(patient_id)

    total_pat_num = len(patient_id)
    print(total_pat_num)
    pat_per_fold = total_pat_num//4 + 1
    print(pat_per_fold)

    ## K-fold cross validation ##
    for exp_id in range(4): #args.folds

        if args.dataset == 'isic':
            trainset = ISIC2018_dataset(dataset_folder=args.data_root, folder=exp_id+1, train_type='train', 
                                           with_name=False)
            validset = ISIC2018_dataset(dataset_folder=args.data_root, folder=exp_id+1, train_type='test',
                                               with_name=False)
        elif args.dataset == 'retouch':   
            pat_ls = glob.glob(args.data_root+'/*')
            total_id = [i for i in range(24)]
            test_id = [i for i in range(exp_id*8,(exp_id+1)*8)]
            # train_root = pat_ls - test_root
            train_id = set(total_id) - set(test_id)
            test_root = [pat_ls[i] for i in test_id]
            train_root = [pat_ls[i] for i in train_id]
            # print(train_root)

            trainset = MyDataset(args,train_root = train_root,mode='train')
            validset = MyDataset(args,train_root = test_root,mode='test')
            print("Train size: %i" % len(train_data))
            print("Test size: %i" % len(test_data))

        elif args.dataset == 'chase':
            overall_id = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14']
            test_id = overall_id[3*exp_id:3*(exp_id+1)]
            train_id = list(set(overall_id)-set(test_id))
            # print(train_id)
            trainset = MyDataset_CHASE(args,train_root = args.data_root,pat_ls=train_id,mode='train')
            validset = MyDataset_CHASE(args,train_root = args.data_root,pat_ls=test_id,mode='test')

        elif args.dataset == 'aqua':
            test_id = patient_id[exp_id*pat_per_fold:(exp_id+1)*pat_per_fold]
            # train_root = pat_ls - test_root
            train_id = list(set(patient_id) - set(test_id))
            print(train_id)
            # print(len(train_id))

            # print(train_root)

            trainset = MyDataset(args=args, train_list = train_id,mode='train',root=root)
            validset = MyDataset(args=args,train_list = test_id,mode='test',root=root)
            # print("Train size: %i" % len(train_data))
            # print("Test size: %i" % len(test_data))


            ####  define how you get the data on your own dataset ######
            pass
        else:
            pass

        train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=6)
        val_loader = torch.utils.data.DataLoader(dataset=validset, batch_size=1, shuffle=False, pin_memory=True, num_workers=2)
        
        print("Train batch number: %i" % len(train_loader))
        print("Test batch number: %i" % len(val_loader))

        #### Above: define how you get the data on your own dataset ######
        model = SDCTrans(num_class=args.num_class).cuda()

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(params)

        if args.pretrained:
            model.load_state_dict(torch.load(args.pretrained,map_location = torch.device('cpu')))
            model = model.cuda()

        solver = Solver(args)

        solver.train(model, train_loader, val_loader,exp_id+1, num_epochs=args.epochs)

if __name__ == '__main__':
    args = parse_args()
    main(args)
    
