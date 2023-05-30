from __future__ import division, print_function

import glob
import os
import shutil
import sys
from argparse import ArgumentParser

import monai
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import RepeatedStratifiedKFold
from torch.nn import CrossEntropyLoss
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from data.Custom_Transforms import ElasticTransform
from model.model_zoo import load_model
from solver.Trainer import Trainer2d

random_state_torch = True
random_state_np = True

if random_state_torch:
    torch.manual_seed(0)
if random_state_np:
    np.random.seed(0)


    


def train(args):


    pretrained = False
    #If imagenet flag is true, ensure network is pretrained and 3 input channels
    if args.imagenet:
        pretrained = True
        args.n_channels = 3
   
    
    """
    Data Strategy
    """
    
    image_size = (256,256) # If > 224 then RandomCrop will be used to create 224
    crop_tumor = True #If cropping around the tumor is used
    tumor_padding = 10 #Padding for crop_tumor


                                                 
                                                    
    
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)
        


    """
    Input data and paths
    """

    #Read target csv
    targets = pd.read_csv(args.targetcsv)



    #Drop patients
    #root_path = '/home/tomovt/lipoma_project/datasets/{}/'.format(dataset)
    #for patient in os.listdir(root_path):
    #    if len(os.listdir(root_path+patient+"/"+modality+"/Scan/")) == 0 or patient in test_set:
    #        targets = targets[targets.ID != patient]


    rskf = RepeatedStratifiedKFold(n_splits=args.n_folds, n_repeats=args.n_repeats,random_state = 0)

    X,y = targets.ID.values,targets.Label.values

    repeatcont=0
    foldcont=0
    cont=0

    ###Metric logging
    acc = []
    auc = []
    f1 = []
    sensitivity = []
    specificity = []
    bacc = []

    for train_index, val_index in rskf.split(X, y):
        
            foldname= str(foldcont+1).zfill(3)
            repeatname= str(repeatcont+1).zfill(2)
            expfolder= os.path.join(args.outputdir, 'run_{}_fold_{}'.format(repeatname,foldname))
            if not os.path.exists(expfolder):
                os.makedirs(expfolder)
            else:
                shutil.rmtree(expfolder)
                os.makedirs(expfolder)

            foldcont += 1
            if foldcont != 0 and foldcont % args.n_folds == 0:
                foldcont = 0
                repeatcont += 1

            if cont<=-1:
                cont+=1
                continue
            else:
                cont += 1

            #Get training data
            #train_index, val_index = idxs

            train_patients,train_y = X[train_index],y[train_index]
            val_patients,val_y = X[val_index],y[val_index]

            data_files = {}
            data_files['root_path'] = args.datadir
            data_files['X_train'] = train_patients
            data_files['y_train'] = train_y
            data_files['X_val'] = val_patients
            data_files['y_val'] = val_y


            hparams = {"n_channels":args.n_channels,
                        "n_classes":args.n_classes,
                        "pretrained":pretrained,
                        "drop_rate":args.drop_rate}
            model = load_model(args.model_name,hparams)


            cuda_device = "cuda:0"
            device = torch.device(cuda_device)

            model_ft = model.to(device)
            loss = CrossEntropyLoss().to(device)



            """
            Optimizer and schedulers
            """
            if args.optimizer == "SGD":
                optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=args.lr,weight_decay=args.wd,momentum=0.9,nesterov=True)
            elif args.optimizer == "ADAMW":
                optimizer_ft = torch.optim.AdamW(model_ft.parameters(), lr=args.lr,weight_decay=args.wd)
            elif args.optimizer == "RMSprop":
                optimizer_ft = torch.optim.RMSprop(model_ft.parameters(),lr=args.lr,weight_decay=args.wd)
            elif args.optimizer == "ADAM":
                optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=args.lr,weight_decay=args.wd)
            else:
                optimizer_ft = None

            if args.lr_schedule == "CosineAnnealing":
                exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=args.n_epochs, eta_min=1e-8)
            elif args.lr_schedule == "ReduceLROnPlateau":
                exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, 'min',factor=0.5,patience=10,min_lr=1e-8)
            elif args.lr_schedule == "SGDR":
                exp_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer_ft, 20, T_mult=2, eta_min=1e-7, last_epoch=-1)
            else:
                exp_lr_scheduler = None


            """
            Augmentation & data
            """
            transformations = [monai.transforms.RandFlip(prob=1,spatial_axis=0),
                                monai.transforms.RandFlip(prob=1,spatial_axis=1),
                                monai.transforms.RandRotate(30,prob=1,padding_mode="zeros"),
                                monai.transforms.RandZoom(prob=1,min_zoom=1,max_zoom=1.1,mode='bilinear',align_corners=False),
                                monai.transforms.RandGaussianNoise(prob=1,std=0.2),
                                ElasticTransform(alpha=9,sigma=3,random_state=np.random.seed(0))] 



            trainer_instance = Trainer2d(data_files=data_files,
                                        model=model_ft,
                                        cost_function=loss,
                                        optimizer=optimizer_ft,
                                        schedulers=exp_lr_scheduler,
                                        transformations=transformations,
                                        stop_epoch=args.patience)



            trainer_instance.train(num_epochs=args.n_epochs,
                                   log_dir=expfolder,
                                   eval_rate=1,
                                   display_rate=10,
                                   batch_size=args.batch_size,
                                   image_size=image_size,
                                   crop_tumor=crop_tumor,
                                   tumor_padding=tumor_padding,
                                   image_net = args.imagenet,
                                   device=cuda_device,
                                   num_classes=args.n_classes)


            acc.append(trainer_instance.best_acc)
            auc.append(trainer_instance.best_auc)
            f1.append(trainer_instance.best_f1)
            sensitivity.append(trainer_instance.best_sensitivity)
            specificity.append(trainer_instance.best_specificity)
            bacc.append((trainer_instance.best_sensitivity+trainer_instance.best_specificity)/2)

    mean_auc = round(np.mean(auc),4)
    std_auc = round(np.std(auc),4)

    mean_f1 = round(np.mean(f1),4)
    std_f1 = round(np.std(f1),4)

    mean_sensitvity = round(np.mean(sensitivity),4)
    std_sensitivity = round(np.std(sensitivity),4)

    mean_specificity = round(np.mean(specificity),4)
    std_specificity = round(np.std(specificity),4)
    
    mean_bacc = round(np.mean(bacc),4)
    std_bacc = round(np.std(bacc),4)


    metrics_df = pd.DataFrame({"Path":args.outputdir,
                                "Mean_BAcc":mean_bacc,
                                "Std_BAcc":std_bacc,
                                "Mean_AUC":mean_auc,
                                "Std_AUC":std_auc,
                                "Mean_F1":mean_f1,
                                "Std_F1":std_f1,
                                "Mean_Sensitivity":mean_sensitvity,
                                "Std_Sensitivity":std_sensitivity,
                                "Mean_Specificity":mean_specificity,
                                "Std_Specificity":std_specificity},
                                index=[0])
    
    metrics_df.to_csv(args.outputdir+"metrics_overview.csv",index=False)

                

if __name__ == "__main__":
    
    parser = ArgumentParser()
    
    #Arguments
    #Architecture choices
    parser.add_argument('--model_name',type=str,default='densenet201',help ='unet | tba')
    parser.add_argument('--optimizer',type=str,default='ADAMW')
    parser.add_argument('--lr_schedule',type=str,default='CosineAnnealing')
    parser.add_argument('--n_channels',type=int,default=3)
    parser.add_argument('--n_classes',type=int,default=2)
    parser.add_argument('--imagenet',type=bool,default=True)

    #HyperParameters
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--wd',type=float,default=0.001)
    parser.add_argument('--drop_rate',type=float,default=0.)
    parser.add_argument('--batch_size',type=int,default=16)

    #Training Arguments
    parser.add_argument('--n_folds',type=int,default=3)
    parser.add_argument('--n_repeats',type=int,default=3)
    parser.add_argument('--n_epochs',type=int,default=100)
    parser.add_argument('--patience',type=int,default=10)

    #Data Arguments
    parser.add_argument('--datadir',type=str,default='/home/data/')
    parser.add_argument('--targetcsv',type=str,default='/home/target.csv')
    parser.add_argument('--outputdir',type=str,default='/home/output/')

    args = parser.parse_args()

    train(args)
    
    torch.cuda.empty_cache()

    