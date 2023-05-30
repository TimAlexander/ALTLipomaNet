from __future__ import division, print_function

import logging
import os
import shutil
from datetime import datetime

import numpy as np

#Generic Libaries
import torch
from sklearn.metrics import accuracy_score, auc, f1_score, recall_score, roc_curve

#Data
from data.Data_Modules import ClassificationDataModule2d, ClassificationDataModule3d

from .Custom_Samplers import Curriculum_sampler

#Solver
from .Earlystop import EarlyStopping


class Trainer2d(object):
    """
     ToDo

    """

    def __init__(self, data_files, model, cost_function, optimizer, schedulers, transformations, stop_epoch=100):
        
        #Data
        self.data_files = data_files
        self.transformations = transformations

        #Model + Solver
        self.model = model
        self.optimizer = optimizer
        self.cost_function = cost_function
        self.schedulers = schedulers

        #Metrics
        self.best_acc = 0
        self.best_loss = 1e6
        self.best_auc = 0
        self.best_sensitivity = 0
        self.best_specificity = 0
        self.best_f1 = 0
        self.epoch = 0
        self.global_step = 0
        self.stop_epoch = stop_epoch
        self.best_model = ''


    def _initialize(self):


        if not os.path.exists(self.log_dir):
            logging.info("Allocating '{:}'".format(self.log_dir))
            os.makedirs(self.log_dir)

        self.train_folder = os.path.join(self.log_dir, 'train')
        if not os.path.exists(self.train_folder):
            logging.info("Allocating '{:}'".format(self.train_folder))
            os.makedirs(self.train_folder)

        self.val_folder = os.path.join(self.log_dir, 'val')
        if not os.path.exists(self.val_folder):
            logging.info("Allocating '{:}'".format(self.val_folder))
            os.makedirs(self.val_folder)

        self.model_folder = os.path.join(self.log_dir, 'models')
        if not os.path.exists(self.model_folder):
            logging.info("Allocating '{:}'".format(self.model_folder))
            os.makedirs(self.model_folder)

        # initialize the early_stopping object
        self.early_stopping = EarlyStopping(patience=self.stop_epoch, verbose=True)


    def train(self, num_epochs, log_dir, eval_rate=1, display_rate=100, batch_size=8,
              image_size=224, num_classes=2, device='cuda:1',image_net=True,crop_tumor=True,tumor_padding=10):

        self.since = datetime.now()
        self.num_epochs = num_epochs
        self.log_dir = log_dir
        self.eval_rate = eval_rate
        self.display_rate = display_rate
        self.num_classes = num_classes
        self.image_size = image_size
        self._initialize()

        ## create data loaders
        data_module = ClassificationDataModule2d(self.data_files['root_path'],
                                                self.data_files['X_train'],
                                                self.data_files['y_train'],
                                                self.data_files['X_val'],
                                                self.data_files['y_val'],
                                                batch_size=batch_size,
                                                image_net=image_net,
                                                image_size=image_size,
                                                transformations=self.transformations,
                                                crop_tumor=crop_tumor,
                                                tumor_padding=tumor_padding)
                                                 
        data_module.prepare_data()
            

        self.dataloaders = {}
        self.dataloaders['train'] = data_module.train_dataloader()
        self.dataloaders['val'] = data_module.val_dataloader()
        self.device = torch.device(device)

        self.end_training = False
        self.current_iteration = 0

        #Initi global weights for Loss Function in case a batch has just sampled from one class
        self.train_weights = self.median_frequency(data_module.dataset["train"].targets)
        self.val_weights = self.median_frequency(data_module.dataset["val"].targets)


        
        for epoch in range(self.num_epochs):
            print("\n==== Epoch [ %d  /  %d ] START ====" % (epoch, self.num_epochs))
            # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 20)
            self.phase = 'train'
            print("<<<= Phase: %s =>>>" % self.phase)
            self.train_model(self.phase, epoch)
            if epoch % self.eval_rate == 0:
                self.phase = 'val'
                self.evaluate_model(self.phase, epoch)
            ## logging epoch
            print("==== Epoch [" + str(epoch) + " / " + str(self.num_epochs) + "] DONE ====")
            if self.end_training:
                break
                
        time_elapsed = datetime.now() - self.since
        print('Training complete in {}'.format(time_elapsed))
        print('Best val Acc: {:4f}'.format(self.best_acc))
        print('Best model: {}'.format(self.best_model))

    def train_model(self, phase, epoch):
        self.model.train()

        sum_loss_train = []
        gt=[]
        preds=[]
        probas= []



        for i_batch, sample_batched in enumerate(self.dataloaders[phase]):

            
            x = sample_batched['Image']
            y = sample_batched['Target']
              
            
            #Calculate Class weights for Loss Function
            w = self.median_frequency(y).to(self.device)
            self.cost_function.weight = w

            #Forward pass
            logits = self.model(x.to(self.device))
            #Calculate Loss
            loss = self.cost_function(logits, y.type(torch.LongTensor).to(self.device))
            #Ensure gradients are reset & then do backward pass & gradient update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.current_iteration += 1
            sum_loss_train.append(loss.item())
            
            with torch.no_grad():
                _, predicted = torch.max(torch.softmax(logits, dim=1), 1)
                
                gt.extend(y.numpy())
                preds.extend(predicted.cpu().squeeze().detach().numpy())
                probas.extend(torch.softmax(logits, dim=1).cpu().numpy())
                
                #Metrics
                acc = accuracy_score(y.cpu().numpy(), predicted.cpu().numpy())
                auc= self.compute_auc_binary(y.numpy(), torch.softmax(logits, dim=1).cpu().numpy())
                f1score= f1_score(y.cpu().numpy(), predicted.cpu().numpy(), average='macro')
                sensitivity = recall_score(y.cpu().numpy(), predicted.cpu().numpy(),pos_label=1)
                specificity = recall_score(y.cpu().numpy(), predicted.cpu().numpy(),pos_label=0)
                bacc = (sensitivity+specificity)/2
                

                if i_batch % self.display_rate == 0:
                    print('[Iteration : ' + str(i_batch) + '] Loss -> ' + str(loss.item()) + '  BAcc -> ' + str(
                        bacc) + '  auc -> ' + str(auc) + ' f1score -> ' + str(f1score))

                del x, y, loss, acc, logits, predicted
                torch.cuda.empty_cache()

                
        epoch_loss = np.mean(sum_loss_train)
        probas= np.asarray(probas)
        epoch_acc = accuracy_score(np.asarray(gt), np.asarray(preds))
        epoch_auc = self.compute_auc_binary(np.asarray(gt), probas)
        epoch_f1score = f1_score(np.asarray(gt), np.asarray(preds), average='macro')
        epoch_sensitivity = recall_score(np.asarray(gt), np.asarray(preds),pos_label=1)
        epoch_specificity = recall_score(np.asarray(gt), np.asarray(preds),pos_label=0)
        epoch_bacc = (epoch_sensitivity+epoch_specificity)/2            
                

        print('Phase: {}. Epoch {} Loss: {:.4f},  BAcc : {:.4f},  auc : {:.4f},  f1score : {:.4f},  specifcity : {:.4f},  sensitvity : {:.4f}'.format(
            phase,epoch, epoch_loss, epoch_bacc,epoch_auc, epoch_f1score,epoch_specificity,epoch_sensitivity))


    def evaluate_model(self, phase, epoch):
        self.model.eval()  # Set model to evaluate mode
        prefix = 'model'
        gt = []
        preds = []
        sum_loss_val = []
        probas =[]


        for i_batch, sample_batched in enumerate(self.dataloaders[phase]):            
            
            x = sample_batched['Image']
            y = sample_batched['Target']
              
            
            #Calculate Class weights for Loss Function
            w = self.median_frequency(y).to(self.device)
            self.cost_function.weight = w

            #Forward pass
            logits = self.model(x.to(self.device))
            #Calculate Loss
            loss = self.cost_function(logits, y.type(torch.LongTensor).to(self.device))

            self.current_iteration += 1
            sum_loss_val.append(loss.item())
            with torch.no_grad():
                _, predicted = torch.max(torch.softmax(logits, dim=1), 1)
                
                gt.extend(y.numpy())
                preds.extend(predicted.cpu().squeeze().detach().numpy())
                probas.extend(torch.softmax(logits, dim=1).cpu().numpy())
                
                #Metrics
                acc = accuracy_score(y.cpu().numpy(), predicted.cpu().numpy())
                auc= self.compute_auc_binary(y.numpy(), torch.softmax(logits, dim=1).cpu().numpy())
                f1score= f1_score(y.cpu().numpy(), predicted.cpu().numpy(), average='macro')
                sensitivity = recall_score(y.cpu().numpy(), predicted.cpu().numpy(),pos_label=1)
                specificity = recall_score(y.cpu().numpy(), predicted.cpu().numpy(),pos_label=0)
                bacc = (sensitivity+specificity)/2
                

                if i_batch % self.display_rate == 0:
                    print('[Iteration : ' + str(i_batch) + '] Loss -> ' + str(loss.item()) + '  BAcc -> ' + str(
                        bacc) + '  auc -> ' + str(auc) + ' f1score -> ' + str(f1score))

                del x, y, loss, acc, logits, predicted
                torch.cuda.empty_cache()

        epoch_loss = np.mean(sum_loss_val)
        
        #If LR Scheduler available follow the schedule
        if isinstance(self.schedulers,torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.schedulers.step(epoch_loss)
        else:
            self.schedulers.step()

        
        probas = np.asarray(probas)
        epoch_acc = accuracy_score(np.asarray(gt), np.asarray(preds))
        epoch_f1score = f1_score(np.asarray(gt), np.asarray(preds), average='macro')
        epoch_auc = self.compute_auc_binary(np.asarray(gt), probas)
        epoch_sensitivity = recall_score(np.asarray(gt), np.asarray(preds),pos_label=1)
        epoch_specificity = recall_score(np.asarray(gt), np.asarray(preds),pos_label=0)
        epoch_bacc = (epoch_sensitivity+epoch_specificity)/2
        

        print('Phase: {}. Epoch {} Loss: {:.4f},  BAcc : {:.4f},  auc : {:.4f},  f1score : {:.4f},  specifcity : {:.4f},  sensitvity : {:.4f}'.format(
            phase,epoch, epoch_loss, epoch_bacc,epoch_auc, epoch_f1score,epoch_specificity,epoch_sensitivity))


        epochname = str(epoch).zfill(3)
        auxaux = prefix + '_epoch_' + str(epochname) + '.tar'
        aux = prefix + '.tar'

        filename = os.path.join(self.model_folder, aux)

        if epoch_loss <= self.best_loss:
            self.best_model = auxaux
            self.best_loss = epoch_loss
            self.best_acc = epoch_acc
            self.best_auc = epoch_auc
            self.best_f1 = epoch_f1score            
            self.best_sensitivity = epoch_sensitivity
            self.best_specificity = epoch_specificity


        #Check EarlyStopping
        self.early_stopping(epoch_loss, self.model, self.optimizer, epoch, filename)
        if self.early_stopping.early_stop:
            self.end_training = True
            print("Early stopping")
            print(self.best_model)

    def save_checkpoint(self, val_loss, metric, model, optimizer, epoch, name):
        """
        saves the model if the evaluation metric increased
        :param val_loss:
        :param model:
        :param optimizer:
        :param epoch:
        :param name:
        :return:
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'metric': metric}, name)

    def median_frequency(self,y):
        unique, histo = np.unique(y, return_counts=True)
        if len(unique)==1:
            #Just one class in the batch->Use global dataset weight
            if self.phase == 'train':
                return self.train_weights
            elif self.phase == 'val':
                return self.val_weights
        else:
            freq = histo / np.sum(histo)
            med_freq = np.median(freq)
            weights = np.asarray(med_freq / freq)
            return torch.from_numpy(weights).float()


    def compute_auc_binary(self, y_test, y_predict_proba):
        fpr, tpr, _ = roc_curve(y_test, y_predict_proba[:, 1])
        aucval = auc(fpr, tpr)
        if np.isnan(aucval):
            aucval = 0
        return aucval
    


##Trainer Class for 3D##

class Trainer3d(object):
    """
        ToDo

    """

    def __init__(self, data_files, model, cost_function, optimizer, schedulers, transformations,curriculum_learning=False, stop_epoch=100):
        
        #Data
        self.data_files = data_files
        self.transformations = transformations

        #Model + Solver
        self.model = model
        self.optimizer = optimizer
        self.cost_function = cost_function
        self.schedulers = schedulers
        self.curriculum_learning = curriculum_learning

        #Metrics
        self.best_acc = 0
        self.best_loss = 1e6
        self.best_auc = 0
        self.best_sensitivity = 0
        self.best_specificity = 0
        self.best_f1 = 0
        self.epoch = 0
        self.global_step = 0
        self.stop_epoch = stop_epoch
        self.best_model = ''


    def _initialize(self):


        if not os.path.exists(self.log_dir):
            logging.info("Allocating '{:}'".format(self.log_dir))
            os.makedirs(self.log_dir)

        self.train_folder = os.path.join(self.log_dir, 'train')
        if not os.path.exists(self.train_folder):
            logging.info("Allocating '{:}'".format(self.train_folder))
            os.makedirs(self.train_folder)

        self.val_folder = os.path.join(self.log_dir, 'val')
        if not os.path.exists(self.val_folder):
            logging.info("Allocating '{:}'".format(self.val_folder))
            os.makedirs(self.val_folder)

        self.model_folder = os.path.join(self.log_dir, 'models')
        if not os.path.exists(self.model_folder):
            logging.info("Allocating '{:}'".format(self.model_folder))
            os.makedirs(self.model_folder)

        # initialize the early_stopping object
        self.early_stopping = EarlyStopping(patience=self.stop_epoch, verbose=True)


    def train(self, num_epochs, log_dir, eval_rate=1, display_rate=100, batch_size=8,
              volume_size=(100,100,100),min_size=100, num_classes=2, device='cuda:1',crop_tumor=True,tumor_padding=10):

        self.since = datetime.now()
        self.num_epochs = num_epochs
        self.log_dir = log_dir
        self.eval_rate = eval_rate
        self.display_rate = display_rate
        self.num_classes = num_classes
        self.volume_size = volume_size
        self.min_size = min_size
        self._initialize()

        ## create data loaders
        data_module = ClassificationDataModule3d(self.data_files['root_path'],
                                                self.data_files['X_train'],
                                                self.data_files['y_train'],
                                                self.data_files['X_val'],
                                                self.data_files['y_val'],
                                                batch_size=batch_size,
                                                min_size=min_size,
                                                volume_size=volume_size,
                                                transformations=self.transformations,
                                                crop_tumor=crop_tumor,
                                                tumor_padding=tumor_padding,
                                                return_patient=True)
                                                 
        data_module.prepare_data()
            

        self.dataloaders = {}
        self.dataloaders['train'] = data_module.train_dataloader()
        self.dataloaders['val'] = data_module.val_dataloader()
        self.device = torch.device(device)

        self.end_training = False
        self.current_iteration = 0

        #Initi global weights for Loss Function in case a batch has just sampled from one class
        self.train_weights = self.median_frequency(data_module.dataset["train"].targets)
        self.val_weights = self.median_frequency(data_module.dataset["val"].targets)

        if self.curriculum_learning:
            #Initizalize a sorting for both classes which is random/follows the directory structue
            lip_mask = np.argwhere(self.data_files["y_train"]==0).flatten()
            alt_mask = np.argwhere(self.data_files["y_train"]==1).flatten()
            self._sorted_lip = list(self.data_files["X_train"][lip_mask])
            self._sorted_alt = list(self.data_files["X_train"][alt_mask])


        
        for epoch in range(self.num_epochs):
            print("\n==== Epoch [ %d  /  %d ] START ====" % (epoch, self.num_epochs))
            # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 20)

            #Currciulum Learning
            if self.curriculum_learning:
                #Create Sampler which indicates the order of the batches
                 sampler=Curriculum_sampler(patients=self.data_files["X_train"], 
                                            sorted_lip=self._sorted_lip,
                                            sorted_alt=self._sorted_alt)
                #Assign sampler to training dataloader
                 self.dataloaders['train'] = data_module.train_dataloader(sampler=sampler)
                 print("Sorted ALT {}".format(self._sorted_alt))

            self.phase = 'train'
            print("<<<= Phase: %s =>>>" % self.phase)
            self.train_model(self.phase, epoch)
            if epoch % self.eval_rate == 0:
                self.phase = 'val'
                self.evaluate_model(self.phase, epoch)
            ## logging epoch
            print("==== Epoch [" + str(epoch) + " / " + str(self.num_epochs) + "] DONE ====")
            if self.end_training:
                break
                
        time_elapsed = datetime.now() - self.since
        print('Training complete in {}'.format(time_elapsed))
        print('Best val Acc: {:4f}'.format(self.best_acc))
        print('Best model: {}'.format(self.best_model))

    def train_model(self, phase, epoch):
        self.model.train()

        sum_loss_train = []
        gt=[]
        preds=[]
        probas= []
        patients = []



        for i_batch, sample_batched in enumerate(self.dataloaders[phase]):

            
            x = sample_batched['Volume']
            y = sample_batched['Target']
            patient = sample_batched['Patient']
              
            
            #Calculate Class weights for Loss Function
            w = self.median_frequency(y).to(self.device)
            self.cost_function.weight = w

            #Forward pass
            logits = self.model(x.to(self.device))
            #Calculate Loss
            loss = self.cost_function(logits, y.type(torch.LongTensor).to(self.device))
            #Ensure gradients are reset & then do backward pass & gradient update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.current_iteration += 1
            sum_loss_train.append(loss.item())
            
            with torch.no_grad():
                _, predicted = torch.max(torch.softmax(logits, dim=1), 1)
                
                gt.extend(y.numpy())
                preds.extend(predicted.cpu().squeeze().detach().numpy())
                probas.extend(torch.softmax(logits, dim=1).cpu().numpy())
                patients.extend(patient)
                
                #Metrics
                acc = accuracy_score(y.cpu().numpy(), predicted.cpu().numpy())
                auc= self.compute_auc_binary(y.numpy(), torch.softmax(logits, dim=1).cpu().numpy())
                f1score= f1_score(y.cpu().numpy(), predicted.cpu().numpy(), average='macro')
                sensitivity = recall_score(y.cpu().numpy(), predicted.cpu().numpy(),pos_label=1)
                specificity = recall_score(y.cpu().numpy(), predicted.cpu().numpy(),pos_label=0)
                bacc = (sensitivity+specificity)/2

                if i_batch % self.display_rate == 0:
                    print('[Iteration : ' + str(i_batch) + '] Loss -> ' + str(loss.item()) + '  BAcc -> ' + str(
                        bacc) + '  auc -> ' + str(auc) + ' f1score -> ' + str(f1score))

                del x, y, loss, acc, logits, predicted
                torch.cuda.empty_cache()


        if self.curriculum_learning:
            #Create ordering for both classes based on output probabilities
            lip_mask = np.argwhere(np.asarray(gt)==0).flatten()
            lip_dict = { pat : 1-prob for pat,prob in zip(np.asarray(patients)[lip_mask],np.asarray(probas)[lip_mask][:,0])}

            alt_mask = np.argwhere(np.asarray(gt)==1).flatten()
            alt_dict = { pat : 1-prob for pat,prob in zip(np.asarray(patients)[alt_mask],np.asarray(probas)[alt_mask][:,1])}

            self._sorted_lip = sorted(lip_dict.keys(), key=lip_dict.get)
            self._sorted_alt = sorted(alt_dict.keys(), key=alt_dict.get)

                
        epoch_loss = np.mean(sum_loss_train)
        probas= np.asarray(probas)
        epoch_acc = accuracy_score(np.asarray(gt), np.asarray(preds))
        epoch_auc = self.compute_auc_binary(np.asarray(gt), probas)
        epoch_f1score = f1_score(np.asarray(gt), np.asarray(preds), average='macro')
        epoch_sensitivity = recall_score(np.asarray(gt), np.asarray(preds),pos_label=1)
        epoch_specificity = recall_score(np.asarray(gt), np.asarray(preds),pos_label=0)
        epoch_bacc = (epoch_sensitivity+epoch_specificity)/2            
                

        print('Phase: {}. Epoch {} Loss: {:.4f},  BAcc : {:.4f},  auc : {:.4f},  f1score : {:.4f},  specifcity : {:.4f},  sensitvity : {:.4f}'.format(
            phase,epoch, epoch_loss, epoch_bacc,epoch_auc, epoch_f1score,epoch_specificity,epoch_sensitivity))


    def evaluate_model(self, phase, epoch):
        self.model.eval()  # Set model to evaluate mode
        prefix = 'model'
        gt = []
        preds = []
        sum_loss_val = []
        probas =[]


        for i_batch, sample_batched in enumerate(self.dataloaders[phase]):            
            
            x = sample_batched['Volume']
            y = sample_batched['Target']
              
            
            #Calculate Class weights for Loss Function
            w = self.median_frequency(y).to(self.device)
            self.cost_function.weight = w

            #Forward pass
            logits = self.model(x.to(self.device))
            #Calculate Loss
            loss = self.cost_function(logits, y.type(torch.LongTensor).to(self.device))

            self.current_iteration += 1
            sum_loss_val.append(loss.item())
            with torch.no_grad():
                _, predicted = torch.max(torch.softmax(logits, dim=1), 1)
                
                gt.extend(y.numpy())
                preds.extend(predicted.cpu().squeeze().detach().numpy())
                probas.extend(torch.softmax(logits, dim=1).cpu().numpy())
                
                #Metrics
                acc = accuracy_score(y.cpu().numpy(), predicted.cpu().numpy())
                auc= self.compute_auc_binary(y.numpy(), torch.softmax(logits, dim=1).cpu().numpy())
                f1score= f1_score(y.cpu().numpy(), predicted.cpu().numpy(), average='macro')
                sensitivity = recall_score(y.cpu().numpy(), predicted.cpu().numpy(),pos_label=1)
                specificity = recall_score(y.cpu().numpy(), predicted.cpu().numpy(),pos_label=0)
                bacc = (sensitivity+specificity)/2
                

                if i_batch % self.display_rate == 0:
                    print('[Iteration : ' + str(i_batch) + '] Loss -> ' + str(loss.item()) + '  BAcc -> ' + str(
                        bacc) + '  auc -> ' + str(auc) + ' f1score -> ' + str(f1score))

                del x, y, loss, acc, logits, predicted
                torch.cuda.empty_cache()

        epoch_loss = np.mean(sum_loss_val)
        
        #If LR Scheduler available follow the schedule
        if isinstance(self.schedulers,torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.schedulers.step(epoch_loss)
        else:
            self.schedulers.step()

        
        probas = np.asarray(probas)
        epoch_acc = accuracy_score(np.asarray(gt), np.asarray(preds))
        epoch_f1score = f1_score(np.asarray(gt), np.asarray(preds), average='macro')
        epoch_auc = self.compute_auc_binary(np.asarray(gt), probas)
        epoch_sensitivity = recall_score(np.asarray(gt), np.asarray(preds),pos_label=1)
        epoch_specificity = recall_score(np.asarray(gt), np.asarray(preds),pos_label=0)
        epoch_bacc = (epoch_sensitivity+epoch_specificity)/2
        

        print('Phase: {}. Epoch {} Loss: {:.4f},  BAcc : {:.4f},  auc : {:.4f},  f1score : {:.4f},  specifcity : {:.4f},  sensitvity : {:.4f}'.format(
            phase,epoch, epoch_loss, epoch_bacc,epoch_auc, epoch_f1score,epoch_specificity,epoch_sensitivity))


        epochname = str(epoch).zfill(3)
        auxaux = prefix + '_epoch_' + str(epochname) + '.tar'
        aux = prefix + '.tar'

        filename = os.path.join(self.model_folder, aux)

        if epoch_loss <= self.best_loss:
            self.best_model = auxaux
            self.best_loss = epoch_loss
            self.best_acc = epoch_acc
            self.best_auc = epoch_auc
            self.best_f1 = epoch_f1score            
            self.best_sensitivity = epoch_sensitivity
            self.best_specificity = epoch_specificity


        #Check EarlyStopping
        self.early_stopping(epoch_loss, self.model, self.optimizer, epoch, filename)
        if self.early_stopping.early_stop:
            self.end_training = True
            print("Early stopping")
            print(self.best_model)

    def save_checkpoint(self, val_loss, metric, model, optimizer, epoch, name):
        """
        saves the model if the evaluation metric increased
        :param val_loss:
        :param model:
        :param optimizer:
        :param epoch:
        :param name:
        :return:
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'metric': metric}, name)

    def median_frequency(self,y):
        unique, histo = np.unique(y, return_counts=True)
        if len(unique)==1:
            #Just one class in the batch->Use global dataset weight
            if self.phase == 'train':
                return self.train_weights
            elif self.phase == 'val':
                return self.val_weights
        else:
            freq = histo / np.sum(histo)
            med_freq = np.median(freq)
            weights = np.asarray(med_freq / freq)
            return torch.from_numpy(weights).float()


    def compute_auc_binary(self, y_test, y_predict_proba):
        fpr, tpr, _ = roc_curve(y_test, y_predict_proba[:, 1])
        aucval = auc(fpr, tpr)
        if np.isnan(aucval):
            aucval = 0
        return aucval