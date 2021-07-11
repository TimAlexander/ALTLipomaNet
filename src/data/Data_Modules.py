from random import shuffle
from .Dataset import ClassificationDataset2d,ClassificationDataset3d
from torch.utils.data import  DataLoader

    
class ClassificationDataModule2d(object):
    
    def __init__(self,root_path,train_patients,train_y,val_patients,val_y,batch_size=2,image_size=224,transformations=None,image_net=True,crop_tumor=True,tumor_padding=10
                 ,return_patient=False):
        super().__init__()
        
        self.root_path = root_path
        self.train_patients = train_patients
        self.train_y = train_y
        self.val_patients = val_patients
        self.val_y = val_y
        self.batch_size = batch_size
        self.return_patient = return_patient

        #Resizing operations
        self.crop_tumor = crop_tumor
        self.tumor_padding = tumor_padding
        self.image_size = image_size

        #Transformations
        self.image_net = image_net
        self.transformations = transformations
    
    def prepare_data(self):
        

        self.dataset = {}
        
        

        self.dataset["train"] =  ClassificationDataset2d(self.root_path,
                                                        self.train_patients,
                                                        self.train_y,
                                                        transformations=self.transformations,
                                                        image_size=self.image_size,
                                                        image_net=self.image_net,
                                                        crop_tumor=self.crop_tumor,
                                                        tumor_padding=self.tumor_padding,
                                                        return_patient=self.return_patient)

        self.dataset["val"] =  ClassificationDataset2d(self.root_path,
                                                        self.val_patients,
                                                        self.val_y,
                                                        transformations=[],
                                                        image_size=[224,224],
                                                        image_net=self.image_net,
                                                        crop_tumor=self.crop_tumor,
                                                        tumor_padding=self.tumor_padding,
                                                        return_patient=self.return_patient)


    def train_dataloader(self):            
        
        return DataLoader(self.dataset["train"],batch_size=self.batch_size,shuffle=True,num_workers=8)
        
    def val_dataloader(self):
        
        return DataLoader(self.dataset["val"],batch_size=self.batch_size,shuffle=False,num_workers=8)
        
        
class ClassificationDataModule3d(object):
    
    def __init__(self,root_path,train_patients,train_y,val_patients,val_y,batch_size=2,min_size=100,volume_size=(100,100,100),transformations=None,crop_tumor=True,tumor_padding=5,return_patient=False):
        super().__init__()
        
        self.root_path = root_path
        self.train_patients = train_patients
        self.train_y = train_y
        self.val_patients = val_patients
        self.val_y = val_y
        self.batch_size = batch_size
        self.return_patient = return_patient

        #Resizing operations
        self.crop_tumor = crop_tumor
        self.tumor_padding = tumor_padding
        self.volume_size = volume_size
        self.min_size = min_size

        #Transformations
        self.transformations = transformations
        

        
    def prepare_data(self):
        

        self.dataset = {}
        
        self.dataset["train"] = ClassificationDataset3d(self.root_path,
                                                        self.train_patients,
                                                        self.train_y,
                                                        transformations=self.transformations,
                                                        return_patient=self.return_patient,
                                                        volume_size=self.volume_size,
                                                        min_size=self.min_size,
                                                        crop_tumor=self.crop_tumor,
                                                        tumor_padding = self.tumor_padding,
                                                        cropping_mode='random_crop'
                                                        )

        self.dataset["val"] =   ClassificationDataset3d(self.root_path,
                                                        self.val_patients,
                                                        self.val_y,
                                                        transformations=[],
                                                        return_patient=self.return_patient,
                                                        volume_size=self.volume_size,
                                                        min_size=self.min_size,
                                                        crop_tumor=self.crop_tumor,
                                                        tumor_padding = self.tumor_padding,
                                                        cropping_mode='center_crop'
                                                        )
    

    def train_dataloader(self,sampler=None):   
        if sampler:
            shuffle = False   #In case we use the curriculum sampler we have to turn of shuffle    
        return DataLoader(self.dataset["train"],batch_size=self.batch_size,shuffle=False,sampler=sampler,num_workers=4)
        
    def val_dataloader(self):
        
        return DataLoader(self.dataset["val"],batch_size=self.batch_size,shuffle=False,num_workers=4)
    
        