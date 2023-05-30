from os import listdir
from os.path import join

import monai
import nibabel as nib
import numpy as np
import torch
from skimage import transform
from skimage.exposure import rescale_intensity
from torch.utils.data import Dataset
from torchvision import transforms

from .Custom_Transforms import (
    CenterCrop3d,
    CornerAndCenterCrop3d,
    RandomCrop3d,
    Resize_Volume_Keeping_AR,
    TumorCrop2d,
    TumorCrop3d,
)

#import torchio as tio


class Raw_Dataset(Dataset):
    """Get raw data in NIFTI format"""
    
    def __init__(self,root_dir,modality="T1",return_path=False):
        
        self.root_dir = root_dir
        self.volume_paths = []
        self.segmentation_paths = []
        self.volume_names = []
        self.segmentation_names = []
        self.return_path = return_path
        for patient in listdir(root_dir):
            try:
                volume_name = listdir(join(root_dir,patient+"/{}/Scan".format(modality)))[0]
                segm_name = listdir(join(root_dir,patient+"/{}/Segmentation".format(modality)))[0]
                self.volume_names.append(volume_name)
                self.segmentation_names.append(segm_name)
                self.volume_paths.append(patient+"/{}/Scan/".format(modality)+volume_name)
                self.segmentation_paths.append(patient+"/{}/Segmentation/".format(modality)+segm_name)
            except Exception as e:
                print(e)
                print(patient)
                
    def __len__(self):
        
        return len(self.volume_names)
        
    def __getitem__(self,idx):
        
        full_volume_path = join(self.root_dir,self.volume_paths[idx])
        volume = nib.load(full_volume_path)
    
        
        full_segmentation_path = join(self.root_dir,self.segmentation_paths[idx])
        segmentation = nib.load(full_segmentation_path)
        
        if self.return_path:
            return volume,self.volume_paths[idx],segmentation,self.segmentation_paths[idx]
        
        return volume,segmentation
        


""" 
2D Dataset
"""
class ClassificationDataset2d(Dataset):
    
        def __init__(self,root_dir,patients,targets,transformations=[],
                     image_size=[224,224],image_net=True,crop_tumor=True,tumor_padding=10,return_patient=False):

            self.root_dir = root_dir
            self.patients = patients
            self.targets = targets
            self.transformations = transformations
            self.image_size = image_size
            self.image_net = image_net
            self.crop_tumor = crop_tumor
            self.tumor_padding = tumor_padding
            self.return_patient = return_patient
            
            
        def __len__(self):
        
            return len(self.patients)
        
        
        def __getitem__(self,idx):
            
            #Load Image, assumes npy format
            image = np.load(join(self.root_dir,self.patients[idx])+".npy").astype(np.float32)
            
          
            #Check if ROI cropping should be done
            if self.crop_tumor:
                tumor_crop = TumorCrop2d(tumor_padding=self.tumor_padding) #Initialize ROI cropping object
                image = tumor_crop((image,self.get_segmentation(idx))) #Crop ROI. For this we need to also load the segmentation.
                
                
            #keep aspect ratio if != 1    
            h,w = image.shape    
            target_h,target_w = self.image_size
            resize_size = self.image_size

            if h != w:
                if  h>w:
                    h = int(target_w/w * h)
                    w = target_w
                elif h<w:
                    w = int(target_h/h * w)
                    h = target_h 
            
                resize_size = (h,w)
        
            #Resize Image
            image = transform.resize(image,resize_size,order=3,anti_aliasing=True)

            if len(image.shape)==2:
                image = np.expand_dims(image,0) #Add channel dimension
                    
            if self.transformations:    
                #Transform    
                size = np.random.randint(0,len(self.transformations)+1)
                #Sample randomly from the possible augmentations
                selected_transforms = np.random.choice(self.transformations,size=size,replace=False)
                if len(selected_transforms) != 0:
                    #If augmentations were selected, compose them and apply them to the image
                    selected_transforms = monai.transforms.Compose(selected_transforms)
                    image = selected_transforms(image)
                
                #Else : No augmentation selected

            if self.image_net:
                self.image_size = 224,224
            #Cropping
            if (image.shape[1] != self.image_size[0]) or (image.shape[2] != self.image_size[1]):
                image = monai.transforms.RandSpatialCrop(roi_size=(self.image_size[0],self.image_size[1]),random_size=False)(image)
                

            #Transformations to be done if we want to use a on ImageNet pretrained network
            if self.image_net:

                image = rescale_intensity(image, out_range=(0.0, 1.0))

                if image.shape[0] == 1:
                    image = np.squeeze(image,0)
                    image = np.stack([image, image, image], axis=0)
                #else assume 3 channels

                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]

                image[0, :, :] = (image[0, :, :] - mean[0]) / std[0]
                image[1, :, :] = (image[1, :, :] - mean[1]) / std[1]
                image[2, :, :] = (image[2, :, :] - mean[2]) / std[2]
            
            image = torch.from_numpy(image)
                
           
            output = {}
            output['Image'] = image.float()
            output['Target'] = self.targets[idx] 

            if self.return_patient:
                output['Patient'] = self.patients[idx]
            
            return output
        
        def get_segmentation(self,idx):
                
            seg = np.load(join(self.root_dir,self.patients[idx])+"-seg.npy").astype(np.int16)
            
            return seg
        
        

        
"""
3D Dataset
"""
class ClassificationDataset3d(Dataset):
    
        def __init__(self,root_dir,patients,targets,crop_tumor=True,tumor_padding=5,cropping_mode="random_crop",min_size=100,volume_size=(100,100,100),
                     transformations=[],return_patient=False):

            self.root_dir = root_dir
            self.patients = patients
            self.targets = targets
            self.transformations = transformations
            self.min_size = min_size
            self.volume_size = volume_size
            self.crop_tumor = crop_tumor
            self.tumor_padding = tumor_padding
            self.cropping_mode = cropping_mode
            self.return_patient = return_patient

            #The size selected for cropping has to be greater or equal of the minimum resizing size
            assert volume_size[0] <= min_size
            assert volume_size[1] <= min_size
            assert volume_size[2] <= min_size
                
                
        def __len__(self):
        
            return len(self.patients)
        
        def __getitem__(self,idx):
            
            
            
            #Load Image, assumes npy format
            volume = np.load(join(self.root_dir,self.patients[idx])+".npy").astype(np.float32)
            
          
            #Check if ROI cropping should be done
            #In contrast to 2D cropping the resulting crop won´t have an aspect ratio of 1 !!
            if self.crop_tumor:
                tumor_crop = TumorCrop3d(tumor_padding=self.tumor_padding) #Initialize ROI cropping object
                volume = tumor_crop((volume,self.get_segmentation(idx))) #Crop ROI. For this we need to also load the segmentation.
            

            #Resize the volume keeping it aspect ratio. Min resizing size should be >= cropping size
            resize_obj = Resize_Volume_Keeping_AR(self.min_size)
            volume = resize_obj(volume)
            

            #Choose transformations
            #Because
            size = np.random.randint(0,len(self.transformations)+1)
            selected_transforms = np.random.choice(self.transformations,size=size,replace=False)
            selected_transforms = monai.transforms.Compose(selected_transforms)
            
            
            #Different cropping modes for the final input
            #Random cropping -> Training
            #Center cropping -> Validation
            #Center+Corner cropping -> Testing
            h,w,d = self.volume_size
            if self.cropping_mode == "random_crop":
                volume = RandomCrop3d(size=(h,w,d))(volume)
                volume = np.expand_dims(volume,0)
                if selected_transforms:
                        volume = selected_transforms(volume)
                volume = torch.from_numpy(volume)
                target = self.targets[idx]
            if self.cropping_mode == "center_crop":
                volume = CenterCrop3d(size=(h,w,d))(volume)
                volume = np.expand_dims(volume,0)
                if selected_transforms:
                        volume = selected_transforms(volume)
                volume = torch.from_numpy(volume)
                target = self.targets[idx]
            elif self.cropping_mode == "corner_and_center_crop":
                volumes = []
                crops = CornerAndCenterCrop3d(size=(h,w,d))(volume)
                for crop in crops:
                    crop = np.expand_dims(crop,0)
                    if selected_transforms:
                        crop = selected_transforms(crop)         
                    crop = torch.from_numpy(crop)
                    volumes.append(crop)
                volume = torch.stack(volumes)
                target = np.repeat(self.targets[idx],len(crops))
                

                
            output = {}
            output['Volume'] = volume.float()
            output['Target'] = target

            if self.return_patient:
                output['Patient'] = self.patients[idx]
            
            return output
        
        def get_segmentation(self,idx):
                
            seg = np.load(join(self.root_dir,self.patients[idx])+"-seg.npy").astype(np.int16)
            
            return seg
        
