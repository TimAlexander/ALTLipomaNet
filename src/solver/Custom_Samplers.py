import numpy as np
import torch
import torch.utils.data


class Curriculum_sampler(torch.utils.data.sampler.Sampler):
    
    #Creates sampler who orders batches based on difficulty for curriculum learning
    
    
    def __init__(self,patients,sorted_lip,sorted_alt):
        self.sorted_lip = np.asarray(sorted_lip)
        self.sorted_alt = np.asarray(sorted_alt)
        self.patient_idx_dict = {patient:i for i,patient in enumerate(patients)}
        
        self.new_len = 2*len(self.sorted_lip) if len(self.sorted_lip) > len(self.sorted_alt) else 2*len(self.sorted_alt)
        
        
    def __len__(self):
        return self.new_len

    def __iter__(self):
        
        return iter(self.sort(self.sorted_lip,self.sorted_alt))
   
    def sort(self,sorted_lip,sorted_alt):
        
        if len(sorted_lip) > len(sorted_alt):
            oversample = np.random.choice(sorted_alt,size=len(sorted_lip)-len(sorted_alt),replace=False)
            idx = [np.argwhere(sorted_alt==patient).flatten()[0] for patient in oversample]
            sorted_alt = np.insert(sorted_alt,idx,oversample)
        else:
            oversample = np.random.choice(sorted_lip,size=len(sorted_alt)-len(sorted_lip),replace=False)
            idx = [np.argwhere(sorted_lip==patient).flatten()[0] for patient in oversample]
            sorted_lip = np.insert(sorted_lip,idx,oversample)
            
        sorted_patients = [patient for patients in zip(sorted_lip, sorted_alt) for patient in patients]
            
        return [self.patient_idx_dict[patient] for patient in sorted_patients]
            
            
    
      
        