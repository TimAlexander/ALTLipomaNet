import torch
import torch.nn as nn

from .ResNet3D import generate_model

ResNets = ['resnet18','resnet34','resnet50','resnet101','resnet152']
DenseNets = ['densenet121','densenet161','densenet169','densenet201']
WideResNets = ['wide_resnet50_2','wide_resnet101_2']
ResNexts = ['resnext50_32x4d','resnext101_32x8d']
ResNets3d = ['resnet10_3d','resnet18_3d','resnet34_3d','resnet50_3d','resnet101_3d','resnet152_3d','resnet200_3d']

def load_model(model_name,hparams):
    
    ##DenseNets
    if model_name in DenseNets:
        
        model = torch.hub.load('pytorch/vision:v0.6.0', model_name, pretrained=hparams["pretrained"],drop_rate=hparams["drop_rate"])
        model.name = model_name

        return change_classifier(model,hparams["n_classes"])
     
    ##ResNets                                
    elif model_name in ResNets:

        model = torch.hub.load('pytorch/vision:v0.6.0', model_name, pretrained=hparams["pretrained"])
        model.name = model_name
                  
        return change_classifier(model,hparams["n_classes"])
    
    elif model_name in WideResNets:

        model = torch.hub.load('pytorch/vision:v0.6.0', model_name, pretrained=hparams["pretrained"])
        model.name = model_name     
                  
        return change_classifier(model,hparams["n_classes"])
    
    elif model_name in ResNexts:

        model = torch.hub.load('pytorch/vision:v0.6.0', model_name, pretrained=hparams["pretrained"])
        model.name = model_name     
                  
        return change_classifier(model,hparams["n_classes"])

    #Start of 3D Models
    elif model_name in ResNets3d:

        if model_name == "resnet10_3d":
             model = generate_model(10,n_input_channels=hparams["n_channels"],n_classes=hparams["n_classes"])
        elif model_name == "resnet18_3d":
             model = generate_model(18,n_input_channels=hparams["n_channels"],n_classes=hparams["n_classes"])
        elif model_name == "resnet34_3d":
             model = generate_model(34,n_input_channels=hparams["n_channels"],n_classes=hparams["n_classes"])
        elif model_name == "resnet50_3d":
             model = generate_model(50,n_input_channels=hparams["n_channels"],n_classes=hparams["n_classes"])
        elif model_name == "resnet101_3d":
             model = generate_model(101,n_input_channels=hparams["n_channels"],n_classes=hparams["n_classes"])
        elif model_name == "resnet152_3d":
             model = generate_model(152,n_input_channels=hparams["n_channels"],n_classes=hparams["n_classes"])
        elif model_name == "resnet200_3d":
             model = generate_model(200,n_input_channels=hparams["n_channels"],n_classes=hparams["n_classes"])

        model.name = model_name
        
        return model

    else:
        raise NameError('Model not found')
            
def change_classifier(model,n_classes):
    classifier_name, old_classifier = model._modules.popitem()
    new_classifier = nn.Linear(old_classifier.in_features,n_classes)
    model.add_module(classifier_name, new_classifier)
    
    return model

