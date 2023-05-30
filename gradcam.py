import argparse
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tqdm import tqdm

from data.Dataset import ClassificationDataset2d
from model.model_zoo import load_model


def get_gradcampp(cam: GradCAMPlusPlus, image: torch.Tensor, target: ClassifierOutputTarget) -> dict:
    """
    Get GradCAM++ for a given image and target

    Args:
    - cam (GradCAMPlusPlus): GradCAM++ object
    - image (torch.Tensor): image
    - target (ClassifierOutputTarget): target

    Returns:
    - gradcampp (torch.Tensor): GradCAM++
    - overlay (numpy.ndarray): overlay
    """

    grayscale_cam = cam(input_tensor=image, targets=target, aug_smooth=False, eigen_smooth=True)
    grayscale_cam = grayscale_cam[0, :]
    img = image[0, :, :, :].numpy().transpose(1, 2, 0)
    # rescale to 0-255'
    img = img - img.min()
    img = img / img.max()

    overlay = show_cam_on_image(
        img, grayscale_cam, use_rgb=True, image_weight=0.5, colormap=cv2.COLORMAP_JET)

    return {'gradcampp': grayscale_cam, 'overlay': overlay}


def get_model(model_path: str, params_path: str) -> torch.nn.Module:
    """
    Load a model from a given path and parameters


    Args:
    - model_path (str): path to the model
    - params_path (str): path to the parameters

    Returns:
    - model (torch.nn.Module): model
    """

    with open(params_path) as f:
        hparams = json.load(f)

    hparams["n_classes"] = hparams["num_classes"]
    model = load_model(hparams["model"], hparams)
    model.load_state_dict(torch.load(
        model_path, map_location=torch.device('cpu'))['model_state_dict'])
    model.to('cpu')
    model.eval()

    return model


def main(args):

    # Load Data
    meta_data = pd.read_csv(args.meta_data_path, delimiter=';')
    patients = meta_data.iloc[:, 0].values
    targets = meta_data.iloc[:, 1].values
    targets[targets == 2] = 0 #rename class 2 to 0

    dataset = ClassificationDataset2d(root_dir=args.data_path, patients=patients, targets=targets, image_size=[
                                      224, 224], image_net=True, crop_tumor=True, tumor_padding=10, return_patient=True)

    # Load Model
    model = get_model(args.model_path, args.params_path)

    # Load GradCAMPLusPlus
    target_layers = [model.features[-1]]
    cam = GradCAMPlusPlus(
        model=model, target_layers=target_layers, use_cuda=False)

    for obj in tqdm(dataset, desc='Generating GradCAM++', leave=True):
        image = obj['Image']
        target = obj['Target']
        patient = obj['Patient']
        gradcampp_obj = get_gradcampp(
            cam, image.unsqueeze(0), [ClassifierOutputTarget(target)])
        gradcampp = gradcampp_obj['gradcampp']
        overlay = gradcampp_obj['overlay']

        # Save image
        plt.imshow(image[0, :, :].numpy(), cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(args.save_path, patient +
                    '_image.png'), bbox_inches='tight', pad_inches=0)
        plt.close()

        # Save GradCAM++ as heatmap
        plt.imshow(gradcampp, cmap='jet')
        plt.axis('off')
        plt.savefig(os.path.join(args.save_path, patient +
                    '_heatmap.png'), bbox_inches='tight', pad_inches=0)
        plt.close()

        # Save Overlay
        plt.imshow(overlay)
        plt.axis('off')
        plt.savefig(os.path.join(args.save_path, patient +
                    '_overlay.png'), bbox_inches='tight', pad_inches=0)
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Grad-CAM')
    parser.add_argument('--data_path', type=str,
                        help='Path to the data folder assuming structure data_path/patient/modality/Scan/Scan.nii.gz')
    parser.add_argument('--meta_data_path', type=str,
                        help='Path to the meta data file')
    parser.add_argument('--save_path', type=str,
                        help='Path to the save folder')
    parser.add_argument('--model_path', type=str,    help='Path to the model')
    parser.add_argument('--params_path', type=str,
                        help='Path to the model parameters')

    main(parser.parse_args())
