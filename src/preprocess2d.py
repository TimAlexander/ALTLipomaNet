import argparse
import os

import numpy as np
from tqdm import tqdm

from data.Dataset import Raw_Dataset
from data.preprocessing import get_maximum_slice, resample_nifti


def main(args):

    # Create the dataset
    dataset = Raw_Dataset(
        args.data_path, modality=args.modality, return_path=True)

    # Iterate over the dataset
    for i in tqdm(range(len(dataset)), desc='Preprocessing images for 2D',leave=True):
        # Get the image and the segmentation
        image, image_path, segmentation, _ = dataset[i]
        # assume path  is data_path/patient/modality/Scan/Scan.nii.gz
        # extract patient name
        patient = image_path.split('/')[-4]

        # Resample the image and the segmentation
        print('Resampling image and segmentation for patient {}'.format(patient))
        old_resolution = np.array(image.header.get_zooms()[:3])
        old_dimension = np.array(image.shape)
        image = resample_nifti(image, new_resolution=np.array([
                               args.resolution, args.resolution, args.resolution]))
        segmentation = resample_nifti(segmentation, new_resolution=np.array([
                                      args.resolution, args.resolution, args.resolution]))
        new_dimension = np.array(image.shape)
        new_resolution = np.array(image.header.get_zooms()[:3])
        print('\tChanged resolution from {} to {}'.format(old_resolution, new_resolution))
        print('\tChanged dimension from {} to {}'.format(old_dimension, new_dimension))


        # Get the maximum slice
        image = image.get_fdata()
        segmentation = segmentation.get_fdata()
        obj = get_maximum_slice(image, segmentation)
        image = obj['scan']
        segmentation = obj['segmentation']

        # Save the image and the segmentation
        np.save(os.path.join(args.save_path, args.modality + '/' + patient+'.npy'), image)
        np.save(os.path.join(args.save_path, args.modality + '/' + patient+'-seg.npy'), segmentation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess 2D images')
    parser.add_argument('--data_path', type=str,
                        help='Path to the data folder assuming structure data_path/patient/modality/Scan/Scan.nii.gz')
    parser.add_argument('--save_path', type=str,
                        help='Path to the save folder')
    parser.add_argument('--resolution', type=float,
                        default=0.5, help='New resolution')
    parser.add_argument('--modality', type=str,
                        default='T1', help='Modality to be used')

    main(parser.parse_args())
