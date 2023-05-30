import nibabel as nib
import numpy as np
import scipy.ndimage as nd
from dipy.align.reslice import reslice


def resample(img, nshape=None, spacing=None, new_spacing=None, order=0, mode='constant'):
    """
        Change image resolution by resampling

        Args:
        - spacing (numpy.ndarray): current resolution
        - new_spacing (numpy.ndarray): new resolution
        - order (int: 0-5): interpolation order

        Returns:
        - resampled image
        """
    if nshape is None:
        if spacing.shape[0]!=1:
            spacing = np.transpose(spacing)

        if new_spacing.shape[0]!=1:
            new_spacing = np.transpose(new_spacing)

        if np.array_equal(spacing, new_spacing):
            return img

        resize_factor = spacing / new_spacing
        new_real_shape = img.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / img.shape

    else:
        if img.shape == nshape:
            return img
        real_resize_factor = np.array(nshape, dtype=float) / np.array(img.shape, dtype=float)

    image = nd.zoom(img, real_resize_factor.ravel(), order=order, mode=mode)

    return image



def resample_nifti(volume: nib.Nifti1Image,new_resolution : np.array([0.5,0.5,0.5])) -> nib.Nifti1Image:
    """ 
    Resample a nifti image to a new resolution

    Args:
    - volume (nib.Nifti1Image): volume to be resampled
    - new_resolution (np.array): new resolution

    Returns:
    - resampled volume
    """

    volume_data = volume.get_fdata()
    affine = volume.affine
    zooms = volume.header.get_zooms()[:3]
    volume, resampled_affine = reslice(volume_data, affine, zooms, new_resolution)
    volume = resample(volume_data,spacing=np.array(zooms),new_spacing=new_resolution)
    return nib.Nifti1Image(volume,affine=resampled_affine)

def get_maximum_slice(scan : np.ndarray,segmentation:np.ndarray) -> dict:
    """
    Get the slice with the maximum area of the segmentation mask

    Args:
    - scan (numpy.ndarray): scan to be processed
    - segmentation (numpy.ndarray): segmentation mask

    Returns:
    - dictionary containing the maximum slice and the corresponding area
    """
    
    idx = np.argmax(np.sum(segmentation,axis=(0,1)))
    return {'scan':scan[:,:,idx],'segmentation':segmentation[:,:,idx]}

