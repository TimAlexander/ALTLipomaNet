import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from skimage.transform import resize


class TumorCrop2d(object):
    """Crop an image to a given size at tumor location. The output will be a square image
       where the HxW is the length of the "longer" side of the tumor

    Args:
        tumor_padding: extra padding on HxW, if available (this means if the extra padding will not cross
        the original image H or W ) the original data will be used, else 0 padding will be used.
    """

    def __init__(self,tumor_padding=0):
        self.tumor_padding = tumor_padding

    def __call__(self, sample):
        image, seg = sample

        #Get corners of tumor
        idx = np.argwhere(seg==1)
        bottom_idx,right_idx = np.argmax(idx,axis=0)
        top_idx,left_idx = np.argmin(idx,axis=0)
        bottom = idx[bottom_idx][0]
        top = idx[top_idx][0]
        right = idx[right_idx][1]
        left = idx[left_idx][1]

        #Create padding so that the longer side of the tumor = maximum image size
        l_pad,t_pad,r_pad,b_pad = self.get_padding(left,top,right,bottom)

        #Add some extra padding if we want to zoom out a bit
        t_pad += self.tumor_padding
        b_pad += self.tumor_padding
        l_pad += self.tumor_padding
        r_pad += self.tumor_padding

        #Create new indicies
        new_top = top-t_pad
        new_bottom = bottom+b_pad
        new_left = left-l_pad
        new_right = right+r_pad

        #Where can we use the new indicies and where do we have to pad because the image ends
        top_flag = new_top >= 0
        bottom_flag = new_bottom < seg.shape[0]
        left_flag = new_left >= 0
        right_flag = new_right < seg.shape[1]

        #Cut of indicies if they are outside of the image and assign corresponding pads
        new_top,new_t_pad = (new_top,0) if top_flag else (0,t_pad-top)
        new_bottom,new_b_bad = (new_bottom,0) if bottom_flag else (seg.shape[0]-1,new_bottom-(seg.shape[0]-1))
        new_left,new_l_pad = (new_left,0) if left_flag else (0,l_pad-left)
        new_right,new_r_pad = (new_right,0) if right_flag else (seg.shape[1]-1,new_right-(seg.shape[1]-1))

        return np.pad(image[new_top:new_bottom,new_left:new_right],((new_t_pad,new_b_bad),(new_l_pad,new_r_pad)))
    
    def get_padding(self,left,top,right,bottom):
        """
        Creates padding around the tumor which ensures an output image with aspect ratio 1
        """
        d_bt = bottom-top
        d_rl = right-left

        if d_bt > d_rl:
            h_padding = (d_bt-d_rl) / 2
            v_padding = 0
        else:
            h_padding = 0
            v_padding = (d_rl-d_bt)/2
        l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
        t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
        r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
        b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5

        padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))

        return padding    


class TumorCrop3d(object):
    
    def __init__(self,tumor_padding=10):
        self.tumor_padding = tumor_padding
        
    def __call__(self,sample):
        vol_data, seg_data = sample
        
        top,bottom,left,right,front,back = self.get_tumor(seg_data)
        
        pad_size = self.tumor_padding
        
        t_pad = pad_size 
        b_pad = pad_size 
        l_pad = pad_size 
        r_pad = pad_size 
        f_pad = pad_size
        bk_pad = pad_size

        #Create new indicies
        new_top = top-t_pad
        new_bottom = bottom+b_pad
        new_left = left-l_pad
        new_right = right+r_pad
        new_front = front-f_pad
        new_back = back+bk_pad

        #Where can we use the new indicies and where do we have to pad because the image ends
        top_flag = new_top >= 0
        bottom_flag = new_bottom < vol_data.shape[0]
        left_flag = new_left >= 0
        right_flag = new_right < vol_data.shape[1]
        front_flag = new_front >= 0
        back_flag = new_back < vol_data.shape[2]

        #Cut of indicies if they are outside of the image and assign corresponding pads
        new_top,new_t_pad = (new_top,0) if top_flag else (0,t_pad-top)
        new_bottom,new_b_bad = (new_bottom,0) if bottom_flag else (vol_data.shape[0]-1,new_bottom-(vol_data.shape[0]-1))
        new_left,new_l_pad = (new_left,0) if left_flag else (0,l_pad-left)
        new_right,new_r_pad = (new_right,0) if right_flag else (vol_data.shape[1]-1,new_right-(vol_data.shape[1]-1))
        new_front,new_f_pad = (new_front,0) if front_flag else (0,f_pad-front)
        new_back,new_bk_pad = (new_back,0) if back_flag else (vol_data.shape[2]-1,new_back-(vol_data.shape[2]-1))
        

        return np.pad(vol_data[new_top:new_bottom+1,new_left:new_right+1,new_front:new_back+1],((new_t_pad,new_b_bad),(new_l_pad,new_r_pad),(new_f_pad,new_bk_pad)))
    
    def get_tumor(self,seg_data): 
        idx = np.argwhere(seg_data==1)
        bottom_idx,right_idx,back_idx = np.max(idx,axis=0)
        top_idx,left_idx,front_idx = np.min(idx,axis=0)
        return top_idx,bottom_idx,left_idx,right_idx,front_idx,back_idx
    
class RandomCrop3d(object):
    
    def __init__(self,size=(100,100,100)):
        self.size = size
    
    def __call__(self,scan): 
        h,w,d = self.size
        assert h <= scan.shape[0], "Cropping H < Scan H"
        assert w <= scan.shape[1], "Cropping W < Scan W"
        assert d <= scan.shape[2], "Cropping D < Scan D"
        
        top_idx = np.random.randint(0,scan.shape[0]-(h-1))
        bottom_idx = top_idx+h
        left_idx = np.random.randint(0,scan.shape[1]-(w-1))
        right_idx = left_idx+w
        front_idx = np.random.randint(0,scan.shape[2]-(d-1))
        back_idx = front_idx+d

        return scan[top_idx:bottom_idx,left_idx:right_idx,front_idx:back_idx]
        
    
    

class CornerAndCenterCrop3d(object):
    
    
    def __init__(self,size):
        self.size = size
        
    def __call__(self,scan):
        
        h,w,d = self.size
        
        assert h <= scan.shape[0], "Cropping H < Scan H"
        assert w <= scan.shape[1], "Cropping W < Scan W"
        assert d <= scan.shape[2], "Cropping D < Scan D"

        #left
        left_top_front = scan[:h,:w,:d]
        left_top_back = scan[:h,:w,scan.shape[2]-d:]
        left_bottom_front = scan[scan.shape[0]-h:,:w,:d]
        left_bottom_back = scan[scan.shape[0]-h:,:w:,scan.shape[2]-d:]

        #right
        right_top_front = scan[:h,scan.shape[1]-w:,:d]
        right_top_back = scan[:h,scan.shape[1]-w:,scan.shape[2]-d:]
        right_bottom_front = scan[scan.shape[0]-h:,scan.shape[1]-w:,:d]
        right_bottom_back = scan[scan.shape[0]-h:,scan.shape[1]-w:,scan.shape[2]-d:]

        #center
        top,bottom = ((scan.shape[0]-h)//2, (scan.shape[0]-h)//2) if (scan.shape[0]-h)%2 == 0 else ((scan.shape[0]-h)//2 +1, (scan.shape[0]-h)//2)
        left,right = ((scan.shape[1]-w)//2, (scan.shape[1]-w)//2) if (scan.shape[1]-w)%2 == 0 else ((scan.shape[1]-w)//2 +1, (scan.shape[1]-w)//2)
        front,back = ((scan.shape[2]-d)//2, (scan.shape[2]-d)//2) if (scan.shape[2]-d)%2 == 0 else ((scan.shape[2]-d)//2 +1, (scan.shape[2]-d)//2)
        center = scan[top:scan.shape[0]-bottom,left:scan.shape[1]-right,front:scan.shape[2]-back]
        
        return np.stack([left_top_front,
                        left_top_back,
                        left_bottom_front,
                        left_bottom_back,
                        right_top_front,
                        right_top_back,
                        right_bottom_front,
                        right_bottom_back,
                        center])
    
    
class CenterCrop3d(object):
    
    
    def __init__(self,size):
        self.size = size
        
    def __call__(self,scan):
        
        h,w,d = self.size
        
        assert h <= scan.shape[0], "Cropping H < Scan H"
        assert w <= scan.shape[1], "Cropping W < Scan W"
        assert d <= scan.shape[2], "Cropping D < Scan D"

        #center
        top,bottom = ((scan.shape[0]-h)//2, (scan.shape[0]-h)//2) if (scan.shape[0]-h)%2 == 0 else ((scan.shape[0]-h)//2 +1, (scan.shape[0]-h)//2)
        left,right = ((scan.shape[1]-w)//2, (scan.shape[1]-w)//2) if (scan.shape[1]-w)%2 == 0 else ((scan.shape[1]-w)//2 +1, (scan.shape[1]-w)//2)
        front,back = ((scan.shape[2]-d)//2, (scan.shape[2]-d)//2) if (scan.shape[2]-d)%2 == 0 else ((scan.shape[2]-d)//2 +1, (scan.shape[2]-d)//2)
        center = scan[top:scan.shape[0]-bottom,left:scan.shape[1]-right,front:scan.shape[2]-back]
        
        return center



class Resize_Volume_Keeping_AR(object):
    """Resizes a Volume maintaing the original aspect ratio.
    """
    
    def __init__(self,min_size):
        """Indicate the minimum magnitude a axis should have. 
        One of the three axis will be resized to this size, while the
        others will be resized according to the aspect ratio.

        Args:
            min_size (int): size of the smallest axis
        """
        self.min_size= min_size
        
    def __call__(self,volume):

        h,w,d = volume.shape   
    
        min_dim = np.argmin(volume.shape)
                
        if min_dim == 0:
            w = int(self.min_size/h * w)
            d = int(self.min_size/h * d)
            h = self.min_size
        elif min_dim ==1:
            h = int(self.min_size/w * h)
            d = int(self.min_size/w * d)
            w = self.min_size
        elif min_dim ==2:
            h = int(self.min_size/d * h)
            w = int(self.min_size/d * w)
            d = self.min_size
        else:
            raise "Error"
            
        volume = resize(volume,(h,w,d),order=3,anti_aliasing=True)
        
        return volume


class ElasticTransform(object):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognit
       https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
    """
        
    def __init__(self,alpha,sigma,random_state=None):
        
        self.alpha = alpha
        self.sigma = sigma
        self.random_state = random_state
        
    def __call__(self,image):
        
        image = image.squeeze(0)
        assert len(image.shape)==2

        if self.random_state is None:
            self.random_state = np.random.RandomState(None)

        shape = image.shape

        dx = gaussian_filter((self.random_state.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
        dy = gaussian_filter((self.random_state.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

        return np.expand_dims(map_coordinates(image, indices, order=1).reshape(shape),0)