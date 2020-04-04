
from torchvision import transforms
import albumentations as A
import albumentations.pytorch as AP
import random
import numpy as np


class AlbumentationTransforms:
  """
  Helper class to create test and train transforms using Albumentations
  """
  def __init__(self, transforms_list=[]):
    
    self.transforms = A.Compose(transforms_list, p=1)


  def __call__(self, img):
    img = np.array(img)
    img=img.astype('uint8')
    #print(img)
    return self.transforms(image=img)['image']
