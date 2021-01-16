# Copyright 2016 Bhautik J Joshi
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# imports
from PIL import Image
import math
import abc
import numpy as np

class AbstractProjection:
  __metaclass__ = abc.ABCMeta

  def __init__(self):
    self.image_theta_phi = []
    pass
    
  @staticmethod
  def _loadImage(imageFile):
    img = Image.open(imageFile)
    imsize = img.size
    parsed = Image.new("RGB", imsize, (255, 255, 255))
    bands = img.split()
    parsed.paste(img, mask=(bands[3] if len(bands) == 4 else None))
    npimage = np.array(parsed.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)
    return npimage, imsize

  def loadImage(self, imageFile):
    self.image, self.imsize = self._loadImage(imageFile)
    self.set_angular_resolution()

  @staticmethod
  def _initImage(width, height):
    image = np.ndarray((height, width, 3), dtype=np.uint8)
    return image

  def initImage(self, width, height):
    self.image = self._initImage(width, height)
    self.imsize = (width, height)
    self.set_angular_resolution()

  @staticmethod
  def _saveImage(img, imgsize, destFile):
    mode = 'RGBA'
    arr = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
    if len(arr[0]) == 3:
        arr = np.c_[arr, 255*np.ones((len(arr),1), np.uint8)]
    img =  Image.frombuffer(mode, imgsize, arr.tostring(), 'raw', mode, 0, 1)
    img.save(destFile)

  def saveImage(self, destFile):
    self._saveImage(self.image, self.imsize, destFile)

  def reprojectToThis(self, sourceProjection):
    for x in range(self.imsize[0]):
      for y in range(self.imsize[1]):
        u = float(x)/float(self.imsize[0])
        v = float(y)/float(self.imsize[1])
        theta, phi = self.angular_position((u,v))
        if theta is None or phi is None:
          pixel = (0,0,0)
        else:
          pixel = sourceProjection.pixel_value((theta, phi))
        self.image[y,x] = pixel

  def point_on_sphere(self, theta, phi):
    r = math.cos(phi)
    return (r*math.cos(theta), r*math.sin(theta), math.sin(phi))

  def pixel_value(self, angle):
    return self._pixel_value(angle)

  @abc.abstractmethod
  def _pixel_value(self, angle):
    return None

  @abc.abstractmethod
  def angular_position(self, texcoord):
    return None

  @abc.abstractmethod
  def set_angular_resolution(self):
    return None