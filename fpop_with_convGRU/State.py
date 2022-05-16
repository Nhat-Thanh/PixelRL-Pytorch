import cv2
import numpy as np
from utils.common import tensor2numpy

class State:
    def __init__(self):
        self.image = None
        self.tensor = None
        self.move_range = 3
    
    def reset(self, image):
        self.image = image
        b, _, h, w = self.image.shape
        previous_state = np.zeros(shape=(b, 64, h, w), dtype=self.image.dtype)
        self.tensor = np.concatenate([self.image, previous_state], axis=1)
    
    def set(self, image):
        self.image = image.copy()
        self.image[:,0,:,:] /= 100
        self.image[:,1,:,:] /= 127
        self.image[:,2,:,:] /= 127

        self.tensor[:,0:1,:,:] = self.image
    
    def step(self, act, inner_state):
        act = tensor2numpy(act)
        inner_state = tensor2numpy(inner_state)

        bgr1 = self.image.copy()
        bgr1 = bgr1 * 0.95 + 0.5 * 0.05

        bgr2 = self.image.copy()
        bgr2 = bgr2 * 1.05 - 0.5 * 0.5


       
        self.tensor[:,0:1,:,:] = self.image
        self.tensor[:,-64:,:,:] = inner_state
