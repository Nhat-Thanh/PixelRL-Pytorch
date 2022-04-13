import cv2
import numpy as np

class State:
    def __init__(self):
        self.image = None
        self.tensor = None
        self.move_range = 3
    
    def reset(self, noise_img):
        self.image = noise_img
    
    def set(self, noise_img):
        self.image = noise_img
    
    def step(self, act):
        act = act.detach().numpy()
        box         = np.zeros(shape=self.image.shape, dtype=self.image.dtype)
        median      = np.zeros(shape=self.image.shape, dtype=self.image.dtype)
        bilateral   = np.zeros(shape=self.image.shape, dtype=self.image.dtype)
        bilateral_2 = np.zeros(shape=self.image.shape, dtype=self.image.dtype)
        gaussian    = np.zeros(shape=self.image.shape, dtype=self.image.dtype)
        gaussian_2  = np.zeros(shape=self.image.shape, dtype=self.image.dtype)

        neutral = (self.move_range - 1) / 2
        move = act.astype(np.float32)
        move = (move - neutral) / 255
        moved_image = self.image + move[:,np.newaxis,:,:]

        b = act.shape[0]
        for i in range(0, b):
            if np.sum(act[i] == 3):
                gaussian[i, 0] = cv2.GaussianBlur(self.image[i, 0],  ksize=(5,5), sigmaX=0.5)

            if np.sum(act[i] == 4):
                bilateral[i, 0] = cv2.bilateralFilter(self.image[i, 0], d=5, sigmaColor=0.1, sigmaSpace=5)

            if np.sum(act[i] == 5):
                median[i, 0] = cv2.medianBlur(self.image[i, 0], ksize=5)

            if np.sum(act[i] == 6):
                gaussian_2[i, 0] = cv2.GaussianBlur(self.image[i, 0], ksize=(5,5), sigmaX=1.5)

            if np.sum(act[i] == 7):
                bilateral_2[i, 0] = cv2.bilateralFilter(self.image[i, 0], d=5, sigmaColor=1.0, sigmaSpace=5)

            if np.sum(act[i] == 8):
                box[i, 0] = cv2.boxFilter(self.image[i, 0], ddepth=-1, ksize=(5,5))

        self.image = moved_image 
        self.image = np.where(act[:,np.newaxis,:,:]==3, gaussian,    self.image)
        self.image = np.where(act[:,np.newaxis,:,:]==4, bilateral,   self.image)
        self.image = np.where(act[:,np.newaxis,:,:]==5, median,      self.image)
        self.image = np.where(act[:,np.newaxis,:,:]==6, gaussian_2,  self.image)
        self.image = np.where(act[:,np.newaxis,:,:]==7, bilateral_2, self.image)
        self.image = np.where(act[:,np.newaxis,:,:]==8, box,         self.image)
