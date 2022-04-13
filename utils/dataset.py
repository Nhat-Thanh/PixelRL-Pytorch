from utils.common import *
import numpy as np
import torch
import os
import cv2

class dataset:
    def __init__(self, dataset_dir, subset):
        self.dataset_dir = dataset_dir
        self.subset = subset
        self.labels = torch.Tensor([])
        self.labels_file = os.path.join(self.dataset_dir, f"labels_{self.subset}.npy")
        self.cur_idx = 0
    
    def generate(self, h_crop_size, w_crop_size, transform=False):      
        if exists(self.labels_file):
            print(f"{self.labels_file} HAS ALREADY EXISTED\n")
            return
        subset_dir = os.path.join(self.dataset_dir, self.subset)
        ls_images = sorted_list(subset_dir)
        num_crop = 20

        labels = np.zeros(shape=(len(ls_images) * num_crop, 1, h_crop_size, w_crop_size), 
                          dtype=np.float32)

        for i in range(0, len(ls_images)):
            image_path = ls_images[i]
            print(image_path)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if transform:
                h = image.shape[0]
                w = image.shape[1]

                if np.random.rand() > 0.5:
                    image = np.fliplr(image)
                if np.random.rand() > 0.5:
                    angle = 10 * np.random.rand()
                    if np.random.rand() > 0.5:
                        angle *= -1
                    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
                    image = cv2.warpAffine(image, M, (w, h))
            
            image = norm01(image)
            for k in range(0, num_crop):
                labels[i * num_crop + k, 0] = random_crop(image, h_crop_size, w_crop_size)
        
        np.save(self.labels_file, labels)

    def load_data(self, shuffle_arrays : bool):
        if not exists(self.labels_file):
            ValueError(f"\n{self.labels_file} DOES NOT EXIST\n")
        self.labels = np.load(self.labels_file)

        if shuffle_arrays:
            np.random.shuffle(self.labels)
        # self.labels = torch.as_tensor(self.labels, dtype=torch.float32)
    
    def get_batch(self, batch_size, shuffle_each_epoch=True):
        # Ignore remaining dataset because of  
        # shape error when run torch.mean()
        isEnd = False
        if self.cur_idx + batch_size > self.labels.shape[0]:
            isEnd = True
            self.cur_idx = 0
            if shuffle_each_epoch:
                indices = np.random.permutation(self.labels.shape[0])
                self.labels = self.labels[indices]
        
        labels = self.labels[self.cur_idx : self.cur_idx + batch_size]
        noise = np.random.normal(0.0, 15.0, labels.shape) / 255
        data = labels + noise
        self.cur_idx += batch_size
        
        return data, labels, isEnd
