import cv2
import numpy as np
import os
from PIL import Image
from utils.common import *

class dataset:
    def __init__(self, dataset_dir, subset):
        self.cur_idx = 0
        self.data = np.array([])
        self.labels = np.array([])
        self.label_file = "labels_{}.npy".format(subset)
        self.data_file = "data_{}.npy".format(subset)
        self.subset_dir = os.path.join(dataset_dir, subset)
        self.text_dir = os.path.join(dataset_dir, "text")
    
    def generate(self, h_crop_size, w_crop_size, transform=False):      
        if exists(self.label_file) and exists(self.data_file):
            print("{} HAS ALREADY EXISTED\n".format(self.data_file))
            print("{} HAS ALREADY EXISTED\n".format(self.label_file))
            return

        ls_images = sorted_list(self.subset_dir)
        ls_texts = sorted_list(self.text_dir)

        labels = []
        data = []

        for i in range(0, len(ls_images)):
            image_path = ls_images[i]
            print(image_path)
            
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if transform:
                if np.random.rand() > 0.5:
                    image = np.fliplr(image)
                if np.random.rand() > 0.5:
                    h, w = image.shape
                    angle = 10 * np.random.rand()
                    if np.random.rand() > 0.5:
                        angle *= -1
                    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
                    image = cv2.warpAffine(image, M, (w, h))

            image_text = Image.fromarray(image)
            text_id = np.random.randint(0, len(ls_texts))
            mask = cv2.imread(ls_texts[text_id], cv2.IMREAD_GRAYSCALE)
            text_value = 255 * (np.random.rand() > 0.5)

            if np.random.rand() > 0.5:
                mask = np.fliplr(mask)
            if np.random.rand() > 0.5:
                h, w = mask.shape
                angle = 10*np.random.rand()
                if np.random.rand() > 0.5:
                    angle *= -1
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
                mask = cv2.warpAffine(mask, M, (w, h))

            mask = random_crop(mask, h_crop_size, w_crop_size)
            mask = Image.fromarray(mask)
            image_text.paste(text_value, mask)
            image_text = np.array(image_text)
            
            image = norm01(image)
            image_text = norm01(image_text)
            h, w = image.shape
            for x in np.arange(start=0, stop=h-h_crop_size, step=h_crop_size - 1):
                for y in np.arange(start=0, stop=w-w_crop_size, step=w_crop_size - 1):
                    subim_data  = image_text[x : x + h_crop_size, y : y + w_crop_size]
                    subim_label = image[x : x + h_crop_size, y : y + w_crop_size]

                    data.append(subim_data)
                    labels.append(subim_label)
        
        data = np.array(data, dtype=np.float32)[:,np.newaxis,:,:]
        labels = np.array(labels, dtype=np.float32)[:,np.newaxis,:,:]
        np.save(self.data_file, data)
        np.save(self.label_file, labels)

    def load_data(self, shuffle_arrays : bool):
        if not exists(self.data_file):
            ValueError("\n{} DOES NOT EXIST\n".format(self.data_file))

        if not exists(self.label_file):
            ValueError("\n{} DOES NOT EXIST\n".format(self.label_file))

        self.data = np.load(self.data_file)
        self.labels = np.load(self.label_file)

        if shuffle_arrays:
            indices = np.random.permutation(self.data.shape[0])
            self.data = self.data[indices]
            self.labels = self.labels[indices]

    def get_batch(self, batch_size, shuffle_each_epoch=True):
        # Ignore remaining dataset because of  
        # shape error when run torch.mean()
        isEnd = False
        if self.cur_idx + batch_size > self.labels.shape[0]:
            isEnd = True
            self.cur_idx = 0
            if shuffle_each_epoch:
                indices = np.random.permutation(self.labels.shape[0])
                self.data = self.data[indices]
                self.labels = self.labels[indices]

        data = self.data[self.cur_idx : self.cur_idx + batch_size] 
        labels = self.labels[self.cur_idx : self.cur_idx + batch_size]
        self.cur_idx += batch_size
        
        return data, labels, isEnd
