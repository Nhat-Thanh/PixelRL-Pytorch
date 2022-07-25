import cv2
import numpy as np
import os
import torch
from utils.common import *

class dataset:
    def __init__(self, dataset_dir, subset, enhance_type):
        self.cur_idx = 0
        self.data = torch.Tensor([])
        self.data_file = "data_{}_{}.npy".format(subset, enhance_type)
        self.data_dir = os.path.join(dataset_dir, "Data", subset)

        self.labels = torch.Tensor([])
        self.label_file = "label_{}_{}.npy".format(subset, enhance_type)
        self.labels_dir = os.path.join(dataset_dir, "Labels", enhance_type, subset)

    def generate(self, h_crop_size, w_crop_size):
        if exists(self.label_file) and exists(self.data_file):
            print("{} HAS ALREADY EXISTED\n".format(self.data_file))
            print("{} HAS ALREADY EXISTED\n".format(self.label_file))
            return

        data_files = sorted_list(self.data_dir)
        label_files = sorted_list(self.labels_dir)

        data = []
        labels = []

        for i in range(0, len(data_files)):
            raw_x_path = data_files[i]
            raw_y_path = label_files[i]

            print(raw_x_path)

            raw_x = cv2.imread(raw_x_path)
            raw_y = cv2.imread(raw_y_path)

            raw_x = norm01(raw_x)
            raw_y = norm01(raw_y)

            h, w, _ = raw_x.shape
            for x in np.arange(start=0, stop=h-h_crop_size, step=h_crop_size - 1):
                for y in np.arange(start=0, stop=w-w_crop_size, step=w_crop_size - 1):
                    subim_data  = raw_x[x : x + h_crop_size, y : y + w_crop_size]
                    subim_label = raw_y[x : x + h_crop_size, y : y + w_crop_size]

                    data.append(subim_data)
                    labels.append(subim_label)

        data = np.array(data, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)

        # b,h,w,c -> b,c,h,w
        data = np.transpose(data, [0, 3, 1, 2])
        labels = np.transpose(labels, [0, 3, 1, 2])

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
            indices = np.random.permutation(self.labels.shape[0])
            self.data = self.data[indices]
            self.labels = self.labels[indices]

    def get_batch(self, batch_size, shuffle_each_epoch=True):
        # Ignore remaining dataset because of  
        # shape error when run torch.mean()
        isEnd = False
        if self.cur_idx + batch_size > self.data.shape[0]:
            isEnd = True
            self.cur_idx = 0
            if shuffle_each_epoch:
                indices = np.random.permutation(self.data.shape[0])
                self.data = self.data[indices]
                self.labels = self.labels[indices]

        data = self.data[self.cur_idx : self.cur_idx + batch_size]
        labels = self.labels[self.cur_idx : self.cur_idx + batch_size]
        self.cur_idx += batch_size

        return data, labels, isEnd

