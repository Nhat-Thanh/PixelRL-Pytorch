import numpy as np
import os
import torch

# list all file in dir and sort
def sorted_list(dir):
    ls = os.listdir(dir)
    ls.sort()
    for i in range(0, len(ls)):
        ls[i] = os.path.join(dir, ls[i])
    return ls

def norm01(src):
    return src / 255

def denorm01(src):
    return src * 255

def exists(path):
    return os.path.exists(path)

def PSNR(y_true, y_pred, max_val=1):
    y_true = np.float32(y_true)
    y_pred = np.float32(y_pred)
    MSE = np.mean(np.square(y_true - y_pred))
    return 10 * np.log10(max_val * max_val / MSE)

def random_crop(src, h_crop_size, w_crop_size):
    h = src.shape[0]
    w = src.shape[1]
    x = np.random.randint(0, h - h_crop_size)
    y = np.random.randint(0, w - w_crop_size)
    return src[x : x + h_crop_size, y : y + w_crop_size]

def random_transform(src):
    _90_left, _90_right, _180 = 1, 3, 2
    operations = {
        0 : (lambda x : x                                       ),
        1 : (lambda x : torch.rot90(x, k=_90_left,  dims=(1, 2))),
        2 : (lambda x : torch.rot90(x, k=_90_right, dims=(1, 2))),
        3 : (lambda x : torch.rot90(x, k=_180,      dims=(1, 2))),
        4 : (lambda x : torch.fliplr(x)                         ),
        5 : (lambda x : torch.flipud(x)                         ),
    }
    idx = np.random.choice([0, 1, 2, 3, 4, 5])
    image_transform = operations[idx](src)
    return image_transform

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()