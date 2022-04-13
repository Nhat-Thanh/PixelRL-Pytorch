from re import A
from tkinter import W
from MyFCN import MyFCN
from State import State
from utils.common import PSNR, denorm01, exists, sorted_list, norm01
import argparse
import torch
import os
import cv2
import numpy as np

torch.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument("--sigma",           type=int,   default=15,   help='-')
parser.add_argument("--model-path",        type=str,   default="checkpoint/",      help='-')
parser.add_argument("--save-images",        type=int,   default=0,      help='-')

FLAG, unparsed = parser.parse_known_args()
sigma = FLAG.sigma
model_path = FLAG.model_path
save_images = (FLAG.save_images == 1)

if save_images:
    os.makedirs("results", exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_actions = 9
gamma = 0.95
t_max = 5

dataset_dir = "dataset/test"
ls_images = sorted_list(dataset_dir)
isEnd = False
current_state = State()
model = MyFCN(n_actions).to(device)

if exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

model.train(False)

sum_reward = 0
total_reward = 0
sum_psnr = 0
for i in range(0, len(ls_images)):
    image_path = ls_images[i]
    label_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    label_image = norm01(label_image)[np.newaxis, ...]
    label_image = np.expand_dims(label_image, 0)

    noise = np.random.normal(0.0, sigma, label_image.shape) / 255
    noise_image = np.clip(label_image + noise, 0.0, 1.0)
    noise_image = np.float32(noise_image)
    current_state.reset(noise_image)

    reward = 0
    sum_reward = 0
    for t in range(0, t_max):
        prev_image = current_state.image.copy()
        statevar = torch.as_tensor(current_state.tensor, dtype=torch.float32).to(device)
        pi, _, inner_state = model.pi_and_v(statevar)

        actions = torch.argmax(pi, dim=1).cpu()
        inner_state = inner_state.cpu()

        current_state.step(actions, inner_state)

        reward = (np.square(label_image - prev_image) - np.square(label_image - current_state.image)) * 255
        sum_reward += np.mean(reward) * np.power(gamma, t)
    
    current_image = np.clip(current_state.image, 0.0, 1.0)
    psnr = PSNR(label_image, current_image)
    total_reward += sum_reward
    sum_psnr += psnr

    if save_images:
        saved_image = denorm01(current_image[0, 0])
        saved_image = np.uint8(saved_image)
        cv2.imwrite(f"results/image_{i}_{psnr:.2f}.png", saved_image)

print(f"Average reward: {total_reward / len(ls_images)}")
print(f"Average PSNR: {sum_psnr / len(ls_images)}")
