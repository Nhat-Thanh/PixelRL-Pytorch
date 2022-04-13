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
t_max = 5

dataset_dir = "dataset/test"
ls_images = sorted_list(dataset_dir)
isEnd = False
current_state = State()
model = MyFCN(n_actions).to(device)

if exists(model_path):
    model.load_state_dict(torch.load(model_path), map_location=torch.device(device))

model.eval()

sum_reward = 0
sum_psnr = 0
for i in range(0, len(ls_images)):
    image_path = ls_images[i]
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = norm01(img)[np.newaxis, ...]
    img = np.expand_dims(img, 0) 
    noise = np.random.normal(0.0, sigma, img.shape)
    noise_img = np.clip(img + noise, 0.0, 1.0)
 
    rewards = []
    metrics = []
    current_state.reset(noise_img)

    for t in range(0, t_max):
        prev_image = current_state.image.copy()
        noise_img = torch.as_tensor(current_state.tensor, dtype=torch.float32).to(device)
        pi, _, inner_state = model.pi_and_v(noise_img)

        actions = torch.argmax(pi, dim=1).cpu()
        inner_state = inner_state.cpu()

        current_state.step(actions, inner_state)

        reward = (np.square(img - prev_image) - np.square(img - current_state.image)) * 255

        rewards.append(reward)
    
    psnr = PSNR(img, current_state.image)
    sum_reward += np.mean(rewards)
    sum_psnr += psnr

    if save_images:
        saved_image = denorm01(current_state.image[0, 0])
        saved_image = np.uint8(saved_image)
        cv2.imwrite(f"results/image_{i}_{psnr:.2f}.png", saved_image)

print(f"Average reward: {sum_reward / len(ls_images)}")
print(f"Average PSNR: {sum_psnr / len(ls_images)}")
