import argparse
import cv2
from MyFCN import MyFCN
import numpy as np
import os
from State import State
import torch
from utils.common import *

torch.manual_seed(1)

# =====================================================================================
# arguments parser
# =====================================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--sigma",         type=int,   default=15,              help='-')
parser.add_argument("--model-path",    type=str,   default="checkpoint/",   help='-')
parser.add_argument("--save-images",   type=int,   default=0,               help='-')
FLAG, unparsed = parser.parse_known_args()


# =====================================================================================
# Global variables
# =====================================================================================

MEAN = 0.0
SIGMA = FLAG.sigma
MODEL_PATH = FLAG.model_path
SAVE_IMAGES = (FLAG.save_images == 1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_ACTIONS = 9
GAMMA = 0.95
T_MAX = 5

DATASET_DIR = "../dataset/denoise/test"
LS_IMAGE_PATHS = sorted_list(DATASET_DIR)


# =====================================================================================
# Test each image
# =====================================================================================

def main():
    if SAVE_IMAGES:
        os.makedirs("results", exist_ok=True)

    CURRENT_STATE = State()

    MODEL = MyFCN(N_ACTIONS).to(DEVICE)
    if exists(MODEL_PATH):
        MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)))
    MODEL.train(False)

    TOTAL_REWARD = 0
    SUM_PSNR = 0
    for i in range(0, len(LS_IMAGE_PATHS)):
        image_path = LS_IMAGE_PATHS[i]
        label_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        label_image = norm01(label_image)[np.newaxis, ...]
        label_image = np.expand_dims(label_image, 0)

        noise = np.random.normal(MEAN, SIGMA, label_image.shape) / 255
        noise_image = np.clip(label_image + noise, 0.0, 1.0)
        noise_image = np.float32(noise_image)
        CURRENT_STATE.reset(noise_image)

        sum_reward = 0
        for t in range(0, T_MAX):
            prev_image = CURRENT_STATE.image.copy()
            statevar = torch.as_tensor(CURRENT_STATE.tensor, dtype=torch.float32).to(DEVICE)
            pi, _, inner_state = MODEL.pi_and_v(statevar)

            actions = torch.argmax(pi, dim=1).cpu()
            inner_state = inner_state.cpu()

            CURRENT_STATE.step(actions, inner_state)

            reward = (np.square(label_image - prev_image) - np.square(label_image - CURRENT_STATE.image)) * 255
            sum_reward += np.mean(reward) * np.power(GAMMA, t)

        current_image = np.clip(CURRENT_STATE.image, 0.0, 1.0)
        psnr = PSNR(label_image, current_image)
        TOTAL_REWARD += sum_reward
        SUM_PSNR += psnr

        if SAVE_IMAGES:
            saved_image = denorm01(current_image[0, 0])
            saved_image = np.uint8(saved_image)
            cv2.imwrite(f"results/image_{i}_psnr({psnr:.2f}).png", saved_image)

    print(f"Average reward: {TOTAL_REWARD * 255 / len(LS_IMAGE_PATHS)}")
    print(f"Average PSNR: {SUM_PSNR / len(LS_IMAGE_PATHS)}")


if __name__ == '__main__':
    main()