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
parser.add_argument("--model-path",    type=str,   default="checkpoint/",   help='-')
parser.add_argument("--save-images",   type=int,   default=0,               help='-')
FLAG, unparsed = parser.parse_known_args()


# =====================================================================================
# Global variables
# =====================================================================================

ROOT_DIR = "../../"

MODEL_PATH = FLAG.model_path
SAVE_IMAGES = (FLAG.save_images == 1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_ACTIONS = 9
GAMMA = 0.95
T_MAX = 15

DATA_DIR = os.path.join(ROOT_DIR, "Dataset/Restore/test_inpaint")
lS_DATA_PATHS = sorted_list(DATA_DIR)
LABEL_DIR = os.path.join(ROOT_DIR, "Dataset/Restore/test")
lS_LABEL_PATHS = sorted_list(DATA_DIR)


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
    for i in range(0, len(lS_DATA_PATHS)):
        text_image_path = lS_DATA_PATHS[i]
        text_image = cv2.imread(text_image_path, cv2.IMREAD_GRAYSCALE)
        text_image = norm01(text_image)[np.newaxis, ...]
        text_image = np.expand_dims(text_image, 0)
        CURRENT_STATE.reset(text_image)

        label_image_path = lS_LABEL_PATHS[i]
        label_image = cv2.imread(label_image_path, cv2.IMREAD_GRAYSCALE)
        label_image = norm01(label_image)[np.newaxis, ...]
        label_image = np.expand_dims(label_image, 0)
 
        sum_reward = 0
        with torch.no_grad():
            for t in range(0, T_MAX):
                prev_image = CURRENT_STATE.image.copy()
                statevar = torch.as_tensor(CURRENT_STATE.tensor, dtype=torch.float32).to(DEVICE)
                pi, _, inner_state = MODEL(statevar)

                actions_prob = torch.softmax(pi, dim=1).cpu()
                actions = torch.argmax(actions_prob, dim=1)
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
            cv2.imwrite("results/image-{}-psnr({:.2f}).png".format(i, psnr), saved_image)

    print("Average reward: {}".format(TOTAL_REWARD * 255 / len(lS_DATA_PATHS)))
    print("Average PSNR: {}".format(SUM_PSNR / len(lS_DATA_PATHS)))


if __name__ == '__main__':
    main()
