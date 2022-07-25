import argparse
from tkinter import CURRENT
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
parser.add_argument("--model-path",   type=str, default="default",            help='-')
parser.add_argument("--save-images",  type=int, default=0,                    help='-')
parser.add_argument("--enhance-type", type=str, default="Foreground Pop-Out", help='-')
FLAG, unparsed = parser.parse_known_args()


# =====================================================================================
# Global variables
# =====================================================================================

ENHANCE_TYPE = FLAG.enhance_type
MODEL_PATH = FLAG.model_path
if (MODEL_PATH == "") or (MODEL_PATH == "default"):
    MODEL_PATH = os.path.join("checkpoint", ENHANCE_TYPE, "model.pt")
SAVE_IMAGES = (FLAG.save_images == 1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_ACTIONS = 13
GAMMA = 0.95
T_MAX = 10

ROOT_DIR = "../../"
DATA_DIR = os.path.join(ROOT_DIR, "Dataset/Enhance_color/Data/test")
LABEL_DIR = os.path.join(ROOT_DIR, "Dataset/Enhance_color/Labels", ENHANCE_TYPE, "test")
LS_DATA_PATHS = sorted_list(DATA_DIR)
LS_LABEL_PATHS = sorted_list(LABEL_DIR)



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
    TOTAL_L2_ERROR = 0
    with torch.no_grad():
        for i in range(0, len(LS_DATA_PATHS)):
            data_path = LS_DATA_PATHS[i]
            data = cv2.imread(data_path)
            data = np.transpose(data, [2, 0, 1])
            data = np.expand_dims(data, 0)
            data= np.float32(data)

            label_path = LS_LABEL_PATHS[i]
            label = cv2.imread(label_path)
            label = np.transpose(label, [2, 0, 1])
            label = np.expand_dims(label, 0)
            lab_label = bgr2lab(label)

            CURRENT_STATE.reset(data)

            sum_reward = 0
            for t in range(0, T_MAX):
                prev_lab_image = bgr2lab(CURRENT_STATE.image)
                CURRENT_STATE.set(prev_lab_image)
                statevar = torch.as_tensor(CURRENT_STATE.tensor, dtype=torch.float32).to(DEVICE)
                pi, _, inner_state = MODEL(statevar)

                actions_prob = torch.softmax(pi, dim=1)
                actions = torch.argmax(actions_prob, dim=1)

                CURRENT_STATE.step(actions, inner_state)

                current_lab_image = bgr2lab(CURRENT_STATE.image)
                reward = L2(lab_label, prev_lab_image) - L2(lab_label, current_lab_image)
                sum_reward += np.mean(reward) * np.power(GAMMA, t)

            current_lab_image = bgr2lab(CURRENT_STATE.image)
            l2_error = np.mean(L2(lab_label, current_lab_image))
            TOTAL_REWARD += sum_reward
            TOTAL_L2_ERROR += l2_error

            if SAVE_IMAGES:
                image = np.transpose(CURRENT_STATE.image[0], [1, 2, 0])
                image = np.clip(image, 0, 255)
                image = np.uint8(image)
                cv2.imwrite("results/image-{}-L2({:.2f}).png".format(i, l2_error), image)

    print("Average reward: {}".format(TOTAL_REWARD / len(LS_DATA_PATHS)))
    print("Average L2 error: {}".format(TOTAL_L2_ERROR / len(LS_DATA_PATHS)))


if __name__ == '__main__':
    main()
