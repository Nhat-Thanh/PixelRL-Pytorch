from MyFCN import MyFCN
from model import PixelWiseA3C_InnerState_ConvR 
from utils.dataset import dataset
from utils.common import PSNR, exists
import argparse
import torch
import os

torch.manual_seed(1)

# =====================================================================================
# arguments parser
# =====================================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--sigma",        type=int,   default=15,              help='-')
parser.add_argument("--episodes",     type=int,   default=5000,            help='-')
parser.add_argument("--batch-size",   type=int,   default=64,              help='-')
parser.add_argument("--save-every",   type=int,   default=500,             help='-')
parser.add_argument("--ckpt-dir",     type=str,   default="checkpoint/",   help='-')
FLAG, unparsed = parser.parse_known_args()

# =====================================================================================
# Global variables
# =====================================================================================

# Noise settings
MEAN = 0.0
SIGMA = FLAG.sigma

# training settings
BATCH_SIZE = FLAG.batch_size
CKPT_DIR = FLAG.ckpt_dir
CKPT_PATH = os.path.join(CKPT_DIR, f"{SIGMA}/ckpt-{SIGMA}.pt")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPISODES = FLAG.episodes
MODEL_PATH = os.path.join(CKPT_DIR, f"{SIGMA}/model-{SIGMA}.pt")
PRETRAINED_PATH = f"initial_weight/denoise_{SIGMA}_gray_ConvGRU_RMC.pt"
SAVE_EVERY = FLAG.save_every

# model settings
N_ACTIONS = 9
LEARNING_RATE = 1e-3

# A3C settings
GAMMA = 0.95
T_MAX = 5
BETA = 1e-2

# Dataset settings
CROP_SIZE = 70
DATASET_DIR = "../dataset/denoise/"


# =====================================================================================
# Train
# =====================================================================================

def main():
    train_set = dataset(DATASET_DIR, "train", (MEAN, SIGMA))
    train_set.generate(CROP_SIZE, CROP_SIZE)
    train_set.load_data(shuffle_arrays=True)

    test_set = dataset(DATASET_DIR, "test", (MEAN, SIGMA))
    test_set.generate(CROP_SIZE, CROP_SIZE)
    test_set.load_data(shuffle_arrays=True)

    MODEL = MyFCN(N_ACTIONS).to(DEVICE)
    OPTIMIZER = torch.optim.Adam(MODEL.parameters(), LEARNING_RATE)
    pixelRL = PixelWiseA3C_InnerState_ConvR(MODEL, T_MAX, GAMMA, BETA)
    pixelRL.setup(OPTIMIZER, LEARNING_RATE, BATCH_SIZE, PSNR,  DEVICE, MODEL_PATH, CKPT_PATH)

    pixelRL.load_checkpoint(CKPT_PATH)
    if not exists(CKPT_PATH):
        print(f"Load pre-trained model at {PRETRAINED_PATH}")
        pixelRL.load_weights(PRETRAINED_PATH)

    pixelRL.train(train_set, test_set, BATCH_SIZE, EPISODES, SAVE_EVERY)


if __name__ == '__main__':
    main()