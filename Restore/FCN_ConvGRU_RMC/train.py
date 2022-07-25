from MyFCN import MyFCN
from model import PixelWiseA3C_ConvGRU_RMC
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
parser.add_argument("--episodes",     type=int,   default=5000,            help='-')
parser.add_argument("--batch-size",   type=int,   default=64,              help='-')
parser.add_argument("--save-every",   type=int,   default=500,             help='-')
parser.add_argument("--ckpt-dir",     type=str,   default="checkpoint/",   help='-')
FLAG, unparsed = parser.parse_known_args()

# =====================================================================================
# Global variables
# =====================================================================================

# Paths
ROOT_DIR = "../../"
PRETRAINED_DIR = os.path.join(ROOT_DIR, "Initial_weights/Restore")

# training settings
BATCH_SIZE = FLAG.batch_size
CKPT_DIR = FLAG.ckpt_dir
CKPT_PATH = os.path.join(CKPT_DIR, "ckpt.pt")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPISODES = FLAG.episodes
MODEL_PATH = os.path.join(CKPT_DIR, "model.pt")
PRETRAINED_PATH = os.path.join(PRETRAINED_DIR, "FCN_ConvGRU_RMC/Inpaint_ConvGRU_RMC.pt")
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
DATASET_DIR = os.path.join(ROOT_DIR, "Dataset/Restore")


# =====================================================================================
# Train
# =====================================================================================

def main():
    train_set = dataset(DATASET_DIR, "train")
    train_set.generate(CROP_SIZE, CROP_SIZE)
    train_set.load_data(shuffle_arrays=True)

    test_set = dataset(DATASET_DIR, "test")
    test_set.generate(CROP_SIZE, CROP_SIZE)
    test_set.load_data(shuffle_arrays=True)

    MODEL = MyFCN(N_ACTIONS).to(DEVICE)
    OPTIMIZER = torch.optim.Adam(MODEL.parameters(), LEARNING_RATE)
    pixelRL = PixelWiseA3C_ConvGRU_RMC(MODEL, T_MAX, GAMMA, BETA)
    pixelRL.setup(OPTIMIZER, LEARNING_RATE, BATCH_SIZE, PSNR,  DEVICE, MODEL_PATH, CKPT_PATH)

    pixelRL.load_checkpoint(CKPT_PATH)
    if not exists(CKPT_PATH):
        print("Load pre-trained model at {}".format(PRETRAINED_PATH))
        pixelRL.load_weights(PRETRAINED_PATH)

    pixelRL.train(train_set, test_set, BATCH_SIZE, EPISODES, SAVE_EVERY)


if __name__ == '__main__':
    main()
