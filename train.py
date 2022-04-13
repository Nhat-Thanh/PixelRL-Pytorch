from MyFCN import MyFCN
from model import PixelWiseA3C_InnerState_ConvR 
from utils.dataset import dataset
from utils.common import PSNR, exists
import argparse
import torch
import os

torch.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument("--sigma",        type=int,   default=15,   help='-')
parser.add_argument("--episodes",        type=int,   default=5,   help='-')
parser.add_argument("--batch-size",      type=int,   default=64,       help='-')
parser.add_argument("--save-every",      type=int,   default=5,        help='-')
parser.add_argument("--ckpt-dir",        type=str,   default="checkpoint/",      help='-')

FLAG, unparsed = parser.parse_known_args()
sigma = FLAG.sigma
episodes = FLAG.episodes
batch_size = FLAG.batch_size
ckpt_dir = FLAG.ckpt_dir
save_every = FLAG.save_every
model_path = os.path.join(ckpt_dir, "model.pt")
pretrain_path = f"initial_weight/denoise_{sigma}_gray_ConvGRU_RMC"
model_path = os.path.join(ckpt_dir, "denoise_{sigma}_ConvGRU_RMC.pt")
ckpt_path = os.path.join(ckpt_dir, "ckpt.pt")

crop_size = 70
dataset_dir = "dataset/"
train_set = dataset(dataset_dir, "train")
train_set.generate(crop_size, crop_size)
train_set.load_data(shuffle_arrays=True)

test_set = dataset(dataset_dir, "test")
test_set.generate(crop_size, crop_size)
test_set.load_data(shuffle_arrays=True)


# -----------------------------------------------------------
#  Train
# -----------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_actions = 9

lr = 1e-3
gamma = 0.95
t_max = 5
beta = 1e-2
model = MyFCN(n_actions).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr)
pixelRL = PixelWiseA3C_InnerState_ConvR(model, t_max, gamma, beta)
pixelRL.setup(optimizer, lr, batch_size, PSNR,  device, model_path, ckpt_path)

pixelRL.load_checkpoint(ckpt_path)
if not exists(ckpt_path):
    print(f"Load pre-trained model at {pretrain_path}")
    pixelRL.load_weights(pretrain_path)

pixelRL.train(train_set, test_set, batch_size, episodes, save_every)
