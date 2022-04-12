from utils.common import exists
import torch.nn.functional as F
from MyFCN import MyFCN
from State import State
import numpy as np
import torch

def MyEntropy(pi):
    prob = torch.softmax(pi, dim=-1, dtype=torch.float32)
    log_prob = F.log_softmax(pi, dim=-1)
    return torch.sum(prob * log_prob, dim=1)

def sample(pi):
    b, n_actions, h, w = pi.shape
    pi_trans = torch.reshape(pi, (-1, n_actions))
    pi_prob = torch.softmax(pi_trans, dim=1)
    actions = torch.multinomial(pi_prob, 1)
    actions = torch.reshape(actions, (b, h, w))
    return actions

def MyLogProb(pi, actions):
    b, n_actions, h, w = pi.shape
    pi_trans = torch.reshape(pi, (-1, n_actions))
    log_prob_pi = F.log_softmax(pi_trans, dim=1)
    act_idx = torch.reshape(actions, (-1, 1))
    selected_pi = torch.take_along_dim(log_prob_pi, act_idx, 1)
    return torch.reshape(selected_pi, (b, 1, h, w))


class PixelWiseA3C_InnerState_ConvR:
    def __init__(self, n_actions, device):
        self.device = device
        self.n_actions = n_actions
        self.model = MyFCN(n_actions).to(device)
        self.current_state = State()
        self.initial_lr = None
        self.optimizer = None
        self.model_path = None
        self.ckpt_path = None
        self.ckpt_man = None
        self.metric = None
        self.gamma = None
        self.t_max = None
        self.beta = None
        self.past_reward = {}
        self.past_value = {}
        self.past_log_pi = {}
        self.past_entropy = {}

    def setup(self, opt_lr, metric, gamma, t_max, beta, model_path, ckpt_path):
        self.initial_lr = opt_lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt_lr)
        self.model_path = model_path
        self.ckpt_path = ckpt_path
        self.metric = metric
        self.gamma = gamma
        self.t_max = t_max
        self.beta = beta

    def load_checkpoint(self, ckpt_path):
        if exists(ckpt_path):
            self.ckpt_man = torch.load(ckpt_path)
            self.optimizer.load_state_dict(self.ckpt_man['optimizer'])
            self.model.load_state_dict(self.ckpt_man['model'])

    def load_weights(self, filepath):
        if exists(filepath):
            ckpt = torch.load(filepath, map_location=torch.device(self.device))
            self.model.load_state_dict(ckpt)

    def evaluate(self, dataset, batch_size):
        rewards = []
        metrics = []
        isEnd = False
        current_state = State()
        while isEnd == False:
            noise, labels, isEnd = dataset.get_batch(batch_size)
            current_state.reset(noise)

            for t in range(0, self.t_max):
                prev_image = torch.clone(current_state.image)
                noise = torch.clone(current_state.tensor).to(self.device)
                pi, _, inner_state = self.model.pi_and_v(noise)
                pi = F.softmax(pi, 1)
                actions = torch.argmax(pi, dim=1)
                current_state.step(actions, inner_state)
                reward = torch.square(
                    labels - prev_image) * 255 - torch.square(labels - current_state.image) * 255
                metric = self.metric(labels, current_state.image)
                rewards.append(reward.numpy())
                metrics.append(metric.numpy())

        reward = np.mean(rewards) * 255
        metric = np.mean(metrics)

        return reward, metric

    def train(self, train_set, test_set, batch_size, episodes, save_every):
        cur_episode = 0
        if self.ckpt_man is not None:
            cur_episode = self.ckpt_man['episode']
        max_episode = cur_episode + episodes
        # torch.autograd.set_detect_anomaly(True)
        while cur_episode < max_episode:
            cur_episode += 1
            noise, labels, _ = train_set.get_batch(batch_size)
            self.current_state.reset(noise)

            # noise = self.current_state.tensor
            # noise = noise.to(self.device)
            # labels = labels.to(self.device)
            sum_reward = self.train_step(noise, labels)

            print(
                f"{cur_episode} / {max_episode} - sum reward: {sum_reward.numpy() * 255:.6f}")

            if cur_episode % save_every == 0:
                reward, metric = self.evaluate(test_set, batch_size)
                print(
                    f"Test - reward: {reward:.6f} - {self.metric.__name__}: {metric:.6f}")

                torch.save(self.model.state_dict(), self.model_path)
                torch.save({
                    'episode': cur_episode,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                }, self.ckpt_path)

            self.optimizer.param_groups[0]['lr'] = self.initial_lr - \
                ((1 - cur_episode / max_episode) ** 0.9)

    def train_step(self, noise, labels):
        # reset gradient
        self.model.train(True)
        self.optimizer.zero_grad()

        sum_reward = 0.0
        for t in range(0, self.t_max):
            prev_image = torch.clone(self.current_state.image)
            noise = torch.clone(self.current_state.tensor).to(self.device)
            pi, v, inner_state = self.model.pi_and_v(noise)

            actions = sample(pi)

            self.current_state.step(actions, inner_state)
            # prev_image = prev_image.to(self.device)
            # cur_image = self.current_state.image.to(self.device)
            reward = torch.square(labels - prev_image) * 255 - \
                torch.square(labels - self.current_state.image) * 255
            self.past_reward[t] = reward.to(self.device)
            self.past_log_pi[t] = MyLogProb(pi, actions)
            self.past_entropy[t] = MyEntropy(pi)
            self.past_value[t] = v
            sum_reward = sum_reward + torch.mean(reward) * (self.gamma ** t)

        pi_loss = 0.0
        v_loss = 0.0
        total_loss = 0.0
        # R = 0 in author's source code
        R = torch.zeros_like(v)
        for k in reversed(range(0, self.t_max)):
            R = R * self.gamma
            R = self.model.conv_smooth(R)
            R = R + self.past_reward[k]
            Advantage = R - self.past_value[k]
            pi_loss = pi_loss - self.past_log_pi[k] * Advantage
            pi_loss = pi_loss - self.beta * self.past_entropy[k]
            v_loss = v_loss + torch.square(Advantage)

        total_loss = torch.mean(pi_loss + v_loss)

        total_loss.backward()
        self.optimizer.step()
        self.model.train(False)

        self.past_log_pi = {}
        self.past_entropy = {}
        self.past_value = {}
        self.past_reward = {}

        return sum_reward

