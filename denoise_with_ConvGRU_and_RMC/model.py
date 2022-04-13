import copy
import numpy as np
from State import State
import torch
from torch.distributions import Categorical
from utils.common import exists

torch.manual_seed(1)

def MyEntropy(pi):
    log_prob = torch.log(pi)
    entropy = torch.stack([- torch.sum(log_prob * pi, dim=1)]).permute(([1, 0, 2, 3]))
    return entropy

def MyLogProb(pi, actions):
    selected_pi = pi.gather(1, actions.unsqueeze(1))
    log_prob = torch.log(selected_pi)
    return log_prob 


class PixelWiseA3C_InnerState_ConvR:
    def __init__(self, model, t_max, gamma, beta=1e-2,
                 pi_loss_coef=1.0, v_loss_coef=0.5):

        self.shared_model = model 
        self.model = copy.deepcopy(self.shared_model)
        self.ckpt_man = None

        self.t_max = t_max
        self.gamma = gamma
        self.beta = beta
        self.pi_loss_coef = pi_loss_coef
        self.v_loss_coef = v_loss_coef

        self.t = 0
        self.t_start = 0
        self.past_log_prob= {}
        self.past_entropy= {}
        self.past_rewards = {}
        self.past_values = {}

    def setup(self, optimizer, init_lr, batch_size,
              metric, device, model_path, ckpt_path):

        self.device = device
        self.metric = metric
        self.model_path = model_path
        self.ckpt_path = ckpt_path
        self.initial_lr = init_lr
        self.optimizer = optimizer
        self.batch_size = batch_size

    def sync_parameters(self):
        for md_1, md_2 in zip(self.model.modules(), self.shared_model.modules()):
            md_1._buffers = md_2._buffers.copy()

        for target, src in zip(self.model.parameters(), self.shared_model.parameters()):
            target.detach().copy_(src.detach())

    def copy_grad(self, src, target):
        target_params = dict(target.named_parameters())
        for name, param in src.named_parameters():
            if target_params[name].grad is None:
                if param.grad is None:
                    continue
                target_params[name].grad = param.grad
            else:
                if param.grad is None:
                    target_params[name].grad = None
                else:
                    target_params[name].grad[...] = param.grad

    def load_checkpoint(self, ckpt_path):
        if exists(ckpt_path):
            self.ckpt_man = torch.load(ckpt_path)
            self.optimizer.load_state_dict(self.ckpt_man['optimizer'])
            self.shared_model.load_state_dict(self.ckpt_man['shared_model'])
            self.model.load_state_dict(self.ckpt_man['model'])

    def load_weights(self, filepath):
        if exists(filepath):
            ckpt = torch.load(filepath, map_location=torch.device(self.device))
            self.model.load_state_dict(ckpt)

    def evaluate(self, dataset, batch_size):
        self.model.train(False)
        self.shared_model.train(False)
        rewards = []
        metrics = []
        current_state = State()
        isEnd = False
        while isEnd == False:
            noise, labels, isEnd = dataset.get_batch(batch_size)
            current_state.reset(noise)

            for _ in range(0, self.t_max):
                prev_image = current_state.image.copy()
                noise = torch.as_tensor(current_state.tensor.copy()).to(self.device)
                pi, _, inner_state = self.model.pi_and_v(noise)
                
                actions = torch.argmax(pi, dim=1)
                current_state.step(actions.cpu(), inner_state.cpu())
                
                reward = (np.square(labels - prev_image) - np.square(labels - current_state.image)) * 255
                metric = self.metric(labels, current_state.image)
                rewards.append(reward)
                metrics.append(metric)

        reward = np.mean(rewards)
        metric = np.mean(metrics)
        return reward, metric

    def train(self, train_set, test_set, batch_size, episodes, save_every):
        self.current_state = State()

        cur_episode = 0
        if self.ckpt_man is not None:
            cur_episode = self.ckpt_man['episode']
        max_episode = cur_episode + episodes

        while cur_episode < max_episode:
            cur_episode += 1
            noise, labels, _ = train_set.get_batch(batch_size)
            self.current_state.reset(noise)

            sum_reward, loss = self.train_step(noise, labels)

            print(f"{cur_episode} / {max_episode} - loss: {loss:.6f} - sum reward: {sum_reward * 255:.6f}")

            if cur_episode % save_every == 0:
                reward, metric = self.evaluate(test_set, batch_size)
                print(f"Test - reward: {reward * 255:.6f} - {self.metric.__name__}: {metric:.6f}")

                # save model weights
                torch.save(self.model.state_dict(), self.model_path)

                # save current training state
                torch.save({
                    'episode': cur_episode,
                    'model': self.model.state_dict(),
                    'shared_model': self.shared_model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                }, self.ckpt_path)

            # self.optimizer.param_groups[0]['lr'] = self.initial_lr - ((1 - cur_episode / max_episode) ** 0.9)

    def train_step(self, noise, labels):
        self.model.train(True)
        self.shared_model.train(True)

        sum_reward = 0.0
        reward = 0.0
        for t in range(0, self.t_max):
            prev_image = self.current_state.image.copy()
            noise = torch.as_tensor(self.current_state.tensor).to(self.device)
            pi, v, inner_state = self.model.pi_and_v(noise)

            pi_trans = pi.permute([0, 2, 3, 1])
            actions = Categorical(pi_trans).sample().detach()

            self.current_state.step(actions.cpu(), inner_state.cpu())
            reward = (np.square(labels - prev_image) - np.square(labels - self.current_state.image)) * 255

            self.past_rewards[t] = torch.as_tensor(reward).to(self.device)
            self.past_log_prob[t] = MyLogProb(pi, actions)
            self.past_entropy[t] = MyEntropy(pi)
            self.past_values[t] = v
            sum_reward += np.mean(reward) * np.power(self.gamma, t)

        pi_loss = 0.0
        v_loss = 0.0
        # R = 0 in author's source code
        R = torch.zeros_like(v).to(self.device)
        for k in reversed(range(0, self.t_max)):
            R *= self.gamma
            R = self.model.conv_smooth(R)
            R += self.past_rewards[k]
            v = self.past_values[k]
            entropy = self.past_entropy[k]
            log_prob = self.past_log_prob[k]
            Advantage = R - v
            pi_loss -= log_prob * Advantage
            pi_loss -= self.beta * entropy 
            v_loss += torch.square(Advantage) / 2

        if self.pi_loss_coef != 1.0:
            pi_loss *= self.pi_loss_coef

        if self.v_loss_coef != 1.0:
            v_loss *= self.v_loss_coef

        total_loss = torch.nanmean(pi_loss + v_loss)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.copy_grad(src=self.model, target=self.shared_model)
        self.sync_parameters()

        self.past_log_prob = {}
        self.past_entropy = {}
        self.past_values = {}
        self.past_rewards = {}

        return sum_reward, total_loss
