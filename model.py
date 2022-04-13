from torch.distributions import Categorical
from utils.common import exists
from State import State
import torch
import copy

torch.manual_seed(1)


def MyEntropy(pi):
    log_prob = torch.log(pi)
    entropy = torch.stack([- torch.sum(log_prob * pi, dim=1)]).permute(([1, 0, 2, 3]))
    return entropy

def sample(pi):
    b, n_actions, h, w = pi.shape
    pi_trans = torch.reshape(pi, (-1, n_actions))
    pi_prob = torch.softmax(pi_trans, dim=1)
    actions = torch.multinomial(pi_prob, 1)
    actions = torch.reshape(actions, (b, h, w))
    return actions

# def MyLogProb(pi, actions):
#     b, n_actions, h, w = pi.shape
#     pi_trans = torch.reshape(pi, (-1, n_actions))
#     log_prob_pi = F.log_softmax(pi_trans, dim=1)
#     act_idx = torch.reshape(actions, (-1, 1))
#     selected_pi = torch.take_along_dim(log_prob_pi, act_idx, 1)
#     return torch.reshape(selected_pi, (b, 1, h, w))


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

    def setup(self, optimizer, init_lr, batch_size, metric, device, model_path, ckpt_path):
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
        rewards = []
        metrics = []
        isEnd = False
        current_state = State()
        self.model.train(False)
        self.shared_model.train(False)
        while isEnd == False:
            noise, labels, isEnd = dataset.get_batch(batch_size)
            current_state.reset(noise)

            for t in range(0, self.t_max):
                prev_image = current_state.image.clone()
                noise = current_state.tensor.clone().to(self.device)
                pi, _, inner_state = self.model.pi_and_v(noise)
                actions = torch.argmax(pi, dim=1)
                current_state.step(actions.detach(), inner_state.detach())
                reward = torch.square(labels - prev_image) * 255 - torch.square(labels - current_state.image) * 255
                metric = self.metric(labels, current_state.image)
                rewards.append(reward)
                metrics.append(metric)

        reward = torch.mean(torch.Tensor(rewards))
        metric = torch.mean(torch.Tensor(metrics))

        return reward, metric

    def train(self, train_set, test_set, batch_size, episodes, save_every):
        cur_episode = 0
        if self.ckpt_man is not None:
            cur_episode = self.ckpt_man['episode']
        max_episode = cur_episode + episodes
        self.current_state = State()
        while cur_episode < max_episode:
            cur_episode += 1
            noise, labels, _ = train_set.get_batch(batch_size)
            self.current_state.reset(noise)

            sum_reward, loss = self.train_step(noise, labels)

            print(f"{cur_episode} / {max_episode} - loss: {loss:.6f} - sum reward: {sum_reward * 255:.6f}")

            if cur_episode % save_every == 0:
                reward, metric = self.evaluate(test_set, batch_size)
                print(f"Test - reward: {reward * 255:.6f} - {self.metric.__name__}: {metric:.6f}")

                torch.save(self.model.state_dict(), self.model_path)
                torch.save({
                    'episode': cur_episode,
                    'model': self.model.state_dict(),
                    'shared_model': self.shared_model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                }, self.ckpt_path)

            # self.optimizer.param_groups[0]['lr'] = self.initial_lr - ((1 - cur_episode / max_episode) ** 0.9)

    def train_step(self, noise, labels):
        # reset gradient
        self.model.train(True)
        self.shared_model.train(True)

        sum_reward = 0.0
        reward = torch.zeros_like(labels, dtype=torch.float32)
        t = 0
        while t < self.t_max:
            self.past_rewards[t - 1] = reward.to(self.device)
            prev_image = self.current_state.image.clone()
            noise = self.current_state.tensor.clone().to(self.device)
            pi, v, inner_state = self.model.pi_and_v(noise)

            pi_trans = pi.permute([0, 2, 3, 1])
            actions = Categorical(pi_trans).sample()

            self.current_state.step(actions.detach(), inner_state.detach())

            reward = torch.square(labels - prev_image) * 255 - \
                torch.square(labels - self.current_state.image) * 255
            self.past_log_prob[t] = MyLogProb(pi, actions)
            self.past_entropy[t] = MyEntropy(pi)
            self.past_values[t] = v
            sum_reward += torch.mean(reward) * (self.gamma ** t)
            t += 1
        self.past_rewards[t - 1] = reward.to(self.device)

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
            log_pi = self.past_log_prob[k]
            Advantage = R - v
            pi_loss -= log_pi * Advantage
            pi_loss -= self.beta * entropy 
            v_loss += (v - R) ** 2 / 2

        if self.pi_loss_coef != 1.0:
            pi_loss *= self.pi_loss_coef

        if self.v_loss_coef != 1.0:
            v_loss *= self.v_loss_coef

        total_loss = torch.mean(pi_loss + v_loss)

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
