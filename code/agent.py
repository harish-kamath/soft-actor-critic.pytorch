import os
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from rltorch.memory import MultiStepMemory, PrioritizedMemory
import time

from .model import TwinnedQNetwork, GaussianPolicy
from .utils import grad_false, hard_update, soft_update, to_batch,\
    update_params, RunningMeanStats
from ...utils import EarlyStopping


class SacAgent:

    def __init__(self, env, log_dir, wandb_writer, num_steps=3000000, batch_size=256,
                 lr=0.0003, hidden_units=[256, 256], memory_size=1e6,
                 gamma=0.99, tau=0.005, entropy_tuning=True, ent_coef=0.2,
                 multi_step=1, per=False, alpha=0.6, beta=0.4,
                 beta_annealing=0.0001, grad_clip=None, updates_per_step=1,
                 start_steps=10000, log_interval=10, target_update_interval=1,
                 eval_interval=1000, cuda=True, seed=0, es=False, es_patience=5000, es_delta=3, critic_ensemble=2, **kwargs):
        self.env = env

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        torch.backends.cudnn.deterministic = True  # It harms a performance.
        torch.backends.cudnn.benchmark = False

        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")

        self.policy = GaussianPolicy(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            hidden_units=hidden_units,
            device=self.device).to(self.device)
        self.critic = TwinnedQNetwork(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            num_ensemble=critic_ensemble,
            hidden_units=hidden_units)
        self.critic_target = TwinnedQNetwork(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            num_ensemble=critic_ensemble,
            hidden_units=hidden_units)
        
        for qnets in self.critic.Q:
            qnets.to(self.device)
        for qnets in self.critic_target.Q:
            qnets.to(self.device)

        # copy parameters of the learning network to the target network
        hard_update(self.critic_target, self.critic)
        # disable gradient calculations of the target network
        grad_false(self.critic_target)

        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.q_optims = [Adam(q_network.parameters(), lr=lr) for q_network in self.critic.Q]

        if entropy_tuning:
            # Target entropy is -|A|.
            self.target_entropy = -torch.prod(torch.Tensor(
                self.env.action_space.shape).to(self.device)).item()
            # We optimize log(alpha), instead of alpha.
            self.log_alpha = torch.zeros(
                1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=lr)
        else:
            # fixed alpha
            self.alpha = torch.tensor(ent_coef).to(self.device)

        if per:
            # replay memory with prioritied experience replay
            # See https://github.com/ku2482/rltorch/blob/master/rltorch/memory
            self.memory = PrioritizedMemory(
                memory_size, self.env.observation_space.shape,
                self.env.action_space.shape, self.device, gamma, multi_step,
                alpha=alpha, beta=beta, beta_annealing=beta_annealing)
        else:
            # replay memory without prioritied experience replay
            # See https://github.com/ku2482/rltorch/blob/master/rltorch/memory
            self.memory = MultiStepMemory(
                memory_size, self.env.observation_space.shape,
                self.env.action_space.shape, self.device, gamma, multi_step)

        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.writer = wandb_writer
        self.train_rewards = RunningMeanStats(log_interval)

        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.num_steps = num_steps
        self.tau = tau
        self.per = per
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.gamma_n = gamma ** multi_step
        self.entropy_tuning = entropy_tuning
        self.grad_clip = grad_clip
        self.updates_per_step = updates_per_step
        self.log_interval = log_interval
        self.target_update_interval = target_update_interval
        self.eval_interval = eval_interval
        self.es=None
        if es:
            self.es = EarlyStopping(patience=es_patience, delta=es_delta, decreasing_is_better=True)
        self.stop = False

    def run(self):
        while True:
            self.train_episode()
            if (self.steps > self.num_steps) or self.stop:
                if self.stop:
                    print(f"Terminated SAC Training due to early stopping! Step {self.steps}")
                self.stop = False
                self.steps = 0
                break

    def is_update(self):
        return len(self.memory) > self.batch_size and\
            self.steps >= self.start_steps
    
    def update_env(self, env):
        self.env = env

    def act(self, state):
        if self.start_steps > self.steps:
            action = self.env.action_space.sample()
        else:
            action = self.explore(state)
        return action

    def explore(self, state):
        # act with randomness
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, _ = self.policy.sample(state)
        return action.cpu().numpy().reshape(-1)

    def exploit(self, state):
        # act without randomness
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, action = self.policy.sample(state)
        return action.cpu().numpy().reshape(-1)

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        return self.critic(states, actions)

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions, next_entropies, _ = self.policy.sample(next_states)
            qvals = self.critic_target(next_states, next_actions)
            qval = qvals[0]
            for q in qvals:
                qval = torch.min(qval,q)
            next_q = qval + self.alpha * next_entropies

        target_q = rewards + (1.0 - dones) * self.gamma_n * next_q

        return target_q

    def train_episode(self):
        self.episodes += 1
        episode_reward = 0.
        episode_steps = 0
        done = False
        state = self.env.reset()
        start_time = time.time()
        while not done:
            action = self.act(state)
            next_state, reward, done, _ = self.env.step(action)
            self.steps += 1
            episode_steps += 1
            episode_reward += reward

            # ignore done if the agent reach time horizons
            # (set done=True only when the agent fails)
            if episode_steps == self.env._max_episode_steps:
                masked_done = False
            else:
                masked_done = done

            if self.per:
                batch = to_batch(
                    state, action, reward, next_state, masked_done,
                    self.device)
                with torch.no_grad():
                    qvals = self.calc_current_q(*batch)
                target_q = self.calc_target_q(*batch)
                error = torch.abs(qvals[0] - target_q).item()
                # We need to give true done signal with addition to masked done
                # signal to calculate multi-step rewards.
                self.memory.append(
                    state, action, reward, next_state, masked_done, error,
                    episode_done=done)
            else:
                # We need to give true done signal with addition to masked done
                # signal to calculate multi-step rewards.
                self.memory.append(
                    state, action, reward, next_state, masked_done,
                    episode_done=done)

            if self.is_update():
                for _ in range(self.updates_per_step):
                    self.learn()

            if self.steps % self.eval_interval == 0:
                self.evaluate()
                self.save_models()

            state = next_state

        # We log running mean of training rewards.
        self.train_rewards.append(episode_reward)

        if self.episodes % self.log_interval == 0:
            self.writer.log({
                'train reward': self.train_rewards.get()})

        print(f'episode: {self.episodes:<4}  '
              f'eps steps: {episode_steps:<4}  '
              f'rew: {episode_reward:<5.1f}  '
              f'time: {time.time() - start_time:<5.3}s')

    def learn(self):
        self.learning_steps += 1
        if self.learning_steps % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        if self.per:
            # batch with indices and priority weights
            batch, indices, weights = \
                self.memory.sample(self.batch_size)
        else:
            batch = self.memory.sample(self.batch_size)
            # set priority weights to 1 when we don't use PER.
            weights = 1.

        q_losses, errors, mean_qs =\
            self.calc_critic_loss(batch, weights)
        policy_loss, entropies = self.calc_policy_loss(batch, weights)

        for i, q_optim in enumerate(self.q_optims):
            update_params(q_optim, self.critic.Q[i], q_losses[i], self.grad_clip)
        
        update_params(
            self.policy_optim, self.policy, policy_loss, self.grad_clip)

        if self.entropy_tuning:
            entropy_loss = self.calc_entropy_loss(entropies, weights)
            update_params(self.alpha_optim, None, entropy_loss)
            self.alpha = self.log_alpha.exp()
            self.writer.log({
                'alpha loss': entropy_loss.detach().item()})

        if self.per:
            # update priority weights
            self.memory.update_priority(indices, errors.cpu().numpy())

        if self.learning_steps % self.log_interval == 0:
            self.writer.log({
                'Mean Q loss': np.mean([q_loss.detach().item() for q_loss in q_losses]),
                'Policy loss': policy_loss.detach().item(),
                'Alpha': self.alpha.detach().item(),
                'Mean Q Value': np.mean(mean_qs),
                'Entropy': entropies.detach().mean().item()})
            if self.es:
                self.stop = self.es(policy_loss.detach().item())

    def calc_critic_loss(self, batch, weights):
        qvals = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch)

        # TD errors for updating priority weights
        errors = torch.abs(qvals[0].detach() - target_q)
        # We log means of Q to monitor training.
        mean_qs = [qval.detach().mean().item() for qval in qvals]

        # Critic loss is mean squared TD errors with priority weights.
        q_losses = [torch.mean((qval - target_q).pow(2) * weights) for qval in qvals]
        return q_losses, errors, mean_qs

    def calc_policy_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch

        # We re-sample actions to calculate expectations of Q.
        sampled_action, entropy, _ = self.policy.sample(states)
        # expectations of Q with clipped double Q technique
        qvals= self.critic(states, sampled_action)
        q = qvals[0]
        for qval in qvals:
            q = torch.min(q,qval)

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        policy_loss = torch.mean((- q - self.alpha * entropy) * weights)
        return policy_loss, entropy

    def calc_entropy_loss(self, entropy, weights):
        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropy).detach()
            * weights)
        return entropy_loss

    def evaluate(self):
        episodes = 10
        returns = np.zeros((episodes,), dtype=np.float32)

        for i in range(episodes):
            state = self.env.reset()
            episode_reward = 0.
            done = False
            while not done:
                action = self.exploit(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
            returns[i] = episode_reward

        mean_return = np.mean(returns)

        self.writer.log({
            'Test Reward': mean_return})
        print('-' * 60)
        print(f'Num steps: {self.steps:<5}  '
              f'reward: {mean_return:<5.1f}')
        print('-' * 60)

    def save_models(self):
        print(f"Saving SAC Models to directory {self.model_dir}")
        self.policy.save(os.path.join(self.model_dir, 'policy.pth'))
        self.critic.save(os.path.join(self.model_dir, 'critic.pth'))
        self.critic_target.save(
            os.path.join(self.model_dir, 'critic_target.pth'))

    def load_models(self, model_dir):
        print(f"Loading SAC Models from directory {model_dir}")
        self.policy.load(os.path.join(model_dir, 'policy.pth'))
        self.critic.load(os.path.join(model_dir, 'critic.pth'))
        self.critic_target.load(
            os.path.join(model_dir, 'critic_target.pth'))

    def __del__(self):
        self.env.close()
