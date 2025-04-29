import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
import torch
from memory import Memory
import yaml
import random
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import argparse

RUNS = 'runs'
os.makedirs(RUNS, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FlappyAgent:
    def __init__(self, hyperparameters_set):
        # Load hyperparameters from YAML file
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
        hyperparameters = all_hyperparameter_sets[hyperparameters_set]
        self.hyperparameters_set = hyperparameters_set

        self.env_id = hyperparameters['env_id']
        self.learning_rate = hyperparameters['learning_rate']
        self.discount_factor = hyperparameters['discount_factor']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.memory_size = hyperparameters['memory_size']
        self.batch_size = hyperparameters['batch_size']
        self.epsilon_initial = hyperparameters['epsilon_initial']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.epsilon_decayf = 0.997  #fast decay for when good performance
        self.epsilon_decays = 1.1  #slow decay after bad performance

        self.rewardtreshH = 100  #high and low reward tresholds
        self.rewardtreshL = 3

        self.stop_training_at_reward = hyperparameters['stop_training_at_reward']
        self.fc1_units = hyperparameters['fc1_units']
        self.env_make_params = hyperparameters.get('env_make_params', {})
        self.double_dqn = hyperparameters['double_dqn']
        self.dueling_dqn = hyperparameters['dueling_dqn']

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = None

        self.LOG = os.path.join(RUNS, f'{self.hyperparameters_set}.log')
        self.MODEL = os.path.join(RUNS, f'{self.hyperparameters_set}.pt')
        self.PLOT = os.path.join(RUNS, f'{self.hyperparameters_set}.png')

    def adaptive_epsilon(self, epsilon, rewards_per_episode):
        if len(rewards_per_episode) >= 50:  # for first 300 runs, it will experiment more and try to explore possible strategies
            avg_reward = np.mean(rewards_per_episode[-100:])

            # Adjust thresholds based on maximum reward, but with sensible minimums
            max_reward = max(rewards_per_episode)
            self.rewardtreshH = max(100, max_reward * 0.5)  # At least 100, or 50% of max
            self.rewardtreshL = max(10, max_reward * 0.1)  # At least 10, or 10% of max

            if avg_reward > self.rewardtreshH:
                epsilon = max(epsilon * self.epsilon_decayf, self.epsilon_min)
            elif avg_reward < self.rewardtreshL:
                epsilon = min(epsilon / self.epsilon_decays, 1.0)
            else:
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
        else:
            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

        return epsilon

    def run(self, is_training=True, render=False):
        if is_training:
            start_time = datetime.now()
            last_plot = start_time
            log_msg = f'{start_time.strftime("%m-%d %H:%M:%S")}: Training started\n'
            print(log_msg)
            with open(self.LOG, 'w') as log_file:
                log_file.write(log_msg + '\n')

        # Create the environment
        env = gymnasium.make(self.env_id, render_mode="human"
        if render else None, **self.env_make_params)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        reward_per_episode = []

        policy = DQN(num_states, num_actions, self.fc1_units, self.dueling_dqn).to(device)

        if is_training:
            memory = ReplayMemory(self.memory_size)

            epsilon = self.epsilon_initial

            target = DQN(num_states, num_actions, self.fc1_units).to(device)
            target.load_state_dict(policy.state_dict())

            step_counter = 0

            self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.learning_rate)

            epsilon_history = []

            best_reward = float('-inf')
        else:
            policy.load_state_dict(torch.load(self.MODEL))

            policy.eval()

        for episode in range(100000):  # itertools.count() for infinite loop until interrupted
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32).to(device)
            episode_reward = 0.0
            while True:

                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        action = policy(state.unsqueeze(0)).squeeze().argmax()

                # Processing:
                new_state, reward, terminated, _, info = env.step(action.item())

                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    memory.add((state, action, new_state, reward, terminated))

                    step_counter += 1

                state = new_state

                # Checking if the player is still alive
                if terminated:
                    break

            reward_per_episode.append(episode_reward)

            if is_training:
                if episode_reward > best_reward:
                    log_msg = f'{datetime.now().strftime("%m-%d %H:%M:%S")}: New best reward: {episode_reward:.2f} ({(episode_reward - best_reward) / best_reward * 100:+.2f}%) at episode {episode}, saving model\n'
                    print(log_msg)
                    with open(self.LOG, 'a') as log_file:
                        log_file.write(log_msg + '\n')

                    torch.save(policy.state_dict(), self.MODEL)
                    best_reward = episode_reward

                current_time = datetime.now()
                if (current_time - last_plot) > timedelta(seconds=10):
                    self.save_plot(reward_per_episode, epsilon_history)
                    last_plot = current_time

                if len(memory) > self.batch_size:
                    batch = memory.sample(self.batch_size)
                    self.optimize_model(batch, policy, target)

                    epsilon = self.adaptive_epsilon(epsilon, reward_per_episode)
                    epsilon_history.append(epsilon)

                    if step_counter > self.network_sync_rate:
                        target.load_state_dict(policy.state_dict())
                        step_counter = 0

        env.close()  # Close the environment if trained definitively

    def save_plot(self, reward_per_episode, epsilon_history):
        fig = plt.figure(1)

        mean_reward = np.zeros(len(reward_per_episode))
        for i in range(len(mean_reward)):
            mean_reward[i] = np.mean(reward_per_episode[max(0, i - 99):i + 1])
        plt.subplot(121)
        plt.ylabel('Mean Reward')
        plt.plot(mean_reward)

        plt.subplot(122)
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        fig.savefig(self.PLOT)
        plt.close(fig)

    def optimize_model(self, batch, policy, target):
        states, actions, new_states, rewards, terminations = zip(*batch)
        states = torch.stack(states).to(device)
        actions = torch.stack(actions).to(device)
        new_states = torch.stack(new_states).to(device)
        rewards = torch.stack(rewards).to(device)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            if self.double_dqn:
                best_actions = policy(new_states).argmax(1)
                target_q = (rewards + (1 - terminations) * self.discount_factor
                            * target(new_states).gather(1, best_actions.unsqueeze(1)).squeeze())
            else:
                target_q = (rewards + (1 - terminations) * self.discount_factor
                            * target(new_states).max(1)[0])

        current_q = policy(states).gather(1, actions.unsqueeze(1)).squeeze()

        loss = self.loss_function(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Flappy Bird DQN Agent')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training Mode', action='store_true')
    args = parser.parse_args()

    dql = FlappyAgent(args.hyperparameters)
    if args.train:
        dql.run(is_training=True, render=False)

    else:
        dql.run(is_training=False, render=True)

