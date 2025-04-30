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

# Directory where logs, models, and plots for runs will be stored
RUNS = 'runs'
os.makedirs(RUNS, exist_ok=True) # Ensure runs directory exists, if not, create one

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FlappyAgent:
    """
    DQN-based agent for the Flappy Bird environment. Has enhancements with dueling DQN and
    double DQN. Supports adaptive epsilon decay, checkpointing to save best models, and
    periodic plotting of training metrics
    """
    def __init__(self, hyperparameters_set):
        # load hyperparameters from YAML file
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
        # select desired hyperparameter set by key
        hyperparameters = all_hyperparameter_sets[hyperparameters_set]
        self.hyperparameters_set = hyperparameters_set

        # unpack hyperparameters for readability
        self.env_id = hyperparameters['env_id']                         # gym environment identifier
        self.learning_rate = hyperparameters['learning_rate']           # learning rate for the optimizer
        self.discount_factor = hyperparameters['discount_factor']       # gamma in Q-learning update
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.memory_size = hyperparameters['memory_size']               # replay buffer capacity
        self.batch_size = hyperparameters['batch_size']                 # minibatch size for training
        self.epsilon_initial = hyperparameters['epsilon_initial']       # initial epsilon for exploration
        self.epsilon_decay = hyperparameters['epsilon_decay']           # multiplicative decay rate each update
        self.epsilon_min = hyperparameters['epsilon_min']               # lower bound of epsilon
        # additional parameters for adaptive epsilon scheduling
        self.epsilon_decayf = 0.997  # fast decay for when good performance
        self.epsilon_decays = 1.1  # slow decay after bad performance

        # reward thresholds for exploration vs exploitation
        self.rewardtreshH = 100  # high and low reward thresholds
        self.rewardtreshL = 3  

        self.stop_training_at_reward = hyperparameters['stop_training_at_reward']
        self.fc1_units = hyperparameters['fc1_units']                   # width of hidden layer
        self.env_make_params = hyperparameters.get('env_make_params', {})
        # flags toggling double DQN and dueling DQN
        self.double_dqn = hyperparameters['double_dqn']
        self.dueling_dqn = hyperparameters['dueling_dqn']

        # loss for Q-value regression
        self.loss_function = torch.nn.MSELoss()
        self.optimizer = None   # gets initialized during training

        # file paths for logging, model checkpoint, plots
        self.LOG = os.path.join(RUNS, f'{self.hyperparameters_set}.log')
        self.MODEL = os.path.join(RUNS, f'{self.hyperparameters_set}.pt')
        self.PLOT = os.path.join(RUNS, f'{self.hyperparameters_set}.png')

    def adaptive_epsilon(self, epsilon, rewards_per_episode):
        """
        adjust epsilon based on recent performance - balance exploration vs exploitation
        - performance is high, decrease epsilon faster toward epsilon_min
        - performance is low, increase epsilon (encourage exploration)
        - else, follow baseline decay
        """
        if len(rewards_per_episode) >= 50:  # for first 300 runs, it will experiment more and try to explore possible strategies
            # average reward over last 100 episodes
            avg_reward = np.mean(rewards_per_episode[-100:])

            # Adjust thresholds based on maximum reward, but with sensible minimums
            max_reward = max(rewards_per_episode)
            self.rewardtreshH = max(100, max_reward * 0.5)  # At least 100, or 50% of max
            self.rewardtreshL = max(10, max_reward * 0.1)  # At least 10, or 10% of max

            if avg_reward > self.rewardtreshH:
                # if above high threshold, decay epsilon faster (greedy)
                epsilon = max(epsilon * self.epsilon_decayf, self.epsilon_min)
            elif avg_reward < self.rewardtreshL:
                # if below low threshold, increase epsilon (exploration)
                epsilon = min(epsilon / self.epsilon_decays, 1.0)
            else:
                # standard decay
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
        else:
            # early training; use baseline decay
            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

        return epsilon

    def run(self, is_training=True, render=False):
        """
        main loop to train or evaluate agent
        - is_training=True: runs training loop with replay buffer
        - is_training = False: loads saved model and plays environment
        """
        if is_training:
            # timestamp for logging/plotting frequency control
            start_time = datetime.now()
            last_plot = start_time
            log_msg = f'{start_time.strftime("%m-%d %H:%M:%S")}: Training started\n'
            print(log_msg)
            # initialize log file
            with open(self.LOG, 'w') as log_file:
                log_file.write(log_msg + '\n')

        # Create the environment
        env = gymnasium.make(self.env_id, render_mode="human"
        if render else None, **self.env_make_params)

        # determine dimensions of state spapce and number of actions
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        reward_per_episode = []     # track episodic returns

        # policy network
        policy = DQN(num_states, num_actions, self.fc1_units, self.dueling_dqn).to(device)

        if is_training:
            # experience replay buffer
            memory = ReplayMemory(self.memory_size)

            epsilon = self.epsilon_initial

            # target network for updates, initially synced with policy
            target = DQN(num_states, num_actions, self.fc1_units).to(device)
            target.load_state_dict(policy.state_dict())

            step_counter = 0        # count steps to decide when to sync networks

            # optimizer for policy network parameters
            self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.learning_rate)

            epsilon_history = []    # record epsilon over time

            best_reward = float('-inf') # track highest episodic return
        else:
            # evaluation mode: load best saved model
            policy.load_state_dict(torch.load(self.MODEL))

            policy.eval()

        # loop over episodes
        for episode in range(100000):  # itertools.count() for infinite loop until interrupted
            # reset env at start of episode
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32).to(device)
            episode_reward = 0.0

            # run steps until terminated
            while True:

                # epsilon-greedy action selection
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        # select action with highest q-value
                        action = policy(state.unsqueeze(0)).squeeze().argmax()

                # Processing:
                new_state, reward, terminated, _, info = env.step(action.item())

                episode_reward += reward

                # convert to tensors
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    # store transition in replay memory
                    memory.add((state, action, new_state, reward, terminated))

                    step_counter += 1

                state = new_state   # advance state

                # Checking if the player is still alive
                if terminated:
                    break   # episode ends

            reward_per_episode.append(episode_reward)

            if is_training:
                # save model if new best performance achieved
                if episode_reward > best_reward:
                    log_msg = f'{datetime.now().strftime("%m-%d %H:%M:%S")}: New best reward: {episode_reward:.2f} ({(episode_reward - best_reward) / best_reward * 100:+.2f}%) at episode {episode}, saving model\n'
                    print(log_msg)
                    with open(self.LOG, 'a') as log_file:
                        log_file.write(log_msg + '\n')

                    torch.save(policy.state_dict(), self.MODEL)
                    best_reward = episode_reward

                # periodically update live plot
                current_time = datetime.now()
                if (current_time - last_plot) > timedelta(seconds=10):
                    self.save_plot(reward_per_episode, epsilon_history)
                    last_plot = current_time

                # perform DQN update if enough experiences collected
                if len(memory) > self.batch_size:
                    batch = memory.sample(self.batch_size)
                    self.optimize_model(batch, policy, target)

                    # update epsilon based on performance
                    epsilon = self.adaptive_epsilon(epsilon, reward_per_episode)
                    epsilon_history.append(epsilon)

                    # sync target network periodically
                    if step_counter > self.network_sync_rate:
                        target.load_state_dict(policy.state_dict())
                        step_counter = 0

        env.close()  # Close the environment if trained definitively

    def save_plot(self, reward_per_episode, epsilon_history):
        """
        generate and save a plot with:
        - left: rolling mean of episodic rewards
        - right: epsilon decay history
        """
        fig = plt.figure(1)

        # compute rolling mean reward
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
        """
        perform single optimization step on the policy netwrok using sampled batch:
        - compute target Q-values from bellman update
        - computer current Q-values
        - minimise MSE loss between current and target Q-values
        """
        states, actions, new_states, rewards, terminations = zip(*batch)
        # stack to tensors and move to device
        states = torch.stack(states).to(device)
        actions = torch.stack(actions).to(device)
        new_states = torch.stack(new_states).to(device)
        rewards = torch.stack(rewards).to(device)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            if self.double_dqn:
                # double DQN: action selected by policy, evaluation by target net
                best_actions = policy(new_states).argmax(1)
                target_q = (rewards + (1 - terminations) * self.discount_factor
                            * target(new_states).gather(1, best_actions.unsqueeze(1)).squeeze())
            else:
                # standard DQN: max over target network outputs
                target_q = (rewards + (1 - terminations) * self.discount_factor
                            * target(new_states).max(1)[0])

        # q-values predicted by policy network for taken actions
        current_q = policy(states).gather(1, actions.unsqueeze(1)).squeeze()

        # compute MSE loss
        loss = self.loss_function(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    # CLI interface: specify hyperparameters set and whether to train
    parser = argparse.ArgumentParser(description='Flappy Bird DQN Agent')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training Mode', action='store_true')
    args = parser.parse_args()

    # instantiate agent
    dql = FlappyAgent(args.hyperparameters)
    if args.train:
        # run training loop without rendering
        dql.run(is_training=True, render=False)

    else:
        # run evaluation loop with rendering
        dql.run(is_training=False, render=True)
