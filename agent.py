import gymnasium as gym
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import random
import torch
from torch import nn
import yaml

from experience_replay import ReplayMemory
from dqn import DQN

from datetime import datetime, timedelta
import argparse
import itertools

import flappy_bird_gymnasium
import os

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu' # force cpu, sometimes GPU not always faster than CPU due to overhead of moving data to GPU

# lots of this is just set up for the agent, the env and wtv

# Deep Q-Learning Agent
class Agent():

    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
            # print(hyperparameters)

        self.hyperparameter_set = hyperparameter_set

        # Hyperparameters (adjustable)
        self.env_id             = hyperparameters['env_id']
        self.learning_rate_a    = hyperparameters['learning_rate_a']
        self.discount_factor_g  = hyperparameters['discount_factor_g']
        self.network_sync_rate  = hyperparameters['network_sync_rate']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size    = hyperparameters['mini_batch_size']

        # Epsilon parameters - directly from hyperparameters
        self.epsilon_init  = hyperparameters['epsilon_init']
        self.epsilon_min   = hyperparameters['epsilon_min']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_decayf = 0.997   # You can hardcode this if you want faster decay after success
        self.epsilon_decays = 1.1   # Hardcoded slow decay after bad performance

        self.rewardtreshH = 10000    # You might want to hardcode this if you don't have reward thresholds in YAML
        self.rewardtreshL = 1000     # Hardcoded low reward threshold

        self.stop_on_reward     = hyperparameters['stop_on_reward']
        self.fc1_nodes          = hyperparameters['fc1_nodes']
        self.env_make_params    = hyperparameters.get('env_make_params', {})
        self.enable_double_dqn  = hyperparameters['enable_double_dqn']
        self.enable_dueling_dqn = hyperparameters['enable_dueling_dqn']

        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')
        self.loss_fn = nn.MSELoss()

    def adaptive_epsilon(self, epsilon, rewards_per_episode):
        if len(rewards_per_episode) >= 300:
            avg_reward = np.mean(rewards_per_episode[-100:])

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
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        # Create instance of the environment.
        # Use "**self.env_make_params" to pass in environment-specific parameters from hyperparameters.yml.
        env = gym.make(self.env_id, render_mode='human' if render else None, **self.env_make_params)
        #actual flappy bird instance

        # Number of possible actions
        num_actions = env.action_space.n

        # Get observation space size
        num_states = env.observation_space.shape[0] # Expecting type: Box(low, high, (shape0,), float64)

        # List to keep track of rewards collected per episode.
        rewards_per_episode = []

        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(device)

        if is_training:
            # Initialize epsilon
            if os.path.exists(self.MODEL_FILE):
                print(f"Loading existing model weights from {self.MODEL_FILE}...")
                policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))

            epsilon = self.epsilon_init

            # Initialize replay memory
            memory = ReplayMemory(self.replay_memory_size)

            # Create the target network and make it identical to the policy network
            target_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            # Policy network optimizer. "Adam" optimizer can be swapped to something else.
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

            # List to keep track of epsilon decay
            epsilon_history = []

            # Track number of steps taken. Used for syncing policy => target network.
            step_count=0

            # Track best reward
            best_reward = -9999999
        else:
            # Load learned policy
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))

            # switch model to evaluation mode
            policy_dqn.eval()

        # Train INDEFINITELY, manually stop the run when you are satisfied (or unsatisfied) with the results
        for episode in itertools.count():

            state, _ = env.reset()  # Initialize environment. Reset returns (state,info).
            state = torch.tensor(state, dtype=torch.float, device=device) # Convert state to tensor directly on device

            terminated = False      # True when agent reaches goal or fails
            episode_reward = 0.0    # Used to accumulate rewards per episode

            # Perform actions until episode terminates or reaches max rewards
            # (on some envs, it is possible for the agent to train to a point where it NEVER terminates, so stop on reward is necessary)
            while(not terminated and episode_reward < self.stop_on_reward):

                # Select action based on epsilon-greedy
                if is_training and random.random() < epsilon:
                    # select random action
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    # select best action
                    with torch.no_grad():
                        # state.unsqueeze(dim=0): Pytorch expects a batch layer, so add batch dimension i.e. tensor([1, 2, 3]) unsqueezes to tensor([[1, 2, 3]])
                        # policy_dqn returns tensor([[1], [2], [3]]), so squeeze it to tensor([1, 2, 3]).
                        # argmax finds the index of the largest element.
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Execute action. Truncated and info is not used.
                new_state,reward,terminated,truncated,info = env.step(action.item())

                # Accumulate rewards
                episode_reward += reward

                # Convert new state and reward to tensors on device
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    # Save experience into memory
                    memory.append((state, action, new_state, reward, terminated))

                    # Increment step counter
                    step_count+=1

                # Move to the next state
                state = new_state

            # Keep track of the rewards collected per episode.
            rewards_per_episode.append(episode_reward)

            # Save model when new best reward is obtained.
            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward


                # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                # If enough experience has been collected
                if len(memory)>self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    # Decay epsilon
                    epsilon = self.adaptive_epsilon(epsilon, rewards_per_episode)
                    epsilon_history.append(epsilon)

                    # Copy policy network to target network after a certain number of steps
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count=0


    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)


    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Transpose the list of experiences and separate each element
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # Stack tensors to create batch tensors
        # tensor([[1,2,3]])
        states = torch.stack(states)

        actions = torch.stack(actions)

        new_states = torch.stack(new_states)

        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            if self.enable_double_dqn:
                best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)

                target_q = rewards + (1-terminations) * self.discount_factor_g * \
                                target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
            else:
                # Calculate target Q values (expected returns)
                target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]
                '''
                    target_dqn(new_states)  ==> tensor([[1,2,3],[4,5,6]])
                        .max(dim=1)         ==> torch.return_types.max(values=tensor([3,6]), indices=tensor([3, 0, 0, 1]))
                            [0]             ==> tensor([3,6])
                '''

        # Calcuate Q values from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        '''
            policy_dqn(states)  ==> tensor([[1,2,3],[4,5,6]])
                actions.unsqueeze(dim=1)
                .gather(1, actions.unsqueeze(dim=1))  ==>
                    .squeeze()                    ==>
        '''

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model (backpropagation)
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Compute gradients
        self.optimizer.step()       # Update network parameters i.e. weights and biases

if __name__ == '__main__':
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)
