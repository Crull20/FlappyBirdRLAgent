import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    DQN implementation with dueling architecture
    arguments:
        input_dim (int): dimension of input feature vector (state size)
        output_dim (int): number of actions (size of action space)
        hidden_dim (int): numberof units in the first hidden layer
        dueling_dqn (bool): use dueling network or not
    """
    def __init__(self, input_dim, output_dim, hidden_dim=256, dueling_dqn=True):
        super(DQN, self).__init__()

        # flag for whether or not to use dueling DQN architecture
        self.dueling_dqn = dueling_dqn

        # maps from state vector to hidden representation
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        if self.dueling_dqn:
            # dueling DQN, split into separate value and advantage streams

            # value stream: estimate value of being in this state
            # map hidden features to intermediate representation
            self.fc_value = nn.Linear(hidden_dim, 256)
            # final scalar state-value output V(s)
            self.value = nn.Linear(256, 1)

            # advantatge stream: estimates advantage of each action
            # map hidden features to an intermediate representation
            self.fc_advantage = nn.Linear(hidden_dim, 256)
            # final advantage outputs A(s, a) for each action
            self.advantage = nn.Linear(256, output_dim)
        else:
            # standard DQN: single head producing Q-values directly
            self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass through network

        arguments: 
            x: input batch of states with shape (batch_size, input_dim)
        returns:
            q_value: estimates with shape (batch_size, output_dim)
        """
        # apply fully connected layer + ReLu activation
        x = F.relu(self.fc1(x))

        if self.dueling_dqn:
            #--value stream--
            # apply intermediate FC + ReLU
            val = F.relu(self.fc_value(x))
            # output V(s), shape (batch_size, 1)
            val = self.value(val)

            #--advantage stream--
            # apply intermediate FC + ReLU
            adv = F.relu(self.fc_advantage(x))
            # Output A(s,a), shape (batch_size, output_dim)
            adv = self.advantage(adv)

            # combine streams into Q-values:
            # Q(s, a) = V(s) + (A(s, a) - mean_a A(s,a))
            q_value = val + (adv - adv.mean(1, keepdim=True))

        else:
            # standard DQN: direct Q-value outputs
            q_value = self.output(x)

        return q_value

if __name__ == "__main__":
    # example usage: instantiate network and pass random input
    input_dim=12        # observation vector length
    output_dim=2        # two discrete actions: flap/do nothing
    model = DQN(input_dim, output_dim)

    # dummy state (batch size 1)
    state = torch.randn(1, input_dim)
    # forward pass to get Q-value predictions
    q_values = model(state)
    print(q_values)        # tensor of shape (1,2) with Q-values for each action
