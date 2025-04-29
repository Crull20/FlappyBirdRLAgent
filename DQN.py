import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, dueling_dqn=True):
        super(DQN, self).__init__()

        self.dueling_dqn = dueling_dqn

        self.fc1 = nn.Linear(input_dim, hidden_dim)

        if self.dueling_dqn:
            self.fc_value = nn.Linear(hidden_dim, 256)
            self.value = nn.Linear(256, 1)
            self.fc_advantage = nn.Linear(hidden_dim, 256)
            self.advantage = nn.Linear(256, output_dim)
        else:
            self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        if self.dueling_dqn:
            val = F.relu(self.fc_value(x))
            val = self.value(val)
            adv = F.relu(self.fc_advantage(x))
            adv = self.advantage(adv)
            q_value = val + (adv - adv.mean(1, keepdim=True))

        else:
            q_value = self.output(x)

        return q_value

if __name__ == "__main__":
    input_dim=12
    output_dim=2
    model = DQN(input_dim, output_dim)
    state = torch.randn(1, input_dim)
    q_values = model(state)
    print(q_values)