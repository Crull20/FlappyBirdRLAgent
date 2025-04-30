# FlappyBirdRLAgent

## Before You Start
Python & Pytorch are required. Make sure both are installed.

https://www.python.org/downloads/

https://pytorch.org/

Additionally, tensorflow & pyyaml are required.

    pip install tensorflow

    pip install pyyaml

## Environment Installation
To install `flappy-bird-gymnasium`, run the following command:

    pip install flappy-bird-gymnasium

After the environment is installed, you may test the environment with the command:

    flappy-bird-gymnasium

After testing the environment runs, ensure that your preferred IDE is using the created environment as a python interpreter.

## Training the Agent
To train an agent, run one of the following commands:

    python flappyAgent.py testbird --train

or

    python flappyAgent.py adaptbird --train

Each command will train a bird according to the given hyperparameters in file `hyperparameters.yml`

## Running the Agent
After the agent has trained (recommended time = `1 hour`), run the following command corresponding
to the bird you trained (`testbird` or `adaptbird`)

    python flappyAgent.py testbird

or

    python flappyAgent.py adaptbird

Now watch as the agent plays Flappy Bird

