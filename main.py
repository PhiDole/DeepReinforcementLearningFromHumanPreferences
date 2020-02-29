import random
import gym
import numpy as np
from collections import deque
from human_preferences import HumanPreference
import os
import matplotlib.pyplot as plt
import datetime
import pathlib
import torch
from torch import nn

mode = "Human"  # "Human" pour avoir les préférences humaines sur les rewards, "self" pour avoir les rewards par défaut
ENV_NAME = "CartPole-v1"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

CONSECUTIVE_RUNS_TO_SOLVE = 100


class NN(nn.Module):
    def __init__(self, obs_size, action_size):
        super(NN, self).__init__()
        self.dense1 = nn.Linear(obs_size, 24)
        self.dense2 = nn.Linear(24, 24)
        self.dense3 = nn.Linear(24, 24)
        self.dense4 = nn.Linear(24, action_size)

    def forward(self, x):
        l1 = self.dense1(x)
        l2 = nn.functional.relu(l1)
        l3 = self.dense2(l2)
        l4 = nn.functional.relu(l3)
        l5 = self.dense3(l4)
        l6 = nn.functional.relu(l5)
        output = self.dense4(l6)
        return output


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.scores = deque(maxlen=CONSECUTIVE_RUNS_TO_SOLVE)

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.rl_model = NN(observation_space, action_space)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.rl_model.parameters(), lr=LEARNING_RATE)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.rl_model(torch.tensor(state).float()).detach().numpy()
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_pred = self.rl_model(torch.tensor(state_next).float()).detach().numpy()
                q_update = (reward + GAMMA * np.amax(q_pred))
            q_values = self.rl_model(torch.tensor(state).float()).detach().numpy()
            q_values[0][action] = q_update

            x = torch.from_numpy(state).float()
            y = torch.from_numpy(q_values).float()

            y_hat = self.rl_model(x)
            loss = self.criterion(y_hat, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()     # TODO
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    def add_score(self, score):
        self.scores.append(score)

    def plot_score(self, mode, episodes):

        data = np.array(self.scores)
        x = []
        y = []
        x_label = "runs"
        y_label = "scores"
        for i in range(0, len(data)):
            x.append(int(i))
            y.append(int(data[i]))

        plt.subplots()
        plt.plot(x, y, label="score per run")

        average_range = len(x)
        plt.plot(x[-average_range:], [np.mean(y[-average_range:])] * len(y[-average_range:]), linestyle="--",
                 label="last " + str(average_range) + " runs average")

        if mode == "Human":
            plt.axvline(x=episodes/2, label="start of Human preference")

        if len(x) > 1:
            trend_x = x[1:]
            z = np.polyfit(np.array(trend_x), np.array(y[1:]), 1)
            p = np.poly1d(z)
            plt.plot(trend_x, p(trend_x), linestyle="-.", label="trend")

        plt.title(ENV_NAME)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.legend(loc="upper left")

        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join(pathlib.Path().absolute(), 'plots', 'dqn', 'dqn_score_' + mode + '_' + datetime_str + ".png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()


def cartpole():
    env = gym.make(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    hp_model = HumanPreference(observation_space, action_space)
    run = 0
    episodes = 50
    for i in range(episodes):
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        trijectory = []
        while True:
            step += 1
            # env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            if mode == "self" or run < episodes / 2:
                reward = reward if not terminal else -reward
            else:
                reward = hp_model.predict(state_next)
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            trijectory.append([state, state_next, action, terminal])
            state = state_next
            if terminal:
                print(
                    "Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                hp_model.add_trijactory(ENV_NAME, trijectory)
                dqn_solver.add_score(step)
                break
            dqn_solver.experience_replay()

        if run % 5 == 0 and mode == "Human":
            hp_model.ask_human()
            hp_model.train()

    if mode == "Human":
        hp_model.plot()
    dqn_solver.plot_score(mode, episodes)


if __name__ == "__main__":
    cartpole()
