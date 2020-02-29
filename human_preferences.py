from torch import nn
import numpy as np
import datetime
import torch
import cv2
import gym
import os
from time import sleep
import pathlib
import matplotlib.pyplot as plt


class HumanPref(nn.Module):
    def __init__(self, obs_size, neuron_size=64):
        super(HumanPref, self).__init__()

        self.obs_size = obs_size
        self.neuron_size = neuron_size

        self.dense1 = nn.Linear(self.obs_size, self.neuron_size)
        self.dense2 = nn.Linear(self.neuron_size, 1)

        self.batch_norm = nn.BatchNorm1d(1)

    def forward(self, x1, x2=None):

        model1_couche1 = self.dense1(x1)
        model1_couche2 = torch.nn.functional.relu(model1_couche1)
        model1_couche3 = self.dense2(model1_couche2)
        model1_couche4 = self.batch_norm(model1_couche3)
        if x2 is None:
            return model1_couche4
        else:
            model2_couche1 = self.dense1(x2)
            model2_couche2 = torch.nn.functional.relu(model2_couche1)
            model2_couche3 = self.dense2(model2_couche2)
            model2_couche4 = self.batch_norm(model2_couche3)
            # output = nn.functional.softmax(torch.stack([model1_couche4, model2_couche4]), dim=0)
            p1_sum = torch.exp(torch.sum(model1_couche1)/len(x1))
            p2_sum = torch.exp(torch.sum(model2_couche4)/len(x2))
            p1 = p1_sum/torch.add(p1_sum, p2_sum)
            p2 = p2_sum / torch.add(p1_sum, p2_sum)
            return torch.stack([p1, p2])


class HumanPreference(object):
    def __init__(self, obs_size, action_size):
        self.trijectories = []
        self.preferences = []
        self.layer_count = 3
        self.neuron_size_init = 64
        self.batch_size_init = 10
        self.learning_rate = 0.00025
        self.obs_size = obs_size
        self.action_size = action_size
        self.neuron_size = obs_size ** 3

        self.loss_l = []

        self.create_model()

    def create_model(self):
        self.model = HumanPref(self.obs_size, self.neuron_size)
        self.criterion = nn.functional.binary_cross_entropy
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        self.model.train()
        if len(self.preferences) < 5:
            return

        batch_size = min(len(self.preferences), self.batch_size_init)
        r = np.asarray(range(len(self.preferences)))
        np.random.shuffle(r)

        min_loss = 1e+10
        max_loss = -1e+10
        lo = 0.0
        for i in r[:batch_size]:
            x0, x1, preference = self.preferences[i]

            pref_dist = np.zeros([2], dtype=np.float32)
            if preference < 2:
                pref_dist[preference] = 1.0
            else:
                pref_dist[:] = 0.5

            x0 = torch.from_numpy(np.asarray(x0)).float()
            x1 = torch.from_numpy(np.asarray(x1)).float()
            y = torch.from_numpy(pref_dist)
            y_hat = self.model(x0, x1)

            loss = self.criterion(y_hat, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if loss.item() > max_loss:
                max_loss = loss.item()
            elif loss.item() < min_loss:
                min_loss = loss.item()

            lo = loss.item()
        print("[ Loss: actual loss =", lo, " max =", max_loss, " min =", min_loss, "]")

        self.loss_l.append(lo)

    def predict(self, obs):
        self.model.eval()
        obs = torch.tensor([obs]).float()
        pred = self.model(obs)
        return pred.detach().numpy()

    def add_preference(self, o0, o1, preference):
        self.preferences.append([o0, o1, preference])

    def add_trijactory(self, trijectory_env_name,  trijectory):
        self.trijectories.append([trijectory_env_name, trijectory])

    def ask_human(self):

        if len(self.trijectories) < 2:
            return

        r = np.asarray(range(len(self.trijectories)))
        np.random.shuffle(r)
        t = [self.trijectories[r[0]], self.trijectories[r[1]]]

        envs = []
        for i in range(len(t)):
            env_name, trijectory = t[i]
            env = gym.make(env_name)
            env.reset()
            env.render()
            envs.append(env)

        cv2.imshow("", np.zeros([1, 1], dtype=np.uint8))

        print("Preference (1,2|3):")
        env_idxs = np.zeros([2], dtype=np.int32)
        preference = -1
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('1'):
                preference = 0
            elif key == ord('2'):
                preference = 1
            elif key == ord('3') or key == ord('0'):
                preference = 2

            if preference != -1:
                break

            for i in range(len(t)):
                envs[i].render()

                env_name, trijectory = t[i]
                obs, future_obs, action, done = trijectory[env_idxs[i]]
                envs[i].step(action)
                env_idxs[i] += 1
                if done or env_idxs[i] >= len(trijectory):
                    envs[i].reset()
                    env_idxs[i] = 0
            sleep(0.02)

        if preference != -1:
            os = []
            for i in range(len(t)):
                env_name, trijectory = t[i]
                o = []

                for j in range(len(trijectory)):
                    o.append(trijectory[j][1])

                os.append(o)

            self.add_preference(os[0], os[1], preference)

        cv2.destroyAllWindows()
        for i in range(len(envs)):
            envs[i].close()

        if preference == 0:
            print(1)
        elif preference == 1:
            print(2)
        elif preference != -1:
            print("neutral")
        else:
            print("no oppinion")


    def plot(self):
        x = np.arange(0, len(self.loss_l))
        y = np.asarray(self.loss_l)
        fig, ax = plt.subplots()
        ax.plot(y)
        ax.set_xlabel('epochs')
        ax.set_ylabel('loss')
        ax.set_title('Loss per epochs')

        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join(pathlib.Path().absolute(), 'plots', 'hp_model', 'hp_model' + datetime_str + ".png")
        plt.savefig(path)
