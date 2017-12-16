# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import opensim as osim
import sys

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
#from rl.random import OrnsteinUhlenbeckProcess

from osim.env import *
from osim.http.client import Client

import argparse
import math





parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--steps', dest='EPISODES', action='store', default=10000, type=int)
parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
args = parser.parse_args()


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return env.action_space.sample()
        act_values = self.model.predict(state)
        return act_values[0]  # returns action

        # if np.random.rand() <= self.epsilon:
        #     a=[]
        #     for i in xrange(self.action_size):
        #         a.append(random.randrange(2))
        #     return a
        # act_values = self.model.predict(state)
        # a=[]
        # for i in act_values[0]:
        #     if i < 0:
        #         a.append(0)
        #     else:
        #         a.append(1)
        # return a  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            # target = self.model.predict(state) original
            # target = self.model.predict(next_state)[0][0:18]
            target = self.model.predict(state)[0][0:18]
            # print(target)
            if not done:
                # target = sorted(self.model.predict(next_state)[0], reverse=True)
                # target = target[0:18]
                # for ele in target:
                #     ele = reward + self.gamma * ele
                a = self.model.predict(next_state)[0] #41 vector
                t = self.target_model.predict(next_state)[0] #41 vector
                _temp= reward + self.gamma * t[np.argmax(a)] #single value
                #approach 1) replace all the 18 values with the max reward
                #approach 2) replace one action with the max reward but which one?
                q=[]
                # print("temp>>",_temp,target[0])
                el=0
                for el in xrange(18):
                    if target[el] < 0:
                        q.append(_temp)
                    else:
                        q.append(target[el])
                _z=np.asarray(q)
                target =np.array([q])
            else:
                target = np.array([target])

                
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    #env = gym.make('CartPole-v1')
    env = RunEnv(args.visualize)

    state_size = env.observation_space.shape[0];
    #env.observation_space.shape
    action_size = env.action_space.shape[0]
    #print(state_size, action_size)
    agent = DQNAgent(state_size, action_size)
    # agent.load("../models/human-ddqn.h5f")
    done = False
    batch_size = 32

    for e in range(args.EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            env.render()
            action = agent.act(state);#print("action: ",type(action),action);print("ENV.STEP: ",env.step(action))
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_model()
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, args.EPISODES, time , agent.epsilon))
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        # if e % 10 == 0:
            # agent.save("../models/human-ddqn.h5f")
