# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam, RMSprop

import opensim as osim
import sys

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
#from rl.random import OrnsteinUhlenbeckProcess

from osim.env import *
from osim.http.client import Client

import argparse
import math





# Command line parameters
parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--steps', dest='EPISODES', action='store', default=10000, type=int)
parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
args = parser.parse_args()





class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.5 # how much to explore minimum
        self.epsilon_decay = 0.995 
        self.learning_rate = 0.001
        self.model = self._build_model()
		

    def _build_model(self):
		# Neural Net for Deep-Q learning Model
		model = Sequential()
		#model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
		model.add(Dense(32, input_dim=self.state_size, activation='relu'))
		model.add(Dense(32, activation='relu'))
		model.add(Dense(32, activation='relu'))
		model.add(Dense(int(self.action_size), activation='relu'))
		model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
		model.summary()
		return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
		if np.random.rand() <= self.epsilon:
			return env.action_space.sample()
		act_values = self.model.predict(state)
		return act_values[0]  # returns action
		

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)[0][0:18]
            if not done:
            	target = sorted(self.model.predict(next_state)[0], reverse=True)
            	target = target[0:18]
            	for ele in target:
            		ele = reward + self.gamma * ele
            target_f = self.model.predict(state)
            target_f[0] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)


    def save(self, name):
		self.model.save_weights(name, overwrite=True)
		


if __name__ == "__main__":
# Load walking environment
	env = RunEnv(args.visualize)
	state_size = env.observation_space.shape[0];
	action_size = env.action_space.shape[0]
	agent = DQNAgent(state_size, action_size)
	#agent.load("../models/human-dqn.h5f")
	done = False
	batch_size = 32
for e in range(args.EPISODES):
	state = env.reset()
	state = np.reshape(state, [1, state_size])
	total_reward=0.0
	done = False
	score = 0
	while not done:
	# for time in range(200):
		env.render()
		action = agent.act(state)
		next_state, reward, done, _ = env.step(action)
		# reward = reward if not done else -10
		total_reward += reward
		next_state = np.reshape(next_state, [1, state_size])

		agent.remember(state, action, reward, next_state, done)
		state = next_state
		if done:
			print("episode: {}/{}, reward: {}, exploration_rate: {:.2}"
				  .format(e, args.EPISODES, total_reward, agent.epsilon))
			# print("episode: {}/{}, time_score: {},reward: {}, exploration_rate: {:.2}"
			# 	  .format(e, args.EPISODES, time, total_reward, agent.epsilon))
			break
	if len(agent.memory) > batch_size:
		agent.replay(batch_size)
	#if e % 10 == 0:
	    #agent.save("../models/human-dqn.h5f")
