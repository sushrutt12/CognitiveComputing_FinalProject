# -*- coding: utf-8 -*-

import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

import opensim as osim
import sys
from rl.policy import BoltzmannQPolicy
from rl.agents import SARSAAgent
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





class SARSAAgent:
    def __init__(self, state_size, action_size,policy):
        self.policy=policy
        self.state_size = state_size
        self.action_size = action_size
        #self.nb_steps_warmup=10
        self.gamma = 0.95    # discount rate
        self.learning_rate = 0.001
        self.model = self._build_model()
		

    def _build_model(self):
		# Neural Net for Deep-Q learning Model
		model = Sequential()
		model.add(Dense(24, input_dim=self.state_size, activation='relu'))
		model.add(Dense(24, activation='relu'))
		model.add(Dense(self.action_size, activation='linear'))
		#model.compile(Adam(lr=1e-3), metrics=['mae'])
		model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
		model.summary()
		return model

    #def remember(self, state, action, reward, next_state, done):
    #    self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
		# if np.random.rand() <= self.epsilon:
			a=[]
			for i in xrange(self.action_size):
				a.append(random.randrange(2))
			return a
		# act_values = self.model.predict(state)
		# a=[]
		# for i in act_values[0]:
		# 	if i < 0:
		# 		a.append(0)
		# 	else:
		# 		a.append(1)
		# return a  # returns action
		

    # def replay(self, batch_size):
    #     minibatch = random.sample(self.memory, batch_size)
    #     for state, action, reward, next_state, done in minibatch:
    #         target = reward
    #         if not done:
    #             target = (reward + self.gamma *
    #                       np.amax(self.model.predict(next_state)[0]))
    #         target_f = self.model.predict(state)
    #         target_f[0][action] = target
    #         self.model.fit(state, target_f, epochs=1, verbose=0)
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)


    def save(self, name):
		self.model.save_weights(name, overwrite=True)
		


if __name__ == "__main__":
# Load walking environment
	env = RunEnv(args.visualize)
	state_size = env.observation_space.shape[0];
	action_size = env.action_space.shape[0]
	policy = BoltzmannQPolicy()
	agent = SARSAAgent(state_size, action_size, policy)
	# agent.load("../models/human-dqn.h5f")
	done = False
	batch_size = 32
for e in range(args.EPISODES):
	state = env.reset()
	state = np.reshape(state, [1, state_size])
	for time in range(500):
		env.render()
		action = agent.act(state)
		next_state, reward, done, _ = env.step(action)
		next_state = np.reshape(next_state, [1, state_size])
		#agent.remember(state, action, reward, next_state, done)
		state = next_state
		if done:
			print("episode: {}/{}, score: {}, reward: {:.4}"
				  .format(e, args.EPISODES, time, reward))
			break
	# if len(agent.memory) > batch_size:
	# 	agent.replay(batch_size)
	# if e % 10 == 0:
	#     agent.save("../models/human-dqn.h5f")
