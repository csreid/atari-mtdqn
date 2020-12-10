import numpy as np
import torch
import time
from torch.optim import RMSprop, Adam
from torch.utils.tensorboard import SummaryWriter
from IPython import embed

from memory import Memory
from qlearner import TargetQLearning, MTQN

import gym
from gym import ObservationWrapper, RewardWrapper
from gym.wrappers import AtariPreprocessing
from gym.wrappers.frame_stack import FrameStack

class TorchWrapper(ObservationWrapper):
	def observation(self, obs):
		return torch.tensor(obs).float()

env1_name = 'BreakoutNoFrameskip-v4'
env2_name = 'PongNoFrameskip-v4'

spaceinvader_likes = [
	'CentipedeNoFrameskip-v4',
	'SpaceInvadersNoFrameskip-v4',
	'AirRaidNoFrameskip-v4',
	'AssaultNoFrameskip-v4',
	'AtlantisNoFrameskip-v4',
	'DemonAttackNoFrameskip-v4',
]

pong_likes = [
	'PongNoFrameskip-v4',
	'BreakoutNoFrameskip-v4',
	'TennisNoFrameskip-v4',
	'VideoPinballNoFrameskip-v4'
]

pacman_likes = [
	'MsPacmanNoFrameskip-v4',
	'BerzerkNoFrameskip-v4',
	'BankHeistNoFrameskip-v4',
	'AlienNoFrameskip-v4',
]

all_games = pong_likes #spaceinvader_likes + pong_likes + pacman_likes

all_envs = [
	TorchWrapper(FrameStack(AtariPreprocessing(gym.make(n)), num_stack=4))
	for n
	in all_games
]
all_eval_envs = [
	TorchWrapper(FrameStack(AtariPreprocessing(gym.make(n)), num_stack=4))
	for n
	in all_games
]

print(all_games)

STEPS = int(1e6)
logger = SummaryWriter()

env1 = TorchWrapper(FrameStack(AtariPreprocessing(gym.make(env1_name)), num_stack=4))
eval_env1 = TorchWrapper(FrameStack(AtariPreprocessing(gym.make(env1_name)), num_stack=4))

env2 = TorchWrapper(FrameStack(AtariPreprocessing(gym.make(env2_name)), num_stack=4))
eval_env2 = TorchWrapper(FrameStack(AtariPreprocessing(gym.make(env2_name)), num_stack=4))

steps_taken = 0

qs = MTQN([
	env.action_space.n
	for env
	in all_envs
]).get_qs()

agts = [
	TargetQLearning(
		n_actions=env.action_space.n,
		target_lag=1000,
		opt_args={
			'lr': 0.0001
		},
		transitions_per_fit=4,
		name=game
	)
	for game, env
	in zip(all_games, all_envs)
]

for idx, agt in enumerate(agts):
	agt.Q = qs[idx]
	agt.opt = Adam(agt.Q.parameters(), lr=0.0001)

def do_step(agt, env, s):
	a = agt.get_action(s)
	sp, r, done, _ = env.step(a)

	if r > 1.:
		r = 1.
	if r < -1.:
		r = -1

	agt.handle_transition(s, a, r, sp, done)

	if done:
		done = False
		sp = env.reset()

	return sp


held_states1 = torch.zeros((100,) + (4, 84, 84))
held_states2 = torch.zeros((100,) + (4, 84, 84))
num_holdouts1 = 0
num_holdouts2 = 0

def run():
	global steps_taken
	global held_states
	global num_holdouts1
	global num_holdouts2
	global agt1
	global agt2

	s1 = env1.reset()
	s2 = env2.reset()
	s_s = [env.reset() for env in all_envs]

	prev_time = time.time()
	while True:
		s_s = [do_step(agt, env, s) for agt, env, s in zip(agts, all_envs, s_s)]

		if (steps_taken % 100) == 0:
			cur_time = time.time()
			print(f'{steps_taken} in {round(cur_time - prev_time)} ({100/(cur_time - prev_time)} steps/sec)')
			prev_time = cur_time

		steps_taken += 1

		if ((steps_taken*4) % 10000) == 0:
			evls = [agt.evaluate(eval_env, 5) for agt, eval_env in zip(agts, all_eval_envs)]

			for idx, evl in evls:
				logger.add_scalar(f'Evaluation ({all_games[idx]})', np.mean(evls1), steps_taken)
				print(f'\tEval: {evls1} | {evls2}')

if __name__ == '__main__':
	try:
		#agt.play(env)
		run()
	except KeyboardInterrupt:
		embed()
