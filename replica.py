import torch
import pickle
from torch.nn import Linear, ReLU, Sigmoid, Sequential, SmoothL1Loss
from torch.optim import Adam
from qlearner import TargetQLearning, MTQN
from gym import ObservationWrapper, RewardWrapper
import time
import gym
import copy

class TorchWrapper(ObservationWrapper):
	def observation(self, obs):
		return torch.tensor(obs).float()

class MTQN(torch.nn.Module):
	def __init__(self, input_shapes, output_shapes):
		super().__init__()
		self.inputs = [
			Sequential(
				Linear(i_s, 80),
				ReLU()
			)

			for i_s in input_shapes
		]

		self.shared = Sequential(
			Linear(80, 80),
			ReLU(),
			Linear(80, 80),
			Sigmoid()
		)

		# Map features to task-specific outputs
		self.outputs = [
			Sequential(
				Linear(80, o_s)
			)

			for o_s in output_shapes
		]

	def _output(self, i, idx):
		o = self.inputs[idx](i)
		o = self.shared(o)
		o = self.outputs[idx](o)

		return o

	def forward(self, *i_s):
		os = [self._output(i, idx) for idx, i in enumerate(i_s)]
		return os

	def get_qs(self):
		return [
			Sequential(
				self.inputs[i],
				self.shared,
				self.outputs[i]
			)
			for i in range(len(self.inputs))
		]


env_names = ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
envs = [
	TorchWrapper(gym.make(name).env)
	for name
	in env_names
]
eval_envs = [
	TorchWrapper(gym.make(name))
	for name
	in env_names
]

STEPS = 1000
EPOCHS=50

qs = MTQN(
	[env.observation_space.shape[0] for env in envs],
	[env.action_space.n for env in envs]
).get_qs()

agts = [
	TargetQLearning(
		n_actions=env.action_space.n,
		target_lag=100,
		loss=SmoothL1Loss,
		memory_len=5000,
		name=env_names[idx],
		memory_shape=(env.observation_space.shape[0],),
		initial_eps=1.,
		final_eps=0.01,
		decay_steps=5000,
		memory_dtype=torch.float
	)
	for idx, env
	in enumerate(envs)
]

for idx, agt in enumerate(agts):
	agt.Q = qs[idx]
	agt.target_Q = copy.deepcopy(agt.Q)
	agt.opt = Adam(agt.Q.parameters(), lr=1e-3)

	print(agt.Q)

def do_step(agt, env, s):
	a = agt.get_action(s)
	sp, r, done, _ = env.step(a)

	agt.handle_transition(s, a, r, sp, done)

	if done:
		done = False
		sp = env.reset()

	return sp

held_states = torch.zeros((100, 4))
s_s = [env.reset() for env in envs]
prev_time = time.time()

for i in range(100):
	s = envs[0].reset()
	held_states[i] = s

for epoch in range(EPOCHS):
	for step in range(STEPS):
		s_s = [do_step(agt, env, s) for agt, env, s in zip(agts, envs, s_s)]

	cur_time = time.time()
	print(f'{epoch} in {round(cur_time - prev_time)} ({100/(cur_time - prev_time)} steps/sec) Evals: {torch.mean(agts[0].Q(held_states))}')
	prev_time = cur_time

	evls = [agt.evaluate(eval_env, 5) for agt, eval_env in zip(agts, eval_envs)]
	print(evls)

	for agt, env in zip(agts, eval_envs):
		agt.play_simple(env)

pickle.dump(agts, open('agts.pickle', 'wb'))
