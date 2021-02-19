import time
import copy

import numpy as np
from numpy import e
import torch
from torch.nn import Sequential, Linear, LeakyReLU, MSELoss, Conv2d, Flatten, Sigmoid, Tanh, BatchNorm2d, BatchNorm1d
from torch.optim import Adam, RMSprop
from torch.nn.utils import clip_grad_value_

from memory import Memory
from reinforcement import Learner

np.seterr(all='raise')

class QLearning(Learner):
	def __init__(
		self,
		n_actions,
		opt=Adam,
		opt_args={},
		loss=MSELoss,
		gamma=0.99,
		do_target=True,
		memory_len=10000,
		name=None,
		memory_shape=(4, 84, 84),
		initial_eps=0.1,
		final_eps=0.01,
		decay_steps=int(1e6),
		memory_dtype=torch.uint8
	):
		self.n_actions = n_actions
		self._memory = Memory(memory_len, memory_shape, dtype=memory_dtype)
		self.Q = Sequential(
			Conv2d(4, 32, kernel_size=8, stride=4),
			LeakyReLU(),
			Conv2d(32, 64, kernel_size=4, stride=2),
			LeakyReLU(),
			Conv2d(64, 64, kernel_size=3, stride=1),
			LeakyReLU(),
			Flatten(),
			Linear(3136, 512),
			LeakyReLU(),
			Linear(512, self.n_actions)
		)
		self._name = name

		self.gamma = gamma

		self.opt = opt(self.Q.parameters(), **opt_args)
		self._base_loss_fn = MSELoss()
		self._steps = 0

		self.eps = initial_eps
		#self.decay = (final_eps / initial_eps) ** (1/decay_steps)

		# Linear Decay
		self.decay = (initial_eps - final_eps) / decay_steps

	def learn(self, batch_size=100, n_samples=100):
		if len(self._memory) < n_samples:
			return 'n/a'

		self.Q.train()
		X, y = self._build_dataset(n_samples)
		y_pred = self.Q(X)
		loss = self._base_loss_fn(y, y_pred) 

		self.opt.zero_grad()
		loss.backward()
		clip_grad_value_(self.Q.parameters(), 1)
		self.opt.step()

		self.Q.eval()
		return loss.item()

	def _build_dataset(self, n):
		with torch.no_grad():
			s_s, a_s, r_s, sp_s, done_mask = self._memory.sample(n)

			vhat_sp_s = torch.max(self.Q(sp_s.float()), dim=1).values
			vhat_sp_s[done_mask] = 0

			targets = self.Q(s_s.float())

			for idx, t in enumerate(targets):
				t[int(a_s[idx].byte())] = r_s[idx] + self.gamma * vhat_sp_s[idx]

			X = s_s.float()
			y = targets
		return X, y

	def handle_transition(self, s, a, r, sp, done):
		s = self._convert_to_torch(s)
		sp = self._convert_to_torch(sp)

		self._memory.append((
			s,
			torch.from_numpy(np.array([a]))[0],
			r,
			sp,
			done
		))

		if (self._steps % 4) == 0:
			self.learn(n_samples=1024)

		self._steps += 1

	def get_action_vals(self, s):
		s = self._convert_to_torch(s)

		return self.Q(s[None, :])

	def exploration_strategy(self, s):
		#self.eps *= self.decay
		self.eps -= self.decay
		if np.random.random() > self.eps:
			ps = np.zeros(self.n_actions)
			best_action = torch.argmax(self.Q(s[None, :]))
			try:
				ps[best_action] = 1.
			except:
				print(self._name)
				exit()
		else:
			ps = np.full(self.n_actions, 1 / self.n_actions)

		return ps

	def deterministic_strategy(self, s):
		s = self._convert_to_torch(s)

		eps = 0.05
		if np.random.random() > eps:
			ps = np.zeros(self.n_actions)
			best_action = torch.argmax(self.Q(s[None, :])).detach().numpy()
			ps[best_action] = 1.
		else:
			ps = np.full(self.n_actions, 1 / self.n_actions)

		return ps

class TargetQLearning(QLearning):
	def __init__(
		self,
		n_actions,
		opt=Adam,
		opt_args={},
		loss=MSELoss,
		gamma=0.99,
		target_lag=100,
		transitions_per_fit=1,
		memory_len=10000,
		name=None,
		memory_shape=(4, 84, 84),
		initial_eps=0.1,
		final_eps=0.01,
		decay_steps=int(1e6),
		memory_dtype=torch.uint8
	):
		super().__init__(
			n_actions=n_actions,
			opt=opt,
			opt_args=opt_args,
			loss=loss,
			gamma=gamma,
			memory_len=memory_len,
			memory_shape=memory_shape,
			name=name,
			initial_eps=initial_eps,
			final_eps=final_eps,
			decay_steps=decay_steps,
			memory_dtype=memory_dtype
		)
		print('Building', name)

		self.target_Q = copy.deepcopy(self.Q)
		self.target_lag = target_lag
		self.transitions_per_fit=transitions_per_fit

	def _build_dataset(self, n):
		with torch.no_grad():
			s_s, a_s, r_s, sp_s, done_mask = self._memory.sample(n)

			vhat_sp_s = torch.max(self.target_Q(sp_s.float()), dim=1).values
			vhat_sp_s[done_mask] = 0

			targets = self.Q(s_s.float())

			for idx, t in enumerate(targets):
				t[int(a_s[idx].byte())] = r_s[idx] + self.gamma * vhat_sp_s[idx]

			X = s_s.float()
			y = targets
		return X, y

	def handle_transition(self, s, a, r, sp, done):
		s = self._convert_to_torch(s)
		sp = self._convert_to_torch(sp)

		self._memory.append((
			s,
			torch.from_numpy(np.array([a]))[0],
			r,
			sp,
			done
		))

		self._steps += 1
		if (self._steps % self.transitions_per_fit) == 0:
			self.learn()

		if (self._steps % self.target_lag) == 0:
			self.target_Q = copy.deepcopy(self.Q)


class MTQN(torch.nn.Module):
	def __init__(self, output_shapes):
		super().__init__()
		# Apply the convolution stack from (Minh, 2015)
		# but output a 1024-dimension meta-state vector
		# with values in (0, 1)
		self.inputs = [
			Sequential(
				Conv2d(4, 32, kernel_size=8, stride=4),
				LeakyReLU(),
				Conv2d(32, 64, kernel_size=4, stride=2),
				LeakyReLU(),
				Conv2d(64, 64, kernel_size=3, stride=1),
				LeakyReLU(),
				Flatten(),
				Linear(3136, 1024),
				Tanh()
			)

			for _ in output_shapes
		]

		for i in self.inputs:
			torch.nn.init.constant_(i[0].weight, 0.)

		# Shared layers map 1024d metastate to a 128d
		# feature vector, with values in (0, 1)
		self.shared = Sequential(
			Linear(1024, 512),
			LeakyReLU(),
			Linear(512, 256),
			LeakyReLU(),
			Linear(256, 128),
			Tanh()
		)

		# Map features to task-specific outputs
		self.outputs = [
			Sequential(
				Linear(128, 64),
				LeakyReLU(),
				Linear(64, s)
			)

			for s in output_shapes
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
