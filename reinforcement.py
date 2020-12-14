import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from scipy.special import softmax

class Learner:
	def __init__(self):
		self._temp = 5000
		self._last_eval = None
		self.name = None

	def get_action(self, s, explore=True):
		avs = self.get_action_vals(s)
		if explore:
			ps = self.exploration_strategy(s)
			return np.random.choice(np.arange(len(ps)), p=ps)

		ps = self.deterministic_strategy(s)
		return np.random.choice(np.arange(len(ps)), p=ps)

	def evaluate(self, env, n, starting_state=None):
		vals = []
		for _ in range(n):
			done = False
			s = env.reset()

			if starting_state is not None:
				s = starting_state
				env.env.state = s

			total_r = 0

			while not done:
				a = self.get_action(s, explore=False)
				s, r, done, _ = env.step(a)
				total_r += r

			vals.append(total_r)

		evl = np.mean(np.array(vals))
		self._last_eval = evl
		return np.mean(np.array(vals))

	def play_simple(self, env):
		s = env.reset()
		done = False

		while not done:
			a = self.get_action(s)
			s, r, done, _ = env.step(a)
			env.render()
			time.sleep(0.015)

	def play(self, env, interval=100):
		fig = plt.figure(constrained_layout=True)
		gs = GridSpec(3, 4, figure=fig)
		v_history_len = 1000

		r1c1 = fig.add_subplot(gs[0:2, 0])
		r1c2 = fig.add_subplot(gs[0:2, 1])
		r1c3 = fig.add_subplot(gs[0:2, 2])
		r1c4 = fig.add_subplot(gs[0:2, 3])

		r2c1 = fig.add_subplot(gs[2, 0:2])
		r2c2 = fig.add_subplot(gs[2, 2:])
		r2c2.set_ylim(0., 1.)
		r2c1.set_ylim(0., 1.)

		s = env.reset()
		a = self.get_action(s)
		vals = self.get_action_vals(s).detach().numpy()

		im1 = r1c1.imshow(s[0])
		im2 = r1c2.imshow(s[1])
		im3 = r1c3.imshow(s[2])
		im4 = r1c4.imshow(s[3])

		actions = env.unwrapped.get_action_meanings() #['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']

		barcontainer = r2c1.bar(actions, np.zeros(self.n_actions))
		[l2] = r2c2.plot(np.arange(v_history_len), np.zeros(v_history_len))

		all_values = []

		def gen():
			nonlocal s

			a = self.get_action(s, explore=False)
			s, r, done, _ = env.step(a)
			vals = self.get_action_vals(s).detach().numpy()
			all_values.append(np.max(vals))

			yield (s, vals, all_values)

		def anim(data):
			(s, vals, all_vals) = data
			all_vals = np.array(all_vals[-v_history_len:])
			vals = vals[0]

			im1.set_data(s[0])
			im2.set_data(s[1])
			im3.set_data(s[2])
			im4.set_data(s[3])

			q_data = np.interp(vals, (vals.min(), vals.max()), (0, 1))

			v_s = np.interp(all_vals, (all_vals.min(), all_vals.max()), (0, 1))
			v_data = np.zeros(v_history_len)

			v_data[-len(v_s):] = v_s

			for idx, bar in enumerate(barcontainer.patches):
				bar.set_height(q_data[idx])

			l2.set_ydata(v_data)

			return [im1, im2, im3, im4, l2, *barcontainer.patches]

		anim = FuncAnimation(fig, anim, gen, interval=interval, blit=True)
		plt.show()
		return anim

	def _convert_to_discrete(self, s, bounds):
		if type(s) is tuple:
			return s

		if torch.is_tensor(s):
			s = s.detach()

		new_obs = tuple(
			np.searchsorted(self.bounds[i], s[i])
			for i in range(4)
		)

		return new_obs

	def _convert_to_torch(self, s):
		if torch.is_tensor(s):
			return s

		new_s = torch.tensor(s, requires_grad=True).float().reshape((-1))
		return new_s

	def set_name(self, name):
		self.name = name

	def handle_transition(self, s, a, r, sp, done):
		raise NotImplementedError

	def get_action_vals(self, s):
		raise NotImplementedError

	def exploration_strategy(self, s):
		raise NotImplementedError

	def deterministic_strategy(self, s):
		raise NotImplementedError
