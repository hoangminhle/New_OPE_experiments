# import numpy as np

import numpy as np

from gym import Env, spaces
from gym.utils import seeding
import sys
from contextlib import closing
from io import StringIO
from gym import utils
from gym.envs.toy_text import discrete

def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()

EXPLICIT_ABSORBING = True
NORMALIZING_FACTOR = 10.0

MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]


class TaxiEnv(discrete.DiscreteEnv):
    """
    The Taxi Problem
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich
    Description:
    There are four designated locations in the grid world indicated by R(ed), G(reen), Y(ellow), and B(lue). When the episode starts, the taxi starts off at a random square and the passenger is at a random location. The taxi drives to the passenger's location, picks up the passenger, drives to the passenger's destination (another one of the four specified locations), and then drops off the passenger. Once the passenger is dropped off, the episode ends.
    Observations: 
    There are 500 discrete states since there are 25 taxi positions, 5 possible locations of the passenger (including the case when the passenger is in the taxi), and 4 destination locations. 
    
    Passenger locations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
    - 4: in taxi
    
    Destinations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
        
    Actions:
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east 
    - 3: move west 
    - 4: pickup passenger
    - 5: dropoff passenger
    
    Rewards: 
    There is a default per-step reward of -1,
    except for delivering the passenger, which is +20,
    or executing "pickup" and "drop-off" actions illegally, which is -10.
    
    Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters (R, G, Y and B): locations for passengers and destinations
    
    state space is represented by:
        (taxi_row, taxi_col, passenger_location, destination)
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.desc = np.asarray(MAP, dtype='c')

        self.locs = locs = [(0,0), (0,4), (4,0), (4,3)]

        num_states = 500
        if EXPLICIT_ABSORBING:
            num_states += 1
            self.s_absorb = num_states-1
        num_rows = 5
        num_columns = 5
        max_row = num_rows - 1
        max_col = num_columns - 1
        initial_state_distrib = np.zeros(num_states)
        num_actions = 6
        P = {state: {action: []
                     for action in range(num_actions)} for state in range(num_states)}
        for row in range(num_rows):
            for col in range(num_columns):
                for pass_idx in range(len(locs) + 1):  # +1 for being inside taxi
                    for dest_idx in range(len(locs)):
                        state = self.encode(row, col, pass_idx, dest_idx)
                        if pass_idx < 4 and pass_idx != dest_idx:
                            initial_state_distrib[state] += 1
                        for action in range(num_actions):
                            # defaults
                            new_row, new_col, new_pass_idx = row, col, pass_idx
                            reward = -1 /NORMALIZING_FACTOR # default reward when there is no pickup/dropoff
                            done = False
                            taxi_loc = (row, col)

                            if action == 0:
                                new_row = min(row + 1, max_row)
                            elif action == 1:
                                new_row = max(row - 1, 0)
                            if action == 2 and self.desc[1 + row, 2 * col + 2] == b":":
                                new_col = min(col + 1, max_col)
                            elif action == 3 and self.desc[1 + row, 2 * col] == b":":
                                new_col = max(col - 1, 0)
                            elif action == 4:  # pickup
                                if (pass_idx < 4 and taxi_loc == locs[pass_idx]):
                                    new_pass_idx = 4
                                else: # passenger not at location
                                    reward = -10 / NORMALIZING_FACTOR
                            elif action == 5:  # dropoff
                                if (taxi_loc == locs[dest_idx]) and pass_idx == 4:
                                    new_pass_idx = dest_idx
                                    done = True
                                    reward = 20 /NORMALIZING_FACTOR
                                elif (taxi_loc in locs) and pass_idx == 4:
                                    new_pass_idx = locs.index(taxi_loc)
                                else: # dropoff at wrong location
                                    reward = -10 /NORMALIZING_FACTOR
                            if EXPLICIT_ABSORBING and done:
                                P[state][action].append((1.0, self.s_absorb, reward, done))
                            else:
                                new_state = self.encode(
                                    new_row, new_col, new_pass_idx, dest_idx)
                                P[state][action].append(
                                    (1.0, new_state, reward, done))
        if EXPLICIT_ABSORBING:
            # absorbing state will self-loop
            for a in range(num_actions):
                P[self.s_absorb][a].append((1.0,self.s_absorb,0, True))
        initial_state_distrib /= initial_state_distrib.sum()
        discrete.DiscreteEnv.__init__(
            self, num_states, num_actions, P, initial_state_distrib)
        self.get_transition_matrix()
        self.explicit_absorbing = EXPLICIT_ABSORBING
        # import pdb; pdb.set_trace()

    def encode(self, taxi_row, taxi_col, pass_loc, dest_idx):
        # (5) 5, 5, 4
        i = taxi_row
        i *= 5
        i += taxi_col
        i *= 5
        i += pass_loc
        i *= 4
        i += dest_idx
        return i

    def decode(self, i):
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i)
        assert 0 <= i < 5
        return reversed(out)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.s)

        def ul(x): return "_" if x == " " else x
        if pass_idx < 4:
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                out[1 + taxi_row][2 * taxi_col + 1], 'yellow', highlight=True)
            pi, pj = self.locs[pass_idx]
            out[1 + pi][2 * pj + 1] = utils.colorize(out[1 + pi][2 * pj + 1], 'blue', bold=True)
        else:  # passenger in taxi
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                ul(out[1 + taxi_row][2 * taxi_col + 1]), 'green', highlight=True)

        di, dj = self.locs[dest_idx]
        out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], 'magenta')
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["South", "North", "East", "West", "Pickup", "Dropoff"][self.lastaction]))
        else: outfile.write("\n")

        # No need to return anything for human
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()

    def get_transition_matrix(self):
        self.P_matrix = np.zeros((self.nS, self.nA, self.nS))
        self.R_matrix = np.zeros((self.nS, self.nA))
        self.done_matrix = np.zeros((self.nS, self.nA))
        for s in range(self.nS):
            for a in range(self.nA):
                for item in self.P[s][a]:
                    self.P_matrix[s,a, item[1]] = item[0]
                    self.R_matrix[s,a] = item[2]
                    self.done_matrix[s,a] = item[3]
                assert sum(self.P_matrix[s,a]) == 1

    def value_iteration(self):
        epsilon = 0.00001
        gamma = 0.99
        not_converged = True
        # Q = np.zeros((self.nS, self.nA), dtype = np.float32)
        Q = np.random.rand(self.nS, self.nA)
        while not_converged:
            Q1 = Q.copy()
            Q = self.R_matrix + gamma * np.dot(self.P_matrix, np.max(Q1, axis=1))
            if np.max(np.abs(Q - Q1)) <= epsilon *(1-gamma)/gamma:
                not_converged = False
        return Q

    def calculate_optimal_value(self):
        epsilon = 0.0001
        gamma = 0.99
        not_converged = True
        V = np.zeros(self.nS, dtype = np.float32)
        while not_converged:
            V1 = V.copy()
            V = np.max(self.R_matrix + gamma * np.dot(self.P_matrix, V1), axis = 1)
            if np.max(np.abs(V - V1)) <= epsilon *(1-gamma)/gamma:
                not_converged = False
        V = (1-gamma)*V # normalize value function
        Q = np.zeros((self.nS, self.nA), dtype = np.float32)
        for s in range(self.nS):
            for a in range(self.nA):
                Q[s,a] = self.R_matrix[s,a] + gamma * np.dot(self.P_matrix[s,a], V)
        return V,Q

    def extract_policy(self, q, temperature = 0.001):
        q_normalized = q / abs(q).max()
        q_normalized = q_normalized - q_normalized.max(axis = 1)[:,None]
        pi = np.zeros((self.nS, self.nA))
        for s in range(self.nS):
            e_x = np.exp(q_normalized[s]/temperature)
            pi[s] = e_x / sum(e_x)
        return pi

# class taxi(object):
# 	n_state = 0
# 	n_action = 6
# 	def __init__(self, length):
# 		self.length = length
# 		self.x = np.random.randint(length)
# 		self.y = np.random.randint(length)
# 		self.possible_passenger_loc = [(0,0), (0,length-1), (length-1,0), (length-1, length-1)]
# 		self.passenger_status = np.random.randint(16)
# 		self.taxi_status = 4
# 		self.n_state = (length**2)*16*5

# 	def reset(self):
# 		length = self.length
# 		self.x = np.random.randint(length)
# 		self.y = np.random.randint(length)
# 		self.possible_passenger_loc = [(0,0), (0,length-1), (length-1,0), (length-1, length-1)]
# 		self.passenger_status = np.random.randint(16)
# 		self.taxi_status = 4
# 		return self.state_encoding()

# 	def state_encoding(self):
# 		length = self.length
# 		return self.taxi_status + (self.passenger_status + (self.x * length + self.y) * 16) * 5

# 	def state_decoding(self, state):
# 		length = self.length
# 		taxi_status = state % 5
# 		state = state / 5
# 		passenger_status = state % 16
# 		state = state / 16
# 		y = state % length
# 		x = state / length
# 		return x,y,passenger_status,taxi_status

# 	def render(self):
# 		MAP = []
# 		length = self.length
# 		for i in range(length):
# 			if i == 0:
# 				MAP.append('-'*(3*length+1))
# 			MAP.append('|' + '  |' * length)
# 			MAP.append('-'*(3*length+1))
# 		MAP = np.asarray(MAP, dtype = 'c')
# 		if self.taxi_status == 4:
# 			MAP[2*self.x+1, 3*self.y+2] = 'O'
# 		else:
# 			MAP[2*self.x+1, 3*self.y+2] = '@'
# 		for i in range(4):
# 			if self.passenger_status & (1<<i):
# 				x,y = self.possible_passenger_loc[i]
# 				MAP[2*x+1, 3*y+1] = 'a'
# 		for line in MAP:
# 			print(''.join(line))
# 		if self.taxi_status == 4:
# 			print('Empty Taxi')
# 		else:
# 			x,y = self.possible_passenger_loc[self.taxi_status]
# 			print('Taxi destination:({},{})'.format(x,y))

# 	def step(self, action):
# 		reward = -1
# 		length = self.length
# 		if action == 0:
# 			if self.x < self.length - 1:
# 				self.x += 1
# 		elif action == 1:
# 			if self.y < self.length - 1:
# 				self.y += 1
# 		elif action == 2:
# 			if self.x > 0:
# 				self.x -= 1
# 		elif action == 3:
# 			if self.y > 0:
# 				self.y -= 1
# 		elif action == 4:	# Try to pick up
# 			for i in range(4):
# 				x,y = self.possible_passenger_loc[i]
# 				if x == self.x and y == self.y and(self.passenger_status & (1<<i)):
# 					# successfully pick up
# 					self.passenger_status -= 1<<i
# 					self.taxi_status = np.random.randint(4)
# 					while self.taxi_status == i:
# 						self.taxi_status = np.random.randint(4)
# 		elif action == 5:
# 			if self.taxi_status < 4:
# 				x,y = self.possible_passenger_loc[self.taxi_status]
# 				if self.x == x and self.y == y:
# 					reward = 20
# 				self.taxi_status = 4
# 		self.change_passenger_status()
# 		return self.state_encoding(), reward

# 	def change_passenger_status(self):
# 		p_generate = [0.3, 0.05, 0.1, 0.2]
# 		p_disappear = [0.05, 0.1, 0.1, 0.05]
# 		for i in range(4):
# 			if self.passenger_status & (1<<i):
# 				if np.random.rand() < p_disappear[i]:
# 					self.passenger_status -= 1<<i
# 			else:
# 				if np.random.rand() < p_generate[i]:
# 					self.passenger_status += 1<<i
# 	def debug(self):
# 		self.reset()
# 		while True:
# 			self.render()
# 			action = input('Action:')
# 			if action > 5 or action < 0:
# 				break
# 			else:
# 				_, reward = self.step(action)
# 				print(reward)