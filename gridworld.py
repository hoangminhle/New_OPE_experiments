import sys
from contextlib import closing

import numpy as np
from io import StringIO

from gym import utils
from gym.envs.toy_text import discrete

EXPLICIT_ABSORBING = True
NORMALIZING_FACTOR = 10.0
UNIFORM_RANDOM_START = True

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": [
        "Sxxx",
        "xOxO",
        "xxxO",
        "OxxG"
    ],
    "8x8": [
        "Sxxxxxxx",
        "xxxxxxxx",
        "xxxOxxxx",
        "xxxxxOxx",
        "xxxOxxxx",
        "xOOxxxOx",
        "xOxxOxOx",
        "xxxOxxxG"
    ],
}

def generate_random_map(size=8, p=0.8):
    """Generates a random valid map (one that has a path from start to goal)
    :param size: size of each side of the grid
    :param p: probability that a tile is frozen
    """
    valid = False

    # DFS to check that it's a valid path.
    def is_valid(res):
        frontier, discovered = [], set()
        frontier.append((0,0))
        while frontier:
            r, c = frontier.pop()
            if not (r,c) in discovered:
                discovered.add((r,c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                        continue
                    if res[r_new][c_new] == 'G':
                        return True
                    if (res[r_new][c_new] !='O'):
                        frontier.append((r_new, c_new))
        return False

    while not valid:
        p = min(1, p)
        res = np.random.choice(['x', 'O'], (size, size), p=[p, 1-p])
        res[0][0] = 'S'
        res[-1][-1] = 'G'
        valid = is_valid(res)
    return ["".join(x) for x in res]

class GridWorldEnv(discrete.DiscreteEnv):
    """
    Winter is here. You and your friends were tossing around a frisbee at the park
    when you made a wild throw that left the frisbee out in the middle of the lake.
    The water is mostly frozen, but there are a few holes where the ice has melted.
    If you step into one of those holes, you'll fall into the freezing water.
    At this time, there's an international frisbee shortage, so it's absolutely imperative that
    you navigate across the lake and retrieve the disc.
    However, the ice is slippery, so you won't always move in the direction you intend.
    The surface is described using a grid like the following
        SFFF
        FHFH
        FFFH
        HFFG
    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located
    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.
    """
    metadata = {'render.modes': ['human', 'ansi']}
    def __init__(self, desc=None, map_name="8x8",is_slippery=False):
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc,dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)

        nA = 4
        # nS = nrow * ncol
        if EXPLICIT_ABSORBING:
            nS = nrow * ncol + 1
            self.s_absorb = nrow * ncol
        else:
            nS = nrow * ncol
        if UNIFORM_RANDOM_START:
            isd = np.array(desc != b'G').astype('float64').ravel()
        else:
            isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()
        if EXPLICIT_ABSORBING: isd = np.pad(isd, (0,1), 'constant')

        self.goal_loc = (np.where(self.desc == b'G')[0][0], np.where(self.desc == b'G')[1][0])

        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row*ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col-1,0)
            elif a == DOWN:
                row = min(row+1,nrow-1)
            elif a == RIGHT:
                col = min(col+1,ncol-1)
            elif a == UP:
                row = max(row-1,0)
            return (row, col)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'G': #b'GO'
                        if EXPLICIT_ABSORBING:
                            li.append((1.0, self.s_absorb, 0, True))
                        else:
                            li.append((1.0, s, 0, True))                        
                    else:
                        if is_slippery:
                            for b in [(a-1)%4, a, (a+1)%4]:
                                newrow, newcol = inc(row, col, b)
                                newstate = to_s(newrow, newcol)
                                newletter = desc[newrow, newcol]
                                done = bytes(newletter) in b'GO'
                                # rew = float(newletter == b'G')
                                if newletter == b'G':
                                    rew = 10.0 / NORMALIZING_FACTOR
                                elif newletter == b'O':
                                    rew = -5.0 / NORMALIZING_FACTOR
                                else:
                                    # rew = -0.1 + 1.0/(abs(self.goal_loc[0]-newrow) + abs(self.goal_loc[1]-newcol) + 1) + (newcol - newrow) / (8*2)
                                    # rew = -0.1-(abs(self.goal_loc[0]-newrow) + abs(self.goal_loc[1]-newcol))/(8*2) + (newcol - newrow) / (8*2)
                                    rew = (-0.1-(abs(self.goal_loc[0]-newrow) + abs(self.goal_loc[1]-newcol))/(8*2) + (newcol - newrow) / (8*2)) /NORMALIZING_FACTOR
                                li.append((1.0/3.0, newstate, rew, done))
                        else:
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) in b'G' #b'GO'
                            # rew = float(newletter == b'G')
                            if newletter == b'G':
                                rew = 10.0 / NORMALIZING_FACTOR
                            elif newletter == b'O':
                                rew = (-5.0-(abs(self.goal_loc[0]-newrow) + abs(self.goal_loc[1]-newcol))/(8*2) + (newcol - newrow) / (8*2))/NORMALIZING_FACTOR
                            else:
                                # rew = -1 + 1.0/(abs(self.goal_loc[0]-newrow) + abs(self.goal_loc[1]-newcol) + 1) + (newcol - newrow) / (8*2)
                                # rew = -0.1-(abs(self.goal_loc[0]-newrow) + abs(self.goal_loc[1]-newcol))/(8*2) + (newrow - newcol) / (8*2)
                                rew = (-0.1-(abs(self.goal_loc[0]-newrow) + abs(self.goal_loc[1]-newcol))/(8*2) + (newcol - newrow) / (8*2))/NORMALIZING_FACTOR
                            li.append((1.0, newstate, rew, done))
        if EXPLICIT_ABSORBING:
            # absorbing state will self-loop
            for a in range(4):
                li = P[self.s_absorb][a]
                li.append((1.0, self.s_absorb, 0, True))

        super(GridWorldEnv, self).__init__(nS, nA, P, isd)
        self.get_transition_matrix()
        self.explicit_absorbing = EXPLICIT_ABSORBING

    def to_s(self, row, col):
        return row*self.ncol + col

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        if EXPLICIT_ABSORBING and self.s == self.s_absorb:
            outfile.write("In absorbing state \n")
            outfile.write("\n".join(''.join(line) for line in self.last_desc)+"\n")    
        else:
            row, col = self.s // self.ncol, self.s % self.ncol
            desc = self.desc.tolist()
            desc = [[c.decode('utf-8') for c in line] for line in desc]
            # import pdb; pdb.set_trace()
            desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
            if self.lastaction is not None:
                outfile.write("  ({})\n".format(["Left","Down","Right","Up"][self.lastaction]))
            else:
                outfile.write("\n")
            outfile.write("\n".join(''.join(line) for line in desc)+"\n")
            self.last_desc = desc
            # self.last_row = row
            # self.last_col = col

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

    # def calculate_optimal_value(self):
    #     epsilon = 0.00001
    #     gamma = 0.9
    #     not_converged = True
    #     V = np.zeros(self.nS, dtype = np.float32)
    #     while not_converged:
    #         V1 = V.copy()
    #         V = np.max(self.R_matrix + gamma * np.dot(self.P_matrix, V1), axis = 1)
    #         if np.max(np.abs(V - V1)) <= epsilon *(1-gamma)/gamma:
    #             not_converged = False
    #     V = (1-gamma)*V # normalize value function
    #     Q = np.zeros((self.nS, self.nA), dtype = np.float32)
    #     for s in range(self.nS):
    #         for a in range(self.nA):
    #             Q[s,a] = self.R_matrix[s,a] + gamma * np.dot(self.P_matrix[s,a], V)
    #     return V,Q

    def extract_policy(self, q, temperature = 0.001):
        q_normalized = q / abs(q).max()
        q_normalized = q_normalized - q_normalized.max(axis = 1)[:,None]
        pi = np.zeros((self.nS, self.nA))
        for s in range(self.nS):
            e_x = np.exp(q_normalized[s]/temperature)
            pi[s] = e_x / sum(e_x)
        return pi

            
