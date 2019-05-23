from ast import literal_eval
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

import matplotlib.pyplot as plt

# Defining actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
STAY = 4

# Define colors
#COLOURS = {0: [1, 1, 1], 1: [0.6, 0.3, 0.0], 3: [0.0, 1.0, 0.0], 10: [0.6, 0, 1]}
COLOURS = {0: [1, 1, 1], 1: [0.0, 0.0, 0.0], 3: [0.0, 0.0, 0.0], 10: [0.0, 0, 0]}

class GridWorld(gym.Env):
    metadata = {'render.modes': ['human']}
    MAP = "1 1 1 1 1 1 1 1 1 1 1 1 1\n" \
          "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
          "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
          "1 0 0 0 0 0 0 0 0 0 0 0 1\n" \
          "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
          "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
          "1 1 0 1 1 1 1 0 0 0 0 0 1\n" \
          "1 0 0 0 0 0 1 1 1 0 1 1 1\n" \
          "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
          "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
          "1 0 0 0 0 0 0 0 0 0 0 0 1\n" \
          "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
          "1 1 1 1 1 1 1 1 1 1 1 1 1"

    def __init__(self, MAP=MAP, goal_reward=1.0, step_reward=-0.1, goals=None, continual=False, random_start_state=True):

        self.n = None
        self.m = None

        self.grid = None
        self.hallwayStates = None
        self.possibleStates = []
        self.walls = []
        
        self.MAP = MAP
        self._map_init()
        
        self.continual = continual

        self.random_start_state = random_start_state
        self.start_state_coord = (1, 1)
        self.state = self.start_state_coord

        self.done = False

        self.goalReached = None
        self.goals = [(1, 1)]
        if goals:
            self.goals = goals

        # Rewards
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.hit_wall_reward =  step_reward

        # Gym spaces for observation and action space
        self.observation_space = spaces.Discrete(len(self.possibleStates))
        self.action_space = spaces.Discrete(5)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        np.random.rand(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)

        if action == STAY and self.state in self.goals and not self.continual:
            self.done = True

        reward = self._get_reward(self.state, action)
        
        x, y = self.state
        if action == UP:
            x = x - 1
        elif action == DOWN:
            x = x + 1
        elif action == RIGHT:
            y = y + 1
        elif action == LEFT:
            y = y - 1
        new_state = (x, y)
        
        if self._get_grid_value(new_state) == 1:  # new_state in walls list
            # stay at old state if new coord is wall
            new_state = self.state
        else:
            self.state = new_state
        
        return [self.goalReached, self.state], reward, self.done, None

    def reset(self):
        self.done = False
        self.goalReached = None
        if self.random_start_state:
            idx = np.random.randint(len(self.possibleStates))
            self.state = self.possibleStates[idx]  # self.start_state_coord
        else:
            self.state = self.start_state_coord
        return [self.goalReached, self.state]

    def render(self, mode='human', draw_arrows=False, draw_values=False, draw_rewards=False, V = None, policy=None, R = None, title=None, grid=False, cmap='RdBu'):

        img = self._gridmap_to_img()        
        fig = plt.figure(1, figsize=(10, 8), dpi=60, facecolor='w', edgecolor='k')
        
        plt.clf()
        plt.xticks(np.arange(0, 2*self.n, 1))
        plt.yticks(np.arange(0, 2*self.m, 1))
        plt.grid(grid)
        if title:
            plt.title(title)

        plt.imshow(img, origin="upper", extent=[0, self.n, self.m, 0])
        fig.canvas.draw()
        
        if draw_rewards & (type(R) is not None): # For showing optimal values
            r = np.zeros((self.n,self.m))+float("-inf")
            for state, value in R.items():
                y, x = literal_eval(state)[1]
                r[y,x] = value.sum()  
            c = plt.imshow(r, origin="upper", cmap=cmap, extent=[0, self.n, self.m, 0])
            fig.colorbar(c, ax=fig.gca())
            
        if draw_values & (type(V) is not None): # For showing optimal values
            v = np.zeros((self.n,self.m))+float("-inf")
            for state, value in V.items():
                y, x = literal_eval(state)[1]
                v[y,x] = value  
            c = plt.imshow(v, origin="upper", cmap=cmap, extent=[0, self.n, self.m, 0])
            fig.colorbar(c, ax=fig.gca())
                
        if draw_arrows & (type(policy) is not None):  # For drawing arrows of optimal policy
            fig = plt.gcf()
            ax = fig.gca()
            for state, action in policy.items():
                y, x = literal_eval(state)[1]
                self._draw_arrows(fig, ax, x, y, action)

        plt.pause(0.00001)  # 0.01
        return

    def _map_init(self):
        self.grid = []
        lines = self.MAP.split('\n')

        for i, row in enumerate(lines):
            row = row.split(' ')
            if self.n is not None and len(row) != self.n:
                raise ValueError(
                    "Map's rows are not of the same dimension...")
            self.n = len(row)
            rowArray = []
            for j, col in enumerate(row):
                rowArray.append(int(col))
                if col == "1":
                    self.walls.append((i, j))
                # possible states
                else:
                    self.possibleStates.append((i, j))
            self.grid.append(rowArray)
        self.m = i + 1

        self._find_hallWays()

    def _find_hallWays(self):
        self.hallwayStates = []
        for x, y in self.possibleStates:
            if ((self.grid[x - 1][y] == 1) and (self.grid[x + 1][y] == 1)) or \
                    ((self.grid[x][y - 1] == 1) and (self.grid[x][y + 1] == 1)):
                self.hallwayStates.append((x, y))

    def _get_grid_value(self, state):
        return self.grid[state[0]][state[1]]

    # specific for self.MAP
    def _getRoomNumber(self, state=None):
        if state == None:
            state = self.state
        # if state isn't at hall way point
        xCount = self._greaterThanCounter(state, 0)
        yCount = self._greaterThanCounter(state, 1)
        room = 0
        if yCount >= 2:
            if xCount >= 2:
                room = 2
            else:
                room = 1
        else:
            if xCount >= 2:
                room = 3
            else:
                room = 0

        return room

    def _greaterThanCounter(self, state, index):
        count = 0
        for h in self.hallwayStates:
            if state[index] > h[index]:
                count = count + 1
        return count

    def _get_reward(self, state, action):
        reward = 0
        if action == STAY and self.state in self.goals and not self.goalReached:
            self.goalReached = state
            reward = self.goal_reward
        elif action == STAY and self.state in self.goals:
            reward = self.step_reward/4
        elif action == STAY:
            reward = self.step_reward/2
        elif self._get_grid_value(state) == 1:
            reward = self.hit_wall_reward
        else:
            reward = self.step_reward
        return reward

    def _draw_arrows(self, fig, ax, x, y, direction):
        if direction == UP:
            x += 0.5
            y += 1
            dx = 0
            dy = -0.4
        if direction == DOWN:
            x += 0.5
            dx = 0
            dy = 0.4
        if direction == RIGHT:
            y += 0.5
            dx = 0.4
            dy = 0
        if direction == LEFT:
            x += 1
            y += 0.5
            dx = -0.4
            dy = 0
        if direction == STAY:
            x += 0.5
            y += 0.5
            dx = 0
            dy = 0
            
            ax.add_patch(plt.Circle((x, y), radius=0.25, fc='k'))
            return

        plt.arrow(x,  # x1
                  y,  # y1
                  dx,  # x2 - x1
                  dy,  # y2 - y1
                  facecolor='k',
                  edgecolor='k',
                  width=0.005,
                  head_width=0.4,
                  )

    def _gridmap_to_img(self):
        row_size = len(self.grid)
        col_size = len(self.grid[0])

        obs_shape = [row_size, col_size, 3]

        img = np.zeros(obs_shape)

        gs0 = int(img.shape[0] / row_size)
        gs1 = int(img.shape[1] / col_size)
        for i in range(row_size):
            for j in range(col_size):
                for k in range(3):
                    if (i, j) == self.state:
                        this_value = COLOURS[10][k]
                    elif (i, j) in self.goals:
                        this_value = COLOURS[3][k]
                    else:

                        colour_number = int(self.grid[i][j])
                        this_value = COLOURS[colour_number][k]
                    img[i * gs0:(i + 1) * gs0, j * gs1:(j + 1)
                                                       * gs1, k] = this_value
        return img
