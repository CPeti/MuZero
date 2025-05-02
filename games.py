from abc import ABC, abstractmethod
from typing import List
import numpy as np

class GameSimulator(ABC):
    """
    Abstract base class for game simulators.
    """
    @abstractmethod
    def reset(self, two_player) -> np.ndarray:
        """
        Reset the game state.
        """
    @abstractmethod
    def get_legal_actions(self) -> List[int]:
        """
        Get the list of legal actions (empty cells).
        """
        pass

    @abstractmethod
    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """
        Take a step in the game.
        
        Args:
            action: The action to take (index of the cell).
        """
        pass

    @abstractmethod
    def get_observation(self) -> np.ndarray:
        """
        Get the current observation of the game.
        """
        pass

    @abstractmethod
    def get_action_space_size(self) -> int:
        """
        Get the size of the action space.
        """
        pass

    @abstractmethod
    def get_all_actions(self) -> List[int]:
        """
        Get all possible actions (all cells).
        """
        pass

class GridEnv(GameSimulator):
    """
    Simple grid environment where the agent moves in a 2D grid.
    The agent starts at the top-left corner and can move down or right.
    The goal is to reach the bottom-right corner. ~ Frozen Lake.
    
    Taken from official MuZero repository.
    """
    def __init__(self, size=3, all_actions=False, randomize=False):
        self.size = size
        self.two_player = False
        self.player = 1
        self.randomize = randomize
        self.state_shape = (size * size,)
        self.all_actions = all_actions
        self.reset()

    def get_legal_actions(self) -> List[int]:
        if self.all_actions:
            legal_actions = list(range(4))
        else:
            legal_actions = list(range(2))
        if self.position[0] == (self.size - 1):
            legal_actions.remove(0)
        if self.position[1] == (self.size - 1):
            legal_actions.remove(1)
        if self.all_actions:
            if self.position[0] == 0:
                legal_actions.remove(2)
            if self.position[1] == 0:
                legal_actions.remove(3)
        return legal_actions

    def step(self, action) -> tuple[np.ndarray, float, bool]:
        if action not in self.get_legal_actions():
            pass
        elif action == 0:
            self.position[0] += 1
        elif action == 1:
            self.position[1] += 1
        if self.all_actions:
            if action == 2:
                self.position[0] -= 1
            elif action == 3:
                self.position[1] -= 1
        
        done = False
        reward = 0
        if self.position == self.goal:
            reward = 1
            if self.randomize:
                # choose a random goal that is not the same as the start position
                while self.position == self.goal:
                    self.goal = [np.random.randint(0, self.size), np.random.randint(0, self.size)]
            else:
                done = True
        return self.get_observation(), reward, done

    def reset(self) -> np.ndarray:
        if self.randomize:
            self.position = [np.random.randint(0, self.size), np.random.randint(0, self.size)]
            # choose a random goal that is not the same as the start position
            while True:
                self.goal = [np.random.randint(0, self.size), np.random.randint(0, self.size)]
                if self.position != self.goal:
                    break
        else:
            self.position = [0, 0]
            self.goal = [self.size - 1, self.size - 1]
        return self.get_observation()

    def render(self) -> None:
        im = np.full((self.size, self.size), "-")
        im[self.goal[0], self.goal[1]] = "G"
        im[self.position[0], self.position[1]] = "A"
        print(im)

    def get_observation(self) -> np.ndarray:
        observation = np.zeros((self.size, self.size))
        observation[self.position[0]][self.position[1]] = 1
        observation[self.goal[0]][self.goal[1]] = -1
        return observation
        
    def get_action_space_size(self) -> int:
        """
        Get the size of the action space.
        """
        if self.all_actions:
            return 4
        return 2
    
    def get_all_actions(self) -> List[int]:
        """
        Get all possible actions (all cells).
        """
        if self.all_actions:
            return [0, 1, 2, 3] # down, right, up, left
        else:
            return [0, 1] # down, right