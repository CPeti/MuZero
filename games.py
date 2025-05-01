from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
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
        self.two_player = two_player

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

class TicTacToe(GameSimulator):
    """
    TicTacToa/Gomoku/Connect x game simulator.
    """
    def __init__(self, board_size: int = 3, target: int = 3):
        self.board_size = board_size
        self.target = target
        self.two_player = True
        self.state_shape = (3, 3, 3)
        self.reset()

    def reset(self) -> np.ndarray:
        """
        Reset the game state.
        """
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        self.player = 1
        self.done = False
        self.winner = None

        return self.get_observation()
    
    def get_action_space_size(self) -> int:
        """
        Get the size of the action space.
        """
        return self.board_size * self.board_size

    def get_legal_actions(self) -> List[int]:
        """
        Get the list of legal actions (empty cells).
        """
        return [i for i in range(self.board_size * self.board_size) if self.board[i // self.board_size, i % self.board_size] == 0]
    
    def get_all_actions(self) -> List[int]:
        """
        Get all possible actions (all cells).
        """
        return [i for i in range(self.board_size * self.board_size)]
    
    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """
        Take a step in the game.
        
        Args:
            action: The action to take (index of the cell).
        """
        if self.done:
            raise ValueError("Game is already done.")
        
        row, col = divmod(action, self.board_size)
        if self.board[row, col] != 0:
            raise ValueError("Invalid action. Cell is already occupied.")
        
        self.board[row, col] = self.player
        if self.check_winner():
            self.done = True
            reward = 1
        elif len(self.get_legal_actions()) == 0:
            self.done = True
            reward = 0
        else:
            self.player = -self.player
            reward = 0


        return self.get_observation(), reward, self.done

    def get_observation(self) -> np.ndarray:
        """
        Get the current observation of the game.
        """
        board_player1 = np.where(self.board == 1, 1, 0)
        board_player2 = np.where(self.board == -1, 1, 0)
        board_to_play = np.full((self.board_size, self.board_size), self.player)
        return np.stack([board_player1, board_player2, board_to_play], dtype=np.int32)
    
    def check_winner(self):
        """
        Check if any player has won by connecting self.target points in a row on the board.
        
        The board is stored in self.board as a numpy array where:
            0 = empty cell
            1 = player 1's mark
        -1 = player 2's mark
        
        Returns:
        int: 0 if no winner, 1 if player 1 wins, -1 if player 2 wins
        """
        # Get board dimensions
        m = self.board.shape[0]
        n_to_win = self.target
        
        # Check rows
        for i in range(m):
            for j in range(m - n_to_win + 1):
                for player in [1, -1]:
                    if np.all(self.board[i, j:j+n_to_win] == player):
                        return player
        
        # Check columns
        for i in range(m - n_to_win + 1):
            for j in range(m):
                for player in [1, -1]:
                    if np.all(self.board[i:i+n_to_win, j] == player):
                        return player
        
        # Check diagonals (top-left to bottom-right)
        for i in range(m - n_to_win + 1):
            for j in range(m - n_to_win + 1):
                for player in [1, -1]:
                    if all(self.board[i+k, j+k] == player for k in range(n_to_win)):
                        return player
        
        # Check diagonals (top-right to bottom-left)
        for i in range(m - n_to_win + 1):
            for j in range(n_to_win - 1, m):
                for player in [1, -1]:
                    if all(self.board[i+k, j-k] == player for k in range(n_to_win)):
                        return player
        
        # No winner
        return 0
    
    def render(self) -> None:
        """
        Render the current state of the game.
        """
        print(self.board)
        if self.done:
            if self.winner == 0:
                print("It's a draw!")
            else:
                print(f"Player {self.winner} wins!")
        else:
            print(f"Player {self.player}'s turn.")

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