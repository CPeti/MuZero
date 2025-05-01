import numpy as np
from games import TicTacToe, GameSimulator
from neural import NNM
from mcts import Node, UMCTS
from typing import List, Optional
import torch

import configs

class EpisodeData:
    def __init__(self):
        self.real_game_states : List[np.ndarray] = []
        self.values : List[float] = []
        self.policies : List[np.ndarray] = [] 
        self.actions : List[int] = []
        self.rewards : List[float] = []

    def store(self, real_game_state, value, policy, action, reward):
        self.real_game_states.append(real_game_state)
        self.values.append(value)
        self.policies.append(policy)
        self.actions.append(action)
        self.rewards.append(reward)

class EpisodeBuffer:
    def __init__(self, capacity: int = 3000):
        self.buffer: List[EpisodeData] = []
        self.capacity = capacity
    @property
    def size(self):
        return len(self.buffer)

    def clear(self):
        """
        Clear the buffer.
        """
        self.buffer: List[EpisodeData] = []

    def store(self, epidata: EpisodeData) -> None:
        """
        Store the episode data in the buffer.
        
        Args:
            epidata: The episode data to store.
        """
        self.buffer.append(epidata)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size: int, window: int = 7) -> List[EpisodeData]:
        """
        Sample a batch of episode data from the buffer.
        
        Args:
            batch_size: The size of the batch to sample.
        
        Returns:
            A list of sampled episode data.
        """
        batch = []
        for i in range(batch_size):
            target_values, target_rewards, target_policies, actions = [], [], [], []
            # choose an episode
            episode = np.random.choice(self.buffer)
            # choose a starting state
            start_idx = np.random.choice(len(episode.real_game_states) - window)
            # choose a window of states
            end_idx = start_idx + window + 1

            states = episode.real_game_states[start_idx:end_idx]

            for cur_idx in range(start_idx, end_idx):
                if cur_idx < len(episode.values):
                    target_values.append(episode.values[cur_idx])
                    target_rewards.append(episode.rewards[cur_idx])
                    target_policies.append(episode.policies[cur_idx])
                    actions.append(episode.actions[cur_idx])
                elif cur_idx == len(episode.values):
                    target_values.append(episode.rewards[-1])
                    target_rewards.append(0)
                    # uniform distribution for the policy
                    target_policies.append(np.ones(episode.policies[0].shape) / episode.policies[0].shape[0])
                    actions.append(np.random.choice(episode.policies[0].shape[0]))
                else:
                    # state past the end of the episode
                    target_values.append(episode.rewards[-1]) # TODO: check if this is correct
                    target_rewards.append(episode.rewards[-1])
                    # uniform distribution for the policy
                    target_policies.append(np.ones(episode.policies[0].shape) / episode.policies[0].shape[0])
                    actions.append(np.random.choice(episode.policies[0].shape[0]))
            # remove last action and reward
            # these will not be evaluated in the loss function
            target_rewards.pop(-1)
            actions.pop(-1)

            batch.append({
                'states': np.array(states),
                'actions': np.array(actions),
                'values': np.array(target_values),
                'policies': np.array(target_policies),
                'rewards': np.array(target_rewards)
            })
        return batch

class RLM:
    def __init__(self, gm: GameSimulator = None, nnm: NNM = None):
        self.episode_buffer = EpisodeBuffer(capacity=512)
        self.gm = gm if gm else TicTacToe()
        self.nnm = nnm if nnm else NNM(self.gm.get_observation().flatten().shape[0], self.gm.get_action_space_size())
        self.umcts = UMCTS(self.nnm, self.gm)

        # Config params
        self.num_episodes = configs.num_episodes # number of episodes to run
        self.max_moves = configs.max_moves # max moves per episode
        self.num_simulations = configs.num_simulations # number of MCTS simulations per move
        self.train_interval = configs.train_interval
        self.batch_size = configs.batch_size
        self.save_interval = configs.save_interval
        self.sample_window = configs.sample_window

    def episode_loop(self) -> None:
        """
        Run the episode loop for the specified number of episodes.
        
        Args:
            num_episodes: The number of episodes to run.
        """
        print(f"Episode 1/{self.num_episodes}")
        self.episode_buffer.clear()
        for episode in range(self.num_episodes):
            with torch.no_grad():
                self.run_episode()
            if (episode + 1) % self.train_interval == 0:
                print(f"Episode {episode + 1}/{self.num_episodes}")
                self.nnm.train_networks(self.episode_buffer, self.batch_size, sample_window=self.sample_window)
            
            if (episode + 1) % self.save_interval == 0:
                self.nnm.save_model(f".\\models\\model_{episode + 1}.pth")

    def run_episode(self) -> None:
        """
        Run a single episode.
        """
        epidata = EpisodeData()
        # get real state
        real_game_state = self.gm.reset()
        epidata.real_game_states.append(real_game_state)

        for move in range(self.max_moves):
            # get abstract_state
            abstract_state = self.nnm.representation(real_game_state)
            # initialize u-MCTS root node
            root = Node(abstract_state=abstract_state)
            legal_actions = self.gm.get_legal_actions()
            self.umcts.expand_node(root, legal_actions)
            # run u-MCTS search
            self.umcts.search(root, self.num_simulations, to_play=self.gm.player)
            # get visit counts distribution
            policy = self.umcts.get_policy(root)
            
            policy_for_all_actions = np.zeros(self.gm.get_action_space_size())
            policy_for_all_actions[legal_actions] = policy

            epidata.values.append(root.Q.item())
            epidata.policies.append(policy_for_all_actions)

            # sample action from visit counts distribution
            action_idx = np.random.choice(len(policy), p=policy)
            action = legal_actions[action_idx]

            # take action in the game
            real_game_state, reward, terminal = self.gm.step(action)

            epidata.actions.append(action)
            epidata.rewards.append(reward)
            epidata.real_game_states.append(real_game_state)
            if terminal:
                break
        
        self.episode_buffer.store(epidata)

    def get_actor_move(self, real_game_state: np.ndarray, legal_actions) -> int:
        """
        Get the action to take in the game using the actor network.
        
        Args:
            real_game_state: The current state of the game.
        
        Returns:
            The action to take in the game.
        """
        with torch.no_grad():
            abstract_state = self.nnm.representation(real_game_state)
            root = Node(abstract_state=abstract_state)
            self.umcts.expand_node(root, legal_actions)
            self.umcts.search(root, self.num_simulations)
            policy = self.umcts.get_policy(root)
            action_idx = np.random.choice(len(policy), p=policy)
            action = legal_actions[action_idx]
        return action
