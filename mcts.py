import numpy as np
from typing import List, Optional, Tuple
from neural import NNM
from games import GameSimulator
from torch import multinomial
import torch
import configs

class Node:
    def __init__(self, parent: Optional['Node'] = None, action: Optional[int] = None, 
                 abstract_state: Optional[np.ndarray] = None, reward: Optional[float] = 0.0):
        """
        Initialize a node in the MCTS tree.
        
        Args:
            parent: The parent node.
            action: The action taken to reach this node.
        """
        self.parent = parent
        self.action = action
        self.to_play = 1
        self.children: List[Node] = []
        self.visits = 0
        self.value_sum = 0.0

        # Initialized when the node is expanded unless root node
        self.abstract_state = abstract_state
        self.reward = reward

        # Config params
        self.C = configs.UCT_C # Exploration constant

    @property
    def Q(self):
        """
        Calculate the average value of the node.
        """
        return self.value_sum / self.visits if self.visits > 0 else 0.0
    
    @property
    def exp_bonus(self):
        """
        Calculate the exploration bonus for the node.
        C x sqrt(log(parent.visits) / (1 + visits)).
        """
        if self.parent is None:
            return 0.0
        return self.C * np.sqrt(np.log(self.parent.visits + 1) / (self.visits + 1))
    
    @property
    def value(self):
        """
        Calculate the value of the node.
        Q + exploration bonus.
        """
        return self.Q + self.exp_bonus
    
    def update_value_sum(self, rewards: List[float], gamma: float = 1.0, to_play: int = 1) -> None:
        """
        Update the value sum of the node based on the rewards and discount factor.
        
        Args:
            rewards: The list of rewards from the leaf node to this node.
            gamma: The discount factor.
        """
        total_reward = 0.0
        for i, r in enumerate(reversed(rewards)):
            total_reward += (gamma ** i) * r

        self.value_sum += total_reward #if self.to_play == to_play else -total_reward
    



class UMCTS:
    def __init__(self, nnm: NNM, gm: GameSimulator):
        """
        Initialize the UMCTS algorithm with a neural network model.
        
        Args:
            nnm: The neural network manager to be used for the MCTS.
            gm: The game manager to be used for the MCTS.
        """
        self.nnm = nnm
        self.gm = gm

        # Config params
        self.discount = configs.discount
        self.max_depth = configs.max_depth # TODO: does nothing for now

    def find_leaf(self, node: Node, to_play: int = 1) -> Tuple[Node, int]:
        """
        Find the leaf node of the given node.
        """
        depth = 0
        while node.children:
            if node.to_play == to_play:
                node = max(node.children, key=lambda n: n.Q + n.exp_bonus)
            else:
                node = min(node.children, key=lambda n: n.Q - n.exp_bonus)
                raise NotImplementedError("Two player not implemented yet")
            depth += 1
        return node, depth
    
    def expand_node(self, node: Node, actions: List[int]) -> None:
        
        """
        Expand the node by adding child nodes for each action and perform a rollout using the neural network model.
        
        Args:
            node: The node to expand.
            actions: The list of possible actions from this node.
        """
        for action in actions:
            with torch.no_grad():
                child_abstract_state, reward = self.nnm.dynamics(node.abstract_state, action)
            child_node = Node(parent=node, action=action, abstract_state=child_abstract_state, reward=reward)
            if self.gm.two_player:
                child_node.to_play = -node.to_play
                raise NotImplementedError("Two player not implemented yet")
            node.children.append(child_node)
    
    def backpropagate(self, node: Node, accum_reward: list, to_play: int = 1) -> None:
        """
        Backpropagate the value from the leaf node to the root node.
        
        Args:
            node: The leaf node to backpropagate from.
            value: The value to backpropagate.
        """
        while node is not None:
            node.visits += 1
            node.update_value_sum(accum_reward, self.discount, to_play)
            if node.to_play == to_play:
                accum_reward.append(node.reward)
            else:
                accum_reward.append(-node.reward)
                raise NotImplementedError("Two player not implemented yet")
            node = node.parent

    def search(self, root: Node, num_simulations: int, to_play: int = 1) -> None:
        """
        Perform the MCTS search from the root node.
        
        Args:
            root: The root node to start the search from.
            num_simulations: The number of simulations to perform.
        """
        for _ in range(num_simulations):
            leaf, depth = self.find_leaf(root, to_play=to_play)
            v_to_play = leaf.to_play

            actions = self.gm.get_all_actions()
            self.expand_node(leaf, actions)
            child = np.random.choice(leaf.children)

            #rollout_depth = self.max_depth - depth
            rollout_depth = 1
            accum_reward = self.rollout(child, rollout_depth)
            self.backpropagate(child, accum_reward, to_play=v_to_play)

    def rollout(self, node: Node, depth: int) -> List[float]:
        with torch.no_grad():
            abstaract_state = node.abstract_state
            accum_reward = []
            for _ in range(depth):
                policy, value = self.nnm.prediction(abstaract_state)
                action = multinomial(policy, num_samples=1).item()
                abstaract_state, reward = self.nnm.dynamics(abstaract_state, action)
                accum_reward.append(reward)
            policy, value = self.nnm.prediction(abstaract_state)
            accum_reward.append(value)
        return accum_reward


    def get_policy(self, root: Node) -> np.ndarray:
        """
        Get normalized visit counts for the root node.
        
        Args:
            root: The root node to get the action from.
            inference: Whether to use inference or not.
        
        Returns:
            Array of visit counts normalized by the number of visits.
        """
        if root.visits == 0:
            return None

        visit_counts = np.array([child.visits for child in root.children])

        visit_counts = visit_counts / root.visits

        return visit_counts


    
    

    
