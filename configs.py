from typing import List, Dict, Any, Tuple 

num_episodes = 100000 # number of episodes to run
max_moves = 15 # max moves per episode
num_simulations = 200 # number of MCTS simulations per move
train_interval = 5
batch_size = 64
save_interval = 1000
sample_window = 8 # number of consecutive to sample from the episode buffer

discount = 0.95
max_depth = 15 # TODO: does nothing for now

UCT_C = 1.412 # sqrt(2)

abstract_state_size: int = 8

class RepresentationNetworkConfig:
    """
    Configuration for the representation network.
    """
    def __init__(self, input_size: int):
        self.input_size = input_size
        self.fc1 = 16
        self.layers: List[Dict[str, Any]] = [
            {"type": "Flatten"},
            {"type": "Linear", "in_features": input_size, "out_features": self.fc1},
            {"type": "ELU"},
            {"type": "Linear", "in_features": self.fc1, "out_features": abstract_state_size},
        ]

class DynamicsNetworkConfig:
    """
    Configuration for the dynamics network.
    """
    def __init__(self, action_space_size: int):
        self.fc_state = 16
        self.fc_reward = 16
        self.state_layers: List[Dict[str, Any]] = [
            {"type": "Linear", "in_features": abstract_state_size + action_space_size, "out_features": self.fc_state},
            {"type": "ELU"},
            {"type": "Linear", "in_features": self.fc_state, "out_features": abstract_state_size},
        ]
        self.reward_layers: List[Dict[str, Any]] = [
            {"type": "Linear", "in_features": abstract_state_size, "out_features": self.fc_reward},
            {"type": "ELU"},
            {"type": "Linear", "in_features": self.fc_reward, "out_features": 1},
        ]

class PredictionNetworkConfig:
    """
    Configuration for the dynamics network.
    """
    def __init__(self, action_space_size: int):
        self.fc_policy = 16
        self.fc_value = 16
        self.policy_layers: List[Dict[str, Any]] = [
            {"type": "Linear", "in_features": abstract_state_size, "out_features": self.fc_policy},
            {"type": "ELU"},
            {"type": "Linear", "in_features": self.fc_policy, "out_features": action_space_size},
        ]
        self.value_layers: List[Dict[str, Any]] = [
            {"type": "Linear", "in_features": abstract_state_size, "out_features": self.fc_value},
            {"type": "ELU"},
            {"type": "Linear", "in_features": self.fc_value, "out_features": 1},
        ]
